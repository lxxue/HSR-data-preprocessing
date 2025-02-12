"""
Human Segmentation with SAM2 and Grounding-DINO

This script detects and segments human subjects from image sequences using SAM2 and 
Grounding-DINO models. It creates binary masks for COLMAP processing by removing dynamic 
human subjects from the scene.

Arguments:
    --data_dir: output directory for the processed data 
    --text: text prompt for detection (default: "human.")

Output Structure:
    data_dir/
    └── masks/
        ├── sam2_human/             # Binary human masks
        ├── sam2_vis_human/         # Visualization of human masks
        ├── sam2_vis_human_dilated/ # Visualization of dilated human masks
        └── colmap/                 # COLMAP-ready masks (inverse of dilated human masks)
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append("third_party/Grounded-SAM-2")
# use as submodule later on

import cv2
import numpy as np
import supervision as sv
import torch
from grounding_dino.groundingdino.util.inference import load_model
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

# adapted from https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_tracking_demo.py


# mask out human pixels to run colmap
def create_colmap_mask(mask_human_dir, mask_colmap_dir):
    print("Creating masks for colmap")
    human_mask_fnames = sorted(mask_human_dir.glob("*.jpg"))

    mask_dilated_human_dir = mask_colmap_dir.parent / "sam2_vis_human_dilated"
    mask_dilated_human_dir.mkdir(exist_ok=True)
    image_dir = mask_colmap_dir.parents[1] / "images" / "selected_frames"

    for i in range(len(human_mask_fnames)):
        human_mask_fname = human_mask_fnames[i]

        # H, W, 3
        human_mask = cv2.imread(str(human_mask_fname))

        # dilate the human mask more aggressively
        # as we don't want feature point around the human
        dilated_human_mask = cv2.dilate(human_mask, np.ones((5, 5), np.uint8), iterations=5)
        dilated_human_mask_vis = cv2.imread(str(image_dir / human_mask_fname.name))
        dilated_human_mask_vis[dilated_human_mask < 127] = 0
        cv2.imwrite(str(mask_dilated_human_dir / human_mask_fname.name), dilated_human_mask_vis)

        dilated_mask_scene = 255 - dilated_human_mask
        cv2.imwrite(str(mask_colmap_dir / (human_mask_fname.name + ".png")), dilated_mask_scene)


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path, required=True)
parser.add_argument(
    "--text", type=str, default="human.", help="text queries need to be lowercased + end with a dot"
)
args = parser.parse_args()
data_dir = args.data_dir
text = args.text

mask_colmap_dir = data_dir / "masks" / "colmap"
mask_human_dir = data_dir / "masks" / "sam2_human"
mask_vis_dir = data_dir / "masks" / "sam2_vis_human"
mask_colmap_dir.mkdir(parents=True, exist_ok=True)
mask_human_dir.mkdir(exist_ok=True)
mask_vis_dir.mkdir(exist_ok=True)

print("Generating human masks")

"""
Hyper parameters
"""
# VERY important: text queries need to be lowercased + end with a dot
SAM2_CHECKPOINT = "third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
PROMPT_TYPE_FOR_VIDEO = "box"  # or "point"
assert PROMPT_TYPE_FOR_VIDEO in [
    "point",
    "box",
    "mask",
], "SAM 2 video predictor only support point/box/mask prompt"


"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_CONFIG
video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# init grounding dino model from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = data_dir / "images" / "selected_frames"
# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=str(video_dir))

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
"""

# prompt grounding dino to get the box coordinates on specific frame
img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
image = Image.open(img_path)

# run Grounding DINO on the image
inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.25,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]],
)

# prompt SAM image predictor to get the mask for the object
image_predictor.set_image(np.array(image.convert("RGB")))

# process the detection results
input_boxes = results[0]["boxes"].cpu().numpy()
OBJECTS = results[0]["labels"]

# prompt SAM 2 image predictor to get the mask for the object
masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

# convert the mask shape to (n, H, W)
if masks.ndim == 3:
    masks = masks[None]
    scores = scores[None]
    logits = logits[None]
elif masks.ndim == 4:
    masks = masks.squeeze(1)


"""
Step 3: Register each object's positive points to video predictor with seperate add_new_points call
"""
# If you are using point prompts, we uniformly sample positive points based on the mask
if PROMPT_TYPE_FOR_VIDEO == "point":
    # sample the positive points from mask for each objects
    all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

    for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
        labels = np.ones((points.shape[0]), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
        )
# Using box prompt
elif PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
# Using mask prompt is a more straightforward way
elif PROMPT_TYPE_FOR_VIDEO == "mask":
    for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
        labels = np.ones((1), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
            inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=object_id, mask=mask
        )
else:
    raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
    inference_state
):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

"""
Step 5: Visualize the segment results across the video and save them
"""

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))

    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
        mask=masks,  # (n, h, w)
        class_id=np.array(object_ids, dtype=np.int32),
    )
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids]
    )
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(mask_vis_dir, f"{frame_idx:04d}.jpg"), annotated_frame)
    mask_all = np.zeros_like(masks[0])
    for mask in masks:
        mask_all = np.logical_or(mask_all, mask)

    mask_all = mask_all.astype(np.uint8) * 255
    cv2.imwrite(os.path.join(mask_human_dir, f"{frame_idx:04d}.jpg"), mask_all)
create_colmap_mask(mask_human_dir, mask_colmap_dir)
