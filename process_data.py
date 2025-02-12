"""
HSR Data Preprocessing Pipeline

This pipeline implements several improvements over the original pipeline used in the HSR paper:

1. Human Segmentation:
   - Replaced RVM + SAM with Grounded-SAM2
   - Advantages: Faster inference, no manual prompting needed

2. Monocular Scene Understanding:
   - Replaced Omnidata with Metric3Dv2
   - Advantage: Significantly reduced inference time

3. Scene Normalization:
   - Removed dependency on Blenderangelo
   - Now centers scene using detected human mesh centers
   - Normalizes scene scale using SMPL body scale as reference
   - Result: Simplified pipeline with consistent metric scale
"""

import argparse
import subprocess
from pathlib import Path

if __name__ == "__main__":

    SAM2_PYTHON_PATH = "/home/lixin/miniconda3/envs/sam21/bin/python"
    METRIC3D_PYTHON_PATH = "/home/lixin/miniconda3/envs/metric3d/bin/python"
    OPENPOSE_PYTHON_PATH = "/usr/bin/python3"
    OPENPOSE_MODEL_PATH = "/home/lixin/softwares/openpose/models/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--gender", type=str, choices=["male", "female"])

    parser.add_argument("--input_path", type=Path)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=1000000)
    parser.add_argument("--image_resize_factor", type=int, default=2)

    # we use a simpler camera model to avoid unstable image undistortion
    parser.add_argument(
        "--camera_model",
        type=str,
        choices=["SIMPLE_PINHOLE", "PINHOLE"],
        default="SIMPLE_PINHOLE",
    )
    # default recommendation: hloc + superpoint + lightglue + sequential matching
    parser.add_argument(
        "--sfm_tool", type=str, choices=["colmap", "hloc", "record3d"], default="hloc"
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["sift", "superpoint_aachen", "r2d2", "disk", "loftr"],
        default="superpoint_aachen",
    )
    parser.add_argument(
        "--matcher_type",
        type=str,
        choices=["NN", "NN-ratio", "superglue", "superpoint+lightglue", "disk+lightglue", "loftr"],
        default="superpoint+lightglue",
    )
    parser.add_argument(
        "--matching_method",
        type=str,
        choices=["exhaustive", "sequential", "poses"],
        default="sequential",
    )

    # in case when automatic scale estimation fails
    # specify a scale manually
    parser.add_argument("--manual_scale", action="store_true")

    parser.add_argument("--steps", type=int, nargs="+", default=[])
    args = parser.parse_args()

    data_dir = str(args.data_dir.resolve())
    gender = args.gender

    input_path = args.input_path
    if input_path is not None:
        input_path = str(input_path.resolve())
    window_size = str(args.window_size)
    frame_start = str(args.frame_start)
    frame_end = str(args.frame_end)
    image_resize_factor = str(args.image_resize_factor)

    camera_model = args.camera_model
    sfm_tool = args.sfm_tool
    feature_type = args.feature_type
    matcher_type = args.matcher_type
    matching_method = args.matching_method

    manual_scale = args.manual_scale
    steps = args.steps

    sparse_dir = f"{data_dir}/camera_poses/{sfm_tool}-{feature_type}-{matching_method}-{matcher_type}/sparse/0"

    # extract frames
    if 0 in steps:
        cmd = [
            "python",
            "select_frames.py",
            "--data_dir",
            data_dir,
            "--input_path",
            input_path,
            "--window_size",
            window_size,
            "--frame_start",
            frame_start,
            "--frame_end",
            frame_end,
            "--image_resize_factor",
            image_resize_factor,
        ]
        subprocess.run(cmd)

    # use rvm to extract masks for feature extraction
    if 1 in steps:
        cmd = [
            SAM2_PYTHON_PATH,
            "create_masks_with_sam2.py",
            "--data_dir",
            data_dir,
            "--text",
            "human.",
        ]
        subprocess.run(cmd)

    # camera localization
    if 2 in steps:
        cmd = [
            "python",
            "estimate_camera_poses.py",
            "--data_dir",
            data_dir,
            "--camera_model",
            camera_model,
            "--sfm_tool",
            sfm_tool,
            "--feature_type",
            feature_type,
            "--matcher_type",
            matcher_type,
            "--matching_method",
            matching_method,
            "--image_resize_factor",
            image_resize_factor,
        ]
        subprocess.run(cmd)

    if 3 in steps:
        cmd = [
            METRIC3D_PYTHON_PATH,
            "extract_monocular_cues_with_Metric3D.py",
            "--data_dir",
            data_dir,
        ]
        subprocess.run(cmd)

    # estimate human pose with romp
    if 4 in steps:
        cmd = ["python", "run_romp.py", "--data_dir", data_dir, "--gender", gender]
        subprocess.run(cmd)

    # 2d keypoints
    if 5 in steps:
        cmd = [
            OPENPOSE_PYTHON_PATH,
            "run_openpose.py",
            "--data_dir",
            data_dir,
            "--model_path",
            OPENPOSE_MODEL_PATH,
        ]
        subprocess.run(cmd)

    if 6 in steps:
        cmd = [
            "python",
            "refine_romp.py",
            "--data_dir",
            data_dir,
            "--gender",
            gender,
            "--sparse_dir",
            sparse_dir,
        ]
        subprocess.run(cmd)

    if 7 in steps:
        cmd = [
            "python",
            "align_human_scene.py",
            "--data_dir",
            data_dir,
            "--sparse_dir",
            sparse_dir,
            "--gender",
            gender,
        ]
        if manual_scale:
            cmd.append("--manual_scale")
        subprocess.run(cmd)

    if 8 in steps:
        cmd = [
            "python",
            "prepare_dataset.py",
            "--data_dir",
            data_dir,
        ]
        subprocess.run(cmd)
