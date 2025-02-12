"""
Monocular Depth and Normal Estimation using Metric3Dv2

Arguments:
    --data_dir: output directory for the processed data 

Output Structure:
    data_dir/
    └── monocular_cues/
        └── metric3d/
            ├── depth/          # depth maps (.jpg and .npy)
            └── normal/         # normal maps (.jpg and .npy)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


def inference_one_image(model, rgb_origin, img_fname, output_dir):
    np_fname = img_fname.replace(".jpg", ".npy")
    #### ajust input size to fit pretrained model
    # keep ratio resize
    input_size = (616, 1064)  # for vit model
    # input_size = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    # TODO: do we need to use intrinsics from colmap for metric depth?
    # intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
    # # remember to scale intrinsic, hold depth
    # intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]

    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(
        rgb,
        pad_h_half,
        pad_h - pad_h_half,
        pad_w_half,
        pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=padding,
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    #### normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()

    ###################### canonical camera space ######################
    # inference
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({"input": rgb})

    # un pad
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[
        pad_info[0] : pred_depth.shape[0] - pad_info[1],
        pad_info[2] : pred_depth.shape[1] - pad_info[3],
    ]

    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None, :, :], rgb_origin.shape[:2], mode="bilinear"
    ).squeeze()

    normalized_pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
    normalized_pred_depth = normalized_pred_depth.cpu().numpy()
    # keep size of (H, W) for consistency with HSR code
    # normalized_pred_depth = rearrange(normalized_pred_depth, "h w -> h w 1")
    depth_dir = output_dir / "depth"
    plt.imsave(depth_dir / img_fname, normalized_pred_depth)
    np.save(depth_dir / np_fname, normalized_pred_depth)

    ###################### canonical camera space ######################

    # TODO: do we need to use intrinsics from colmap for metric depth?
    # #### de-canonical transform
    # canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
    # pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
    # pred_depth = torch.clamp(pred_depth, 0, 300)

    #### you can now do anything with the metric depth
    # such as evaluate predicted depth
    # if depth_file is not None:
    #     gt_depth = cv2.imread(depth_file, -1)
    #     gt_depth = gt_depth / gt_depth_scale
    #     gt_depth = torch.from_numpy(gt_depth).float().cuda()
    #     assert gt_depth.shape == pred_depth.shape

    #     mask = (gt_depth > 1e-8)
    #     abs_rel_err = (torch.abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask]).mean()
    #     print('abs_rel_err:', abs_rel_err.item())

    #### normal are also available
    if "prediction_normal" in output_dict:  # only available for Metric3Dv2, i.e. vit model
        pred_normal = output_dict["prediction_normal"][:, :3, :, :]
        normal_confidence = output_dict["prediction_normal"][
            :, 3, :, :
        ]  # see https://arxiv.org/abs/2109.09881 for details
        # un pad and resize to some size if needed
        pred_normal = pred_normal.squeeze()
        pred_normal = pred_normal[
            :,
            pad_info[0] : pred_normal.shape[1] - pad_info[1],
            pad_info[2] : pred_normal.shape[2] - pad_info[3],
        ]

        # upsample to original size
        pred_normal = torch.nn.functional.interpolate(
            pred_normal[None, :, :, :], rgb_origin.shape[:2], mode="bilinear"
        ).squeeze()
        pred_normal = pred_normal / torch.norm(pred_normal, dim=0, keepdim=True)
        # normal_norm = np.linalg.norm(pred_normal.cpu().numpy(), axis=0)

        pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
        pred_normal_vis = (pred_normal_vis + 1) / 2
        normal_dir = output_dir / "normal"
        plt.imsave(normal_dir / img_fname, pred_normal_vis)
        np.save(normal_dir / np_fname, pred_normal_vis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    output_dir = data_dir / "monocular_cues" / "metric3d"
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(exist_ok=True)
    normal_dir = output_dir / "normal"
    normal_dir.mkdir(exist_ok=True)

    img_dir = data_dir / "images" / "selected_frames"
    img_fnames = sorted(img_dir.glob("*.jpg"))

    model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_giant2", pretrain=True)
    model.cuda().eval()
    for img_fname in tqdm(img_fnames):
        img = cv2.imread(str(img_fname))[:, :, ::-1]
        inference_one_image(model, img, img_fname.name, output_dir)
