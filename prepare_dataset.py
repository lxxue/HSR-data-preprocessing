"""
HSR Dataset Preparation

This script converts preprocessed data into the format required by HSR. It processes and 
organizes images, masks, depth maps, SMPL parameters and camera poses into a standardized 
structure.

Arguments:
    --data_dir: output directory for the processed data

Output Structure:
    data_dir/
    └── processed/
        ├── image/                # input images
        ├── sam_mask/             # human segmentation masks 
        ├── depth/                # monocular depth maps
        ├── normal/               # monocular normal maps
        ├── cameras.npz           # camera parameters
        ├── cameras_normalize.npz # camera parameters with dummy scale matrices
        ├── mean_shape.npy        # scale and mean SMPL shape parameters
        ├── poses.npy             # SMPL pose parameters
        ├── normalize_trans.npy   # SMPL translations
        ├── intrinsic.npy         # camera intrinsics
        ├── c2ws.npy              # camera-to-world matrices
        └── scene_pcd.ply         # scene point cloud
"""

import argparse
import glob
import pickle as pkl
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_normalized_data(input_dir):
    intrinsic = np.load(input_dir / "intrinsic.npy")
    with open(input_dir / "c2ws.pkl", "rb") as f:
        c2ws = pkl.load(f)
    with open(input_dir / "aligned_smpl.pkl", "rb") as f:
        aligned_smpl = pkl.load(f)
    return intrinsic, c2ws, aligned_smpl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir

    print("Convert the processed data to the format required by HSR")

    img_dir = data_dir / "images" / "selected_frames"
    sam_mask_dir = data_dir / "masks" / "sam2_human"
    depth_dir = data_dir / "monocular_cues" / "metric3d" / "depth"
    normal_dir = data_dir / "monocular_cues" / "metric3d" / "normal"

    save_dir = data_dir / "processed"
    save_dir.mkdir(exist_ok=True)
    (save_dir / "image").mkdir(parents=False, exist_ok=True)
    (save_dir / "sam_mask").mkdir(parents=False, exist_ok=True)
    (save_dir / "depth").mkdir(parents=False, exist_ok=True)
    (save_dir / "normal").mkdir(parents=False, exist_ok=True)

    img_pattern = re.compile(r"(\d+).(jpg|jpeg|png)$", re.IGNORECASE)
    img_paths = [file for file in img_dir.glob("*") if img_pattern.match(file.name)]
    img_paths = sorted(img_paths)
    sam_mask_paths = sorted(glob.glob(f"{sam_mask_dir}/*.jpg"))
    depth_paths = sorted(glob.glob(f"{depth_dir}/*.npy"))
    normal_paths = sorted(glob.glob(f"{normal_dir}/*.npy"))

    intrinsic, c2ws, aligned_smpl = load_normalized_data(data_dir / "normalized")
    K = np.eye(4, dtype=np.float32)
    K[:3, :3] = intrinsic

    input_img = cv2.imread(img_paths[0])
    img_h, img_w = input_img.shape[:2]

    output_trans = []
    output_pose = []
    output_scale = []
    output_shape = []
    output_P = {}
    c2w_list = []
    suffix = next(iter(c2ws.keys())).split(".")[1]
    for idx, img_path in enumerate(tqdm(img_paths)):
        img_path = str(img_path)
        assert idx == int(img_path.split("/")[-1].split(".")[0])
        img = cv2.imread(img_path)
        sam_mask = cv2.imread(sam_mask_paths[idx])
        depth = np.load(depth_paths[idx])
        normal = np.load(normal_paths[idx])

        cv2.imwrite(str(save_dir / "image" / f"{idx:04d}.png"), img)
        cv2.imwrite(str(save_dir / "sam_mask" / f"{idx:04d}.png"), sam_mask)
        np.save(save_dir / "depth" / f"{idx:04d}.npy", depth)
        np.save(save_dir / "normal" / f"{idx:04d}.npy", normal)

        smpl_pose = aligned_smpl["pose"][idx]
        smpl_trans = aligned_smpl["trans"][idx].astype(np.float32)
        smpl_scale = aligned_smpl["scale"][idx]
        smpl_shape = aligned_smpl["shape"][idx]

        c2w = c2ws[f"{idx:04d}.{suffix}"]

        w2c = np.linalg.inv(c2w)
        P = K @ w2c
        output_trans.append(smpl_trans)
        output_pose.append(smpl_pose)
        output_scale.append(smpl_scale)
        output_shape.append(smpl_shape)
        output_P[f"cam_{idx}"] = P.astype(np.float32)
        c2w_list.append(c2w)

    mean_shape = np.array(output_shape).mean(axis=0)
    # use smaller scale since we tend to overestimate the scale with naked smpl
    mean_scale = np.quantile(np.array(output_scale), 0.1, axis=0)
    # mean_scale = np.array(output_scale).mean(axis=0)
    # print("mean scale:", mean_scale)
    mean_shape = np.concatenate([mean_scale, mean_shape], axis=0)
    np.save(save_dir / "mean_shape.npy", mean_shape)
    np.save(save_dir / "poses.npy", np.array(output_pose))
    np.save(save_dir / "normalize_trans.npy", np.array(output_trans))
    np.save(save_dir / "intrinsic.npy", K.astype(np.float32))
    np.save(save_dir / "c2ws.npy", np.array(c2w_list, dtype=np.float32))
    np.savez(save_dir / "cameras.npz", **output_P)
    cameras_new = {}
    for i in range(len(output_P)):
        # we have a dummpy scale matrix here as our camera is already normalized in the previous step
        cameras_new[f"scale_mat_{i}"] = np.eye(4, dtype=np.float32)
        cameras_new[f"world_mat_{i}"] = output_P[f"cam_{i}"]
    np.savez(save_dir / "cameras_normalize.npz", **cameras_new)
    # copy scene point cloud to the processed folder
    scene_pcd = data_dir / "normalized" / "scene_pcd.ply"
    shutil.copy(scene_pcd, save_dir / "scene_pcd.ply")
    print("All steps finished")
