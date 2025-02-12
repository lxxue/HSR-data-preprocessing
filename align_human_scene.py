"""
Human-Scene Alignment

This script aligns SMPL meshes with a reconstructed sparse scene through the following steps:
1. Detects ground plane in the sparse reconstruction
2. Computes scale and rigid transformation to align SMPL mesh with scene
3. Normalizes both scene and SMPL mesh to unit human scale for metric reconstruction
4. Supports manual scale input when automatic ground plane detection fails

Arguments:
    --data_dir: output directory for the processed data
    --sparse_dir: path to COLMAP sparse reconstruction
    --gender: SMPL model gender (male/female) 
    --manual_scale: flag to manually specify scale factor

Output Structure:
    data_dir/
    └── normalized/
        ├── scene_pcd.ply                    # normalized scene point cloud
        ├── aligned_smpl.pkl                 # aligned SMPL parameters
        ├── c2ws.pkl                         # normalized camera poses
        └── intrinsic.npy                    # camera intrinsics
"""

import argparse
import pickle as pkl
from pathlib import Path

import numpy as np
import open3d as o3d

# adapted from https://github.com/apple/ml-neuman/blob/main/preprocess/export_alignment.py
from utils.align_human_scene_utils import (
    convert_smpl_to_world,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_point_cloud_binary,
    read_smpl,
    solve_transformation,
    transform_verts_with_scale,
    visualize_smpl_and_scene,
)


def center_and_scale_human_and_scene(scene_pcd, c2ws, new_smpls, scale, center):
    scene_pcd.translate(-center, relative=True)
    scene_pcd.scale(1.0 / scale, center=np.zeros((3, 1)))

    dist_to_origins = []
    for image_name in c2ws:
        c2ws[image_name][:3, 3] -= center
        c2ws[image_name][:3, 3] /= scale
        dist_to_origin = np.linalg.norm(c2ws[image_name][:3, 3])
        dist_to_origins.append(dist_to_origin)
        # check if the cameras are whithin sphere of raidus 3.0 used in the model

    dist_to_origins = np.array(dist_to_origins)
    # print("camera center dists to origin:", dist_to_origins)
    print("Camera center distance to origin statistics:")
    print("mean: ", dist_to_origins.mean())
    print("std: ", dist_to_origins.std())
    print("max: ", dist_to_origins.max())
    print("min: ", dist_to_origins.min())

    for i in range(len(new_smpls["scale"])):
        new_smpls["scale"][i] /= scale
        new_smpls["trans"][i] = (new_smpls["trans"][i] * scale - center) / scale


def align_human_scene(data_dir, sparse_dir, gender, manual_scale):
    scene_pcd = read_point_cloud_binary(sparse_dir)
    # legacy mode point cloud runs significantly faster
    scene_pcd = scene_pcd.to_legacy()
    c2ws = read_extrinsics_binary(sparse_dir)
    raw_smpls = read_smpl(data_dir / "smpl" / "refined_ROMP")
    num_frames = len(c2ws)
    new_smpls = {"scale": [], "pose": [], "trans": [], "shape": []}

    if manual_scale:
        # specify the scale manually in a interactive way
        success = False
        while not success:
            new_smpls = {"scale": [], "pose": [], "trans": [], "shape": []}
            scale = float(input("Enter the scale: "))
            suffix = next(iter(c2ws.keys())).split(".")[1]
            translations = []
            for i in range(num_frames):
                c2w = c2ws[f"{i:04d}.{suffix}"]
                transformed_smpl_verts = transform_verts_with_scale(
                    raw_smpls["verts"][i], c2w, scale
                )
                _, translation_i, _, _ = convert_smpl_to_world(
                    raw_smpls, i, scale, c2w, transformed_smpl_verts, new_smpls, gender
                )
                translations.append(translation_i)
            background = [scene_pcd]
            mean_translation = np.mean(translations, axis=0)
            visualize_smpl_and_scene(background, new_smpls, gender, sphere_radius=-1.0)
            success = input("Is the alignment good enough (y/n)? ") == "y"

        center_and_scale_human_and_scene(scene_pcd, c2ws, new_smpls, scale, mean_translation)

    else:
        # scene_pcd.paint_uniform_color([0.0, 0.0, 1.0])

        print("Visualizing the sparse scene point cloud and the detected ground floor")
        success = False
        while not success:
            plane_model, inliers = scene_pcd.segment_plane(0.0005, 3, 10000)
            inlier_pcd = scene_pcd.select_by_index(inliers)
            inlier_pcd_hull, _ = inlier_pcd.compute_convex_hull()
            inlier_pcd_hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(inlier_pcd_hull)
            inlier_pcd_hull_ls.paint_uniform_color([1.0, 0.0, 0.0])
            o3d.visualization.draw([scene_pcd, inlier_pcd_hull_ls])
            success = input("Is the ground floor (visualized in red lines) detected (y/n)? ") == "y"

        suffix = list(c2ws.keys())[0].split(".")[1]
        scales = []
        translations = []
        for i in range(num_frames):
            c2w = c2ws[f"{i:04d}.{suffix}"]
            transformed_smpl_verts, scale = solve_transformation(
                raw_smpls["verts"][i], plane_model, c2w
            )
            print(f"Estimated scale for frame {i:04d}: {scale:.2f}")
            scale_i, translation_i, _, _ = convert_smpl_to_world(
                raw_smpls, i, scale, c2w, transformed_smpl_verts, new_smpls, gender
            )
            scales.append(scale_i)
            translations.append(translation_i * scale_i)
        # mean_scale = np.mean(scales)
        mean_scale = np.quantile(scales, 0.1)
        mean_translation = np.mean(translations, axis=0)
        background = [scene_pcd, inlier_pcd]
        inlier_pcd.translate(-mean_translation, relative=True)
        inlier_pcd.scale(1.0 / mean_scale, center=np.zeros((3, 1)))
        center_and_scale_human_and_scene(scene_pcd, c2ws, new_smpls, mean_scale, mean_translation)
        print("Visualizing the aligned scene and human after normalization")
        visualize_smpl_and_scene(background, new_smpls, gender, sphere_radius=-1.0)

    success = False
    while not success:
        sphere_radius = float(input("Enter a radius for the bounding sphere of the whole scene: "))
        visualize_smpl_and_scene(background, new_smpls, gender, sphere_radius=sphere_radius)
        success = input("Is the green sphere good enough (y/n)? ") == "y"

    output_dir = data_dir / "normalized"
    output_dir.mkdir(exist_ok=True, parents=False)
    intrinsic = read_intrinsics_binary(sparse_dir)
    np.save(output_dir / "intrinsic.npy", intrinsic)
    o3d.io.write_point_cloud(str(output_dir / "scene_pcd.ply"), scene_pcd)
    with open(output_dir / "c2ws.pkl", "wb") as f:
        pkl.dump(c2ws, f)
    with open(data_dir / "normalized" / "aligned_smpl.pkl", "wb") as f:
        pkl.dump(new_smpls, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--sparse_dir", type=Path, required=True)
    parser.add_argument("--gender", type=str, choices=["male", "female"], required=True)
    parser.add_argument("--manual_scale", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    sparse_dir = args.sparse_dir.resolve()
    gender = args.gender
    manual_scale = args.manual_scale
    print("Aligning the estimated SMPL mesh to the reconstructed sparse scene.")
    align_human_scene(data_dir, sparse_dir, gender, manual_scale)
