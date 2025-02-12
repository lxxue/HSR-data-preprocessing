"""
Structure-from-Motion Pipeline

This script performs camera pose estimation using various SfM tools (COLMAP, HLOC, Record3D).
Each tool offers different feature extraction and matching methods, suitable for different 
scenarios.

Arguments:
    --data_dir: output directory for the processed data 
    --camera_model: camera model type (SIMPLE_PINHOLE, PINHOLE)
    --sfm_tool: sfm method (colmap, hloc, record3d)
    --feature_type: feature extractor type (sift, superpoint_aachen, etc.)
    --matcher_type: feature matcher type (NN, superglue, etc.)
    --matching_method: matching strategy (exhaustive, poses, sequential)
    --image_resize_factor: image downscaling factor used in the selct_frames.py script 

Output Structure:
    data_dir/
    └── camera_poses/
        └── {sfm-tool}-{features}-{matching}-{matcher}/  # camera poses and sparse reconstruction
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pycolmap
from hloc import (
    extract_features,
    match_dense,
    match_features,
    pairs_from_exhaustive,
    pairs_from_retrieval,
    pairs_from_sequence,
    reconstruction,
)

from utils.record3d_utils import create_database_from_record3d, load_record3d_cameras, triangulate


# modified from nerfstudio
def run_command(cmd: list, verbose=False):
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    cmd_str = " ".join(cmd)
    out = subprocess.run(cmd_str, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
        print(out.stderr.decode("utf-8"))
        sys.exit(1)
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out


# modified based on nerfstudio and my own implementation
def run_colmap(data_dir, camera_model, matching_method):
    # (colmap_dir / "database.db").unlink(missing_ok=True)
    colmap_dir = data_dir / "camera_poses" / f"colmap-sift-{matching_method}-NN"
    image_dir = data_dir / "images" / "selected_frames"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    # feature extraction
    feature_extractor_cmd = [
        "colmap",
        f"feature_extractor",
        "--database_path",
        f"{colmap_dir / 'database.db'}",
        "--image_path",
        str(image_dir),
        "--ImageReader.mask_path",
        f"{data_dir / 'masks' / 'colmap'}",
        f"--ImageReader.camera_model",
        f"{camera_model}",
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.estimate_affine_shape",
        "true",
        "--SiftExtraction.domain_size_pooling",
        "true",
        "--SiftExtraction.use_gpu",
        "1",
    ]
    run_command(feature_extractor_cmd)

    # feature matching
    feature_matcher_cmd = [
        f"colmap",
        f"{matching_method}_matcher",
        f"--database_path",
        f"{colmap_dir / 'database.db'}",
        "--SiftMatching.guided_matching",
        "true",
        "--SiftMatching.use_gpu",
        "1",
    ]
    if matching_method == "sequential":
        # https://github.com/colmap/colmap/issues/636
        feature_matcher_cmd += ["--SequentialMatching.overlap", "20"]
        feature_matcher_cmd += ["--SequentialMatching.quadratic_overlap", "0"]
    subprocess.run(feature_matcher_cmd)

    db_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"

    # sparse reconstruction
    sparse_dir.mkdir(parents=False, exist_ok=True)
    mapper_cmd = [
        "colmap mapper",
        f"--database_path {db_path}",
        f"--image_path {image_dir}",
        f"--output_path {sparse_dir}",
        "--Mapper.ba_global_function_tolerance 1e-6",  # from nerfstuido
    ]
    print("Sparse reconstruction started")
    run_command(mapper_cmd)
    print("Sparse reconstruction finished")

    # refine intrinsics
    bundle_adjuster_cmd = [
        "colmap bundle_adjuster",
        f"--input_path {sparse_dir / '0'}",
        f"--output_path {sparse_dir / '0'}",
        "--BundleAdjustment.refine_principal_point 1",
    ]
    print("Refining intrinsics started")
    run_command(bundle_adjuster_cmd)
    print("Refining intrinsics finished")

    return


# mostly from nerfstudio
def run_hloc(data_dir, camera_model, feature_type, matching_method, matcher_type, num_matched):
    hloc_dir = data_dir / "camera_poses" / f"hloc-{feature_type}-{matching_method}-{matcher_type}"
    image_dir = data_dir / "images" / "selected_frames"
    mask_dir = data_dir / "masks" / "colmap"
    pairs = hloc_dir / f"pairs.txt"
    features = hloc_dir / "features.h5"
    matches = hloc_dir / "matches.h5"

    hloc_dir.mkdir(parents=True, exist_ok=True)
    if "loftr" not in matcher_type:
        feature_conf = extract_features.confs[feature_type]
        matcher_conf = match_features.confs[matcher_type]
    else:
        print("LoFTR does not need feature extraction")
        feature_conf = {}
        matcher_conf = match_dense.confs[matcher_type]
    retrieval_conf = extract_features.confs["netvlad"]

    references = [str(p.relative_to(image_dir).as_posix()) for p in image_dir.iterdir()]
    references = sorted(references)
    if matcher_type != "loftr":
        extract_features.main(
            feature_conf, image_dir, image_list=references, feature_path=features, mask_dir=mask_dir
        )
    if matching_method == "exhaustive":
        pairs_from_exhaustive.main(pairs, image_list=references)
    elif matching_method == "retrieval":
        retrieval_path = extract_features.main(retrieval_conf, image_dir, hloc_dir)
        if num_matched >= len(references):
            num_matched = len(references)
        pairs_from_retrieval.main(retrieval_path, pairs, num_matched=num_matched)
    elif matching_method == "sequential":
        pairs_from_sequence.main(pairs, image_list=references, window_size=20)
    else:
        raise NotImplementedError
    if matcher_type != "loftr":
        match_features.main(matcher_conf, pairs, features=features, matches=matches)
    else:
        match_dense.main(
            matcher_conf,
            pairs,
            image_dir=image_dir,
            export_dir=hloc_dir,
            max_kps=8192,
            overwrite=False,
            features=features,
            matches=matches,
        )

    image_options = pycolmap.ImageReaderOptions(camera_model=camera_model)

    sfm_dir = hloc_dir / "sparse" / "0"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.main(
        sfm_dir,
        image_dir,
        pairs,
        features,
        matches,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_options=image_options,
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--camera_model",
        type=str,
        choices=["SIMPLE_PINHOLE", "PINHOLE"],
        default="SIMPLE_PINHOLE",
    )
    parser.add_argument("--sfm_tool", type=str, choices=["colmap", "hloc", "record3d"])
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["sift", "superpoint_aachen", "r2d2", "d2net-ss", "sosnet", "disk", "loftr"],
    )
    parser.add_argument(
        "--matcher_type",
        type=str,
        choices=["NN", "NN-ratio", "superglue", "superpoint+lightglue", "disk+lightglue", "loftr"],
    )
    parser.add_argument(
        "--matching_method", type=str, choices=["exhaustive", "poses", "sequential"]
    )
    parser.add_argument("--image_resize_factor", type=int, default=1)

    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    sfm_tool = args.sfm_tool
    camera_model = args.camera_model
    feature_type = args.feature_type
    matcher_type = args.matcher_type
    matching_method = args.matching_method
    image_resize_factor = args.image_resize_factor
    if sfm_tool == "colmap":
        assert feature_type == "sift" and matcher_type == "NN"
        run_colmap(data_dir, camera_model, matching_method)
    elif sfm_tool == "hloc":
        run_hloc(
            data_dir, camera_model, feature_type, matching_method, matcher_type, num_matched=40
        )
    elif sfm_tool == "record3d":
        c2ws, intrinsics = load_record3d_cameras(data_dir, image_resize_factor)
        create_database_from_record3d(
            data_dir, c2ws, intrinsics, feature_type, matching_method, matcher_type
        )
        sfm_dir = triangulate(
            data_dir,
            sfm_tool,
            feature_type,
            matching_method,
            matcher_type,
            num_matched=40,
            num_ba_iterations=3,
        )
        image_dir = data_dir / "images" / "selected_frames"
    else:
        raise NotImplementedError
