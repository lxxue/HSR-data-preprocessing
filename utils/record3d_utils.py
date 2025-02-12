import json
import subprocess

import numpy as np
from hloc import extract_features, match_dense, match_features, pairs_from_poses, triangulation
from hloc.utils.read_write_model import Camera, Image, write_model

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation


# modified from nerfstudio record3d_to_json
def load_record3d_cameras(data_dir, video_resize_factor: int):
    # metadata_path = input_dir / "EXR_RGBD" / "metadata.json"
    metadata_path = data_dir / "images" / "metadata.json"
    indices = np.load(data_dir / "images" / "selected_idxs.npy")
    with open(metadata_path, encoding="UTF-8") as file:
        metadata_dict = json.load(file)

    poses_data = np.array(metadata_dict["poses"])  # (N, 3, 4)
    # NB: Record3D / scipy use "scalar-last" format quaternions (x y z w)
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    camera_to_worlds = np.concatenate(
        [Rotation.from_quat(poses_data[:, :4]).as_matrix(), poses_data[:, 4:, None]],
        axis=-1,
    ).astype(np.float32)
    camera_to_worlds = camera_to_worlds[indices]

    homogeneous_coord = np.zeros_like(camera_to_worlds[..., :1, :])
    homogeneous_coord[..., :, 3] = 1
    camera_to_worlds = np.concatenate([camera_to_worlds, homogeneous_coord], -2)

    # Camera intrinsics
    K = np.array(metadata_dict["K"]).reshape((3, 3)).T
    focal_length = K[0, 0]

    h = metadata_dict["h"]
    w = metadata_dict["w"]

    # TODO(akristoffersen): The metadata dict comes with principle points,
    # but caused errors in image coord indexing. Should update once that is fixed.
    cx, cy = w / 2, h / 2

    if video_resize_factor > 1:
        h = int(h / video_resize_factor)
        w = int(w / video_resize_factor)
        focal_length = focal_length / video_resize_factor
        cx = cx / video_resize_factor
        cy = cy / video_resize_factor

    intrinsics = {"focal_length": focal_length, "h": h, "w": w, "cx": cx, "cy": cy}
    return camera_to_worlds, intrinsics


# from https://github.com/NVlabs/instant-ngp/issues/1080#issuecomment-1499700290
def convert_pose_to_opencv_format(c2w):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    c2w = np.matmul(c2w, flip_yz)
    return c2w


def create_database_from_record3d(
    data_dir, c2ws, intrinsics, feature_type, matching_method, matcher_type
):
    images = {}
    for i in range(c2ws.shape[0]):
        c2w = c2ws[i]
        c2w_cv = convert_pose_to_opencv_format(c2w)
        w2c_cv = np.linalg.inv(c2w_cv)
        R = w2c_cv[:3, :3]
        q = Quaternion(matrix=R, atol=1e-06)
        qvec = np.array([q.w, q.x, q.y, q.z])
        tvec = w2c_cv[:3, -1]

        image_id = i + 1
        img_name = f"{i:04d}.png"
        image = Image(
            id=image_id, qvec=qvec, tvec=tvec, camera_id=1, name=img_name, xys=[], point3D_ids=[]
        )
        images[image_id] = image

    cameras = {}
    camera_id = 1
    fx = fy = intrinsics["focal_length"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    camera = Camera(
        id=camera_id,
        model="PINHOLE",
        width=intrinsics["w"],
        height=intrinsics["h"],
        params=[fx, fy, cx, cy],
    )
    cameras[camera_id] = camera

    points3D = {}

    print("Writing the COLMAP model")
    record3d_dir = (
        data_dir
        / "localization"
        / f"record3d-{feature_type}-{matching_method}-{matcher_type}"
        / "sparse"
        / "0"
    )
    record3d_dir.mkdir(exist_ok=True, parents=True)
    write_model(
        images=images, cameras=cameras, points3D=points3D, path=str(record3d_dir), ext=".bin"
    )


def triangulate(
    data_dir,
    sfm_tool,
    feature_type,
    matching_method,
    matcher_type,
    num_matched,
    num_ba_iterations,
):
    assert matching_method == "poses"
    record3d_dir = (
        data_dir / "localization" / f"{sfm_tool}-{feature_type}-{matching_method}-{matcher_type}"
    )
    images = data_dir / "images" / "selected_frames"
    sfm_pairs = record3d_dir / "pairs.txt"
    features = record3d_dir / "features.h5"
    matches = record3d_dir / "matches.h5"
    mask_dir = data_dir / "masks" / "colmap"

    references = [str(p.relative_to(images)) for p in images.iterdir()]
    references = sorted(references)
    if "loftr" not in matcher_type:
        feature_conf = extract_features.confs[feature_type]
        matcher_conf = match_features.confs[matcher_type]
        extract_features.main(
            feature_conf, images, image_list=references, feature_path=features, mask_dir=mask_dir
        )
    else:
        print("LoFTR does not need feature extraction")
        feature_conf = {}
        matcher_conf = match_dense.confs[matcher_type]

    colmap_input = record3d_dir / "sparse" / "0"

    pairs_from_poses.main(colmap_input, sfm_pairs, num_matched)
    if "loftr" not in matcher_type:
        match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
    else:
        match_dense.main(
            matcher_conf,
            sfm_pairs,
            image_dir=images,
            export_dir=record3d_dir,
            max_kps=8192,
            overwrite=False,
            features=features,
            matches=matches,
        )

    if num_ba_iterations == 0:
        colmap_sparse = record3d_dir / "sparse" / "0"
        colmap_sparse.mkdir(exist_ok=True, parents=False)
        _ = triangulation.main(
            colmap_sparse,  # output model
            colmap_input,  # input model
            images,
            sfm_pairs,
            features,
            matches,
        )
        return colmap_sparse

    for i in range(num_ba_iterations):
        colmap_sparse = record3d_dir / "sparse" / f"{i}"
        colmap_sparse.mkdir(exist_ok=True, parents=False)
        _ = triangulation.main(
            colmap_sparse,  # output model
            colmap_input,  # input model
            images,
            sfm_pairs,
            features,
            matches,
        )

        colmap_ba = record3d_dir / "sparse" / f"{i}_ba"
        colmap_ba.mkdir(exist_ok=True, parents=True)
        cmd = [
            "colmap",
            "bundle_adjuster",
            "--input_path",
            str(colmap_sparse),
            "--output_path",
            str(colmap_ba),
        ]
        subprocess.run(cmd)

        colmap_input = colmap_ba

    output_path = colmap_input

    return output_path
