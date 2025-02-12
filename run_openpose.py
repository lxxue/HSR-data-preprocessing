"""
OpenPose Keypoint Detection

This script processes images to detect human keypoints using OpenPose. It uses human masks 
from SAM2 to identify the main actor in scenes with multiple people.

Arguments:
    --data_dir: output directory for the processed data
    --model_path: path to OpenPose model directory
    --net_resolution: network input resolution (default: "720x480")

Output Structure:
    data_dir/
    └── keypoints/
        └── openpose/
            ├── *.npy          # 2d keypoint coordinates
            └── *.png          # visualization
"""

import argparse
import glob
import time
from pathlib import Path

import cv2
import numpy as np
import pyopenpose as op
from sklearn.neighbors import NearestNeighbors


def crop_image(img, bbox, batch=False):
    if batch:
        return img[
            :,
            int(bbox[1]) : (int(bbox[1]) + int(bbox[3] - bbox[1])),
            int(bbox[0]) : (int(bbox[0]) + int(bbox[2] - bbox[0])),
        ]
    else:
        return img[
            int(bbox[1]) : (int(bbox[1]) + int(bbox[3] - bbox[1])),
            int(bbox[0]) : (int(bbox[0]) + int(bbox[2] - bbox[0])),
        ]


def recover_cropped_img(cropped_img, bbox, W, H):
    img = np.zeros((H, W, 3))
    img[
        int(bbox[1]) : (int(bbox[1]) + int(bbox[3] - bbox[1])),
        int(bbox[0]) : (int(bbox[0]) + int(bbox[2] - bbox[0])),
    ] = cropped_img
    return img


def recover_cropped_joints(joints_cropped, bbox):
    joints = np.zeros(joints_cropped.shape)
    joints[:, 0] = joints_cropped[:, 0] + int(bbox[0])  # int(bbox[1])
    joints[:, 1] = joints_cropped[:, 1] + int(bbox[1])  # int(bbox[0])
    joints[:, 2] = joints_cropped[:, 2]
    return joints


def read_img(img_path, mask_path):
    _img = cv2.imread(img_path)
    W, H = _img.shape[1], _img.shape[0]

    mask = cv2.imread(mask_path)[:, :, 0]
    where = np.asarray(np.where(mask))
    bbox_min = where.min(axis=1)
    bbox_min = bbox_min - 25
    bbox_max = where.max(axis=1)
    bbox_max = bbox_max + 25
    left, top, right, bottom = bbox_min[1], bbox_min[0], bbox_max[1], bbox_max[0]
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, W)
    bottom = min(bottom, H)
    crop_bbox = (left, top, right, bottom)
    bbox_center = np.array([left + (right - left) / 2, top + (bottom - top) / 2])
    _img_crop = crop_image(_img, crop_bbox)

    return _img_crop.copy(), crop_bbox, W, H, bbox_center


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--net_resolution", type=str, default="720x480")
    args = parser.parse_args()
    data_dir = args.data_dir
    model_path = args.model_path
    net_resolution = args.net_resolution

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = args.model_path
    params["scale_number"] = 1
    params["scale_gap"] = 0.25
    # need to be multiple of 16
    # params["net_resolution"] = "960x544"  # 1312x736 720x480
    # The first number is height and the second is width, so we need to adjust it based on the input resolution
    params["net_resolution"] = net_resolution

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    imagePaths = op.get_images_on_directory(str(args.data_dir / "images" / "selected_frames"))

    maskPaths = sorted(glob.glob(str(args.data_dir / "masks" / "sam2_human" / "*.jpg")))
    start = time.time()

    (args.data_dir / "keypoints" / "openpose").mkdir(parents=True, exist_ok=True)

    # Process and display images
    nbrs = NearestNeighbors(n_neighbors=1)
    for idx, imagePath in enumerate(imagePaths):
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        maskPath = maskPaths[idx]
        _, _, _, _, bbox_center = read_img(imagePath, maskPath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        poseKeypoints = datum.poseKeypoints
        # poseKeypoints = recover_cropped_joints(poseKeypoints, crop_bbox)

        nbrs.fit(poseKeypoints[:, 8, :2])

        actor = nbrs.kneighbors(bbox_center.reshape(1, -1), return_distance=False).ravel()[0]
        poseKeypoints = poseKeypoints[actor]
        openpose_dir = args.data_dir / "keypoints" / "openpose"
        np.save(openpose_dir / f"{idx:04d}.npy", poseKeypoints)
        cv2.imwrite(str(openpose_dir / f"{idx:04d}.png"), datum.cvOutputData)
        # output_img = imageToProcess
        # for jth in range(0, poseKeypoints.shape[0]):
        #     output_img = cv2.circle(
        #         imageToProcess,
        #         tuple(poseKeypoints.astype(np.int32)[jth, :2]),
        #         3,
        #         (0, 0, 255),
        #         -1,
        #     )
        # cv2.imwrite(str(args.data_dir / "openpose" / f"{idx:04d}.png"), output_img)
    end = time.time()
    print(f"OpenPose demo successfully finished. Total time: {end-start} seconds")
