"""
Frame Selection Utility for Videos and Image Sequences

Arguments:
    --input_path: path to the input video file or a directory of images
    --data_dir: output directory for the processed data 
    --window_size: number of frames to consider in each selection window (default: 10)
    --frame_start: starting frame number to process (default: 0)
    --frame_end: ending frame number (inclusive) to process (default: 1000000)
    --image_resize_factor: factor by which to reduce image size (1, 2, 4, or 8)

Output Structure:
    data_dir/
    ├── images/
    │   ├── all_frames/         # Contains all processed frames
    │   ├── selected_frames/    # Contains selected sharp frames
    │   └── selected_idxs.npy   # Numpy array of selected frame indices
"""

import argparse
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# modified from https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py
def compute_variance_of_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def get_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = compute_variance_of_laplacian(gray)
    return fm


def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group()) if match else -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # we accept either a video or a directory of images
    parser.add_argument("--input_path", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, required=True)
    # we select the sharpest frame in every window
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=1000000)
    parser.add_argument("--image_resize_factor", type=int, default=1, choices=[1, 2, 4, 8])

    args = parser.parse_args()
    input_path = args.input_path
    data_dir = args.data_dir
    window_size = args.window_size
    frame_start = args.frame_start
    frame_end = args.frame_end
    image_resize_factor = args.image_resize_factor

    data_dir.mkdir(parents=False, exist_ok=True)
    image_dir = data_dir / "images"

    output_all_dir = image_dir / "all_frames"
    output_all_dir.mkdir(parents=True, exist_ok=True)
    output_selected_dir = image_dir / "selected_frames"
    output_selected_dir.mkdir(parents=False, exist_ok=True)
    imgs = []
    sharps = []
    if input_path.is_dir():
        image_pattern = re.compile(r"(\d+).(jpg|jpeg|png)$", re.IGNORECASE)
        image_files = [file for file in input_path.glob("*") if image_pattern.match(file.name)]
        image_files = sorted(image_files, key=lambda x: extract_number(x.name))
        # For record3d data, we also copy the metadata files containing intrinsics & extrinsics
        if (input_path.parent / "metadata.json").exists():
            print("Copying Record3D metadata.json")
            subprocess.run(
                ["cp", str(input_path.parent / "metadata.json"), str(image_dir / "metadata.json")]
            )
        for file in tqdm(image_files):
            img = cv2.imread(str(file))
            if image_resize_factor > 1:
                new_h = img.shape[0] // image_resize_factor
                new_w = img.shape[1] // image_resize_factor
                img = cv2.resize(img, (new_w, new_h))
            imgs.append(img)
            sharps.append(get_sharpness(img))
    else:
        vidcap = cv2.VideoCapture(str(input_path))
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(length)):
            success, img = vidcap.read()
            if success:
                if image_resize_factor > 1:
                    new_h = img.shape[0] // image_resize_factor
                    new_w = img.shape[1] // image_resize_factor
                    img = cv2.resize(img, (new_w, new_h))

                # For record3d RGBD video, we only keep the RGB part
                # w = img.shape[1]
                # img = img[:, w // 2 :]
                imgs.append(img)
                sharps.append(get_sharpness(img))
            else:
                print(f"Failed to read frame {i}")

    # skip frames in the beginning and in the end if needed
    imgs = imgs[frame_start : frame_end + 1]
    sharps = sharps[frame_start : frame_end + 1]

    selected_imgs = []
    # select the sharpest frame in every window
    length = len(imgs)
    idxs = []
    for i in range(0, length, window_size):
        sharp_window = sharps[i : i + window_size]
        idx = np.argmax(sharp_window)
        idxs.append(idx + i)
        selected_imgs.append(imgs[i + idx])
    for i, img in enumerate(imgs):
        cv2.imwrite(str(output_all_dir / f"{i:04d}.jpg"), img)
    for i, img in enumerate(selected_imgs):
        cv2.imwrite(str(output_selected_dir / f"{i:04d}.jpg"), img)
    np.save(image_dir / "selected_idxs.npy", np.array(idxs))
    print(f"Extracted {len(selected_imgs)} frames from {length} frames")
