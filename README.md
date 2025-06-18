# HSR-data-preprocessing

This repository provides a preprocessing pipeline for monocular videos containing human motion in static scenes. 

Given an input video, our pipeline estimates camera poses, reconstructs human poses in world coordinates, and extracts monocular geometric cues (depth and surface normals). 
The processed data can then be used by [HSR](https://github.com/lxxue/HSR) to create human-scene reconstructions.

This preprocessing pipeline is maintained as a standalone repository to facilitate its use in other applications beyond HSR.



## General pipeline

The pipeline consists of the following sequential steps:

0. Extract and select sharp frames from a video or an image sequence

1. Generate human masks

2. Estimate camera poses 

3. Generate monocular depth and normal maps

4. Estimate human poses in the camera coordinate frame

5. Extract human 2D keypoints

6. Refine human poses with 2D keypoints and temporal smoothness

7. Align human poses in the world coordinate frame and scale scene to metric units using human body scale

8. Save processed data in HSR-compatible format

## Setup


Clone the repository and its submodules:

```bash
git clone https://github.com/lxxue/HSR-data-preprocessing.git --recursive
```

Setup the environment for the main repository and `Grounded-SAM2` / `hloc` / `ROMP`:
```bash
conda create -n hsr-data python=3.10
conda activate hsr-data
# SAM2.1 requires torch >=2.5.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# For Grounded-SAM2
cd third_party/Grouned-SAM-2
cd checkpoints
bash download_ckpts.sh
cd ../
cd gdino_checkpoints
bash download_ckpts.sh
cd ../
export CUDA_HOME="/usr/local/cuda-12.1"
pip install -e .
pip install --no-build-isolation -e grounding_dino
pip install opencv-python supervision transformers addict yapf pycocotools timm

# For hloc
cd ../../
cd third_party/Hierarchical-Localization
git submodule update --init --recursive
pip install -e .
pip install pyquaternion scipy

pip install cython 
pip install simple-romp
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/download.html
pip install smplx open3d
```

Create a separate environment for Metric3Dv2 following the [official instructions](https://github.com/YvanYin/Metric3D?tab=readme-ov-file#-installation).

Build openpose python package following the [official guide](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md).

Update python paths in [process_data.py](./process_data.py) and [cmd.sh](./cmd.sh).

```python
# process_data.py
SAM2_PYTHON_PATH = "/home/lixin/miniconda3/envs/hsr-data/bin/python"
METRIC3D_PYTHON_PATH = "/home/lixin/miniconda3/envs/metric3d/bin/python"
OPENPOSE_PYTHON_PATH = "/usr/bin/python3"
OPENPOSE_MODEL_PATH = "/home/lixin/softwares/openpose/models/"

# cmd.sh
SAM2_PYTHON_PATH="/home/lixin/miniconda3/envs/hsr-data/bin/python"
METRIC3D_PYTHON_PATH="/home/lixin/miniconda3/envs/metric3d/bin/python"
OPENPOSE_PYTHON_PATH="/usr/bin/python3"
OPENPOSE_MODEL_PATH="/home/lixin/softwares/openpose/models/"

```

Download [SMPL model](https://smpl.is.tue.mpg.de/download.php) (version 1.1.0 for Python 2.7 (female/male)) and place them under `checkpoints/smpl`:

```bash
mkdir -p checkpoints/smpl
mv /path_to_smpl_models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl checkpoints/smpl/SMPL_FEMALE.pkl
mv /path_to_smpl_models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl checkpoints/smpl/SMPL_MALE.pkl
```

Prepare SMPL model files needed by ROMP according to the [official instructions](https://github.com/Arthur151/ROMP/blob/a8558aed480af850756f84e2a7c787e359bddbd0/trace/README.md#metadata) and place them under `checkpoints/romp`:

```bash
mkdir -p checkpoints/romp
mv /path_to_romp_models/SMPL_MALE.pth checkpoints/romp/SMPL_MALE.pth
mv /path_to_romp_models/SMPL_FEMALE.pth checkpoints/romp/SMPL_FEMALE.pth
```

## Usage

We provide a python script [process_data.py](./process_data.py) and a shell script [run_process_data.sh](./run_process_data.sh) as examples to process the data.

```bash
# Modify the arguments in run_process_data.sh first to fit your data
# Run each step with indices, e.g. 0 1 2 (modify indices as needed)
bash run_process_data.sh 0 1 2
```

You can also run each step separately by uncommenting the corresponding command in [cmd.sh](./cmd.sh).

```bash
bash cmd.sh
```

Each script contains detailed documentation of its functionality. For example, in [select_frames.py](./select_frames.py):

```python
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
```


## Acknowledgements


This work builds upon several excellent open-source projects. We would like to thank the authors of:
[Vid2Avatar](https://github.com/MoyGcc/vid2avatar), 
[NeuMAN](https://github.com/apple/ml-neuman), 
[hloc](https://github.com/cvg/Hierarchical-Localization),
[colmap](https://github.com/colmap/colmap)
[Metric3D](https://github.com/YvanYin/Metric3D), 
[Grounded-SAM2](https://github.com/IDEA-Research/Grounded-SAM-2)
[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), 
[ROMP](https://github.com/Arthur151/ROMP) 
.

## BibTex

If you find this work useful for your research, please consider citing our paper:

```
@inproceedings{xue2024hsr,
    author={Xue, Lixin and Guo, Chen and Zheng, Chengwei and Wang, Fangjinhua and Jiang, Tianjian and Ho, Hsuan-I and Kaufmann, Manuel and Song, Jie and Hilliges Otmar},
    title={{HSR:} Holistic 3D Human-Scene Reconstruction from Monocular Videos},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2024}
}
```