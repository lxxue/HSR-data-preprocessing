# create a env for sam2 and hloc
conda create -n hsr-data python=3.10
conda activate hsr-data

cd third_party/Grouned-SAM-2
cd checkpoints
bash download_ckpts.sh
cd ../
cd gdino_checkpoints
bash download_ckpts.sh
cd ../
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
export CUDA_HOME="/usr/local/cuda-12.1"
pip install -e .
pip install --no-build-isolation -e grounding_dino
pip install opencv-python supervision transformers addict yapf pycocotools timm

cd third_party/Hierarchical-Localization
git submodule update --init --recursive
pip install -e .
pip install pyquaternion
pip install scipy

pip install cython 
pip install simple-romp
# download smpl model for romp and put them in checkpoints/romp/SMPL_{GENDER}.pth
# TODO: add instruction for romp smpl checkpoint here
# TODO: add instruction for adding smpl pkl files

# for pytorch3d
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/download.html

# for romp smpl mask
# pip install git+https://github.com/mattloper/chumpy

# for smplx
pip install smplx
pip install open3d

# create a new env for metric3dv2 and install the dependencies

# install openpose python packages and checkpoints