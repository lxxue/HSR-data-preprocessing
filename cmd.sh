# python paths
SAM2_PYTHON_PATH="/home/lixin/miniconda3/envs/sam21/bin/python"
METRIC3D_PYTHON_PATH="/home/lixin/miniconda3/envs/metric3d/bin/python"
OPENPOSE_PYTHON_PATH="/usr/bin/python3"
OPENPOSE_MODEL_PATH="/home/lixin/softwares/openpose/models/"

# dataset paths
INPUT_PATH="data/neuman/bike/images"
DATA_DIR="data/neuman_hsr_processed/bike"
WINDOW_SIZE=1
FRAME_START=0
FRAME_END=103 # inclusive
IMAGE_RESIZE_FACTOR=1

CAMERA_MODEL="SIMPLE_PINHOLE"
SFM_TOOL="colmap"
FEATURE_TYPE="sift"
MATCHING_METHOD="sequential"
MATCHER_TYPE="NN"
# SFM_TOOL="hloc"
# FEATURE_TYPE="superpoint_aachen"
# MATCHING_METHOD="sequential"
# MATCHER_TYPE="superpoint+lightglue"
# SFM_TOOL="hloc"
# FEATURE_TYPE="loftr"
# MATCHING_METHOD="sequential"
# MATCHER_TYPE="loftr"

GENDER="male"

SPARSE_DIR="${DATA_DIR}/camera_poses/${SFM_TOOL}-${FEATURE_TYPE}-${MATCHING_METHOD}-${MATCHER_TYPE}/sparse/0"

# ${SAM2_PYTHON_PATH} select_frames.py \
#     --input_path ${INPUT_PATH} \
#     --data_dir ${DATA_DIR} \
#     --window_size ${WINDOW_SIZE} \
#     --frame_start ${FRAME_START} \
#     --frame_end ${FRAME_END} \
#     --image_resize_factor ${IMAGE_RESIZE_FACTOR}

# ${SAM2_PYTHON_PATH} create_masks_with_sam2.py \
#     --data_dir ${DATA_DIR} \
#     --text "human."

# ${SAM2_PYTHON_PATH} estimate_camera_poses.py \
#     --data_dir ${DATA_DIR} \
#     --camera_model ${CAMERA_MODEL} \
#     --sfm_tool ${SFM_TOOL} \
#     --feature_type ${FEATURE_TYPE} \
#     --matcher_type ${MATCHER_TYPE} \
#     --matching_method ${MATCHING_METHOD} \
#     --image_resize_factor ${IMAGE_RESIZE_FACTOR}

# ${METRIC3D_PYTHON_PATH} extract_monocular_cues_with_Metric3D.py \
#     --data_dir ${DATA_DIR}

# ${SAM2_PYTHON_PATH} run_romp.py \
#     --data_dir ${DATA_DIR} \
#     --gender ${GENDER}

# ${OPENPOSE_PYTHON_PATH} run_openpose.py \
#     --data_dir ${DATA_DIR} \
#     --model_path ${OPENPOSE_MODEL_PATH}

# ${SAM2_PYTHON_PATH} refine_romp.py \
#     --data_dir ${DATA_DIR} \
#     --sparse_dir ${SPARSE_DIR} \
#     --gender ${GENDER}

# # This step needs GUI
# # If automatic scale estimation failed (e.g. when the wrong ground plane is detected)
# # you can uncomment --manual_scale and set the scale manually
# ${SAM2_PYTHON_PATH} align_human_scene.py \
#     --data_dir ${DATA_DIR} \
#     --sparse_dir ${SPARSE_DIR} \
#     --gender ${GENDER} \
#     # --manual_scale

# ${SAM2_PYTHON_PATH} prepare_dataset.py \
#     --data_dir ${DATA_DIR}