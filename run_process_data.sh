INPUT_PATH="data/neuman/bike/images"
DATA_DIR="data/neuman_hsr_processed/bike"
WINDOW_SIZE=1
FRAME_START=0
FRAME_END=103 # inclusive
IMAGE_RESIZE_FACTOR=1
SFM_TOOL="colmap"
FEATURE_TYPE="sift"
MATCHER_TYPE="NN"
# SFM_TOOL="hloc"
# FEATURE_TYPE="superpoint_aachen"
# MATCHER_TYPE="superpoint+lightglue"
# SFM_TOOL="hloc"
# FEATURE_TYPE="loftr"
# MATCHER_TYPE="loftr"
MATCHING_METHOD="sequential"
GENDER="male"

python process_data.py \
    --input_path ${INPUT_PATH} \
    --data_dir ${DATA_DIR} \
    --window_size ${WINDOW_SIZE} \
    --frame_start ${FRAME_START} \
    --frame_end ${FRAME_END} \
    --image_resize_factor ${IMAGE_RESIZE_FACTOR} \
    --sfm_tool ${SFM_TOOL} \
    --feature_type ${FEATURE_TYPE} \
    --matcher_type ${MATCHER_TYPE} \
    --matching_method ${MATCHING_METHOD} \
    --gender ${GENDER} \
    --steps ${@:1} \
    # --manual_scale # uncomment this line to use manual scale
