
import random
import numpy as np



# Path to original NuScenes dataset
SRC_DATASET = "WRITE_SOURCE_NUSCENES_DATASET_PATH_HERE"

# Path where modified / OOD injected dataset will be stored
DST_DATASET = "WRITE_DESTINATION_DATASET_PATH_HERE"

# Path containing COCO / custom object assets for OOD pasting
ASSETS_DIR = "WRITE_ASSETS_DIRECTORY_PATH_HERE"



# Number of OOD objects pasted per scene
N_PASTES_PER_SCENE = 50

# Fraction of objects that should be novel (vs in-distribution)
NOVEL_RATE = 0.5

# Minimum and maximum number of cameras to modify per frame
MIN_CAMS_PER_FRAME = 3
MAX_CAMS_PER_FRAME = 6



# Scale range applied to pasted object patches
PATCH_SCALE_RANGE = (0.3, 0.6)

# Maximum rotation (degrees) applied to patches
PATCH_ROTATE_DEG = 8

# IMPORTANT DEFAULT:
# If True, objects will only be pasted in the lower half of images
PATCH_LOWER_HALF_ONLY = True



SEED = 42

random.seed(SEED)
np.random.seed(SEED)



# Whether to save debug visualizations
SAVE_DEBUG_IMAGES = True

# Output directory for debug results
DEBUG_OUTPUT_DIR = "WRITE_DEBUG_OUTPUT_PATH_HERE"


# Confidence threshold for YOLO / detection models
DETECTION_CONF_THRESHOLD = 0.25

# Device for running detection models
DEVICE = "cuda"   # change to "cpu" if GPU not available
