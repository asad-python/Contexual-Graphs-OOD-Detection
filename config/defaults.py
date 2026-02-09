"""
Default Configuration File for Contextual Graph OOD
-------------------------------------------------------------

This file contains all global parameters and default settings used
throughout the pipeline. Modify paths and parameters here instead
of editing individual modules.

-----------------------
PATCH DEFAULT BEHAVIOR
-----------------------
- Objects are pasted ONLY in the LOWER HALF of the image by default
- Conservative scaling and rotation are used for realism
- Augmentations are intentionally mild to avoid unrealistic artifacts
"""

import random
import numpy as np

# ==========================
# DATASET PATHS (USER SETUP)
# ==========================

# Path to original NuScenes dataset
SRC_DATASET = "WRITE_SOURCE_NUSCENES_DATASET_PATH_HERE"

# Path where modified / OOD injected dataset will be stored
DST_DATASET = "WRITE_DESTINATION_DATASET_PATH_HERE"

# Path containing COCO / custom object assets for OOD pasting
ASSETS_DIR = "WRITE_ASSETS_DIRECTORY_PATH_HERE"


# ==========================
# OOD INJECTION PARAMETERS
# ==========================

# Number of OOD objects pasted per scene
N_PASTES_PER_SCENE = 50

# Fraction of objects that should be novel (vs in-distribution)
NOVEL_RATE = 0.5

# Minimum and maximum number of cameras to modify per frame
MIN_CAMS_PER_FRAME = 3
MAX_CAMS_PER_FRAME = 6


# ==========================
# PATCH AUGMENTATION DEFAULTS
# ==========================

# Scale range applied to pasted object patches
PATCH_SCALE_RANGE = (0.3, 0.6)

# Maximum rotation (degrees) applied to patches
PATCH_ROTATE_DEG = 8

# IMPORTANT DEFAULT:
# If True, objects will only be pasted in the lower half of images
PATCH_LOWER_HALF_ONLY = True


# ==========================
# REPRODUCIBILITY
# ==========================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ==========================
# VISUALIZATION SETTINGS
# ==========================

# Whether to save debug visualizations
SAVE_DEBUG_IMAGES = True

# Output directory for debug results
DEBUG_OUTPUT_DIR = "WRITE_DEBUG_OUTPUT_PATH_HERE"


# ==========================
# DETECTION / MODEL SETTINGS
# ==========================

# Confidence threshold for YOLO / detection models
DETECTION_CONF_THRESHOLD = 0.25

# Device for running detection models
DEVICE = "cuda"   # change to "cpu" if GPU not available
