import os
import sys
import cv2
import random
import math
import numpy as np
import skimage.io
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tensorflow as tf
# import keras
from tensorflow import keras


# IMPORTS

# Have to specify path in order to import"
ROOT_DIR = '/Users/OAA/PycharmProjects/Mask_RCNN/Mask_RCNN_TF2'

# Change path
sys.path.insert(1, ROOT_DIR)

# Import
import importlib.util as ilu


# import Mask-RCNN-TF2.mrcnn

import mrcnn
from mrcnn import (config,
                   model,
                   utils,
                   visualize)


# Import COCO Config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco

# Directory SETUP

# Directory to save logs and trained models
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Trained Weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Setting batch size to 1 - as we'll be running inference on 1 image at a time.
    GPU_COUNT = 1  # Batch size = GPU_COUNT * IMAGES_PER_GPU
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

model = mrcnn.model.MaskRCNN(mode="inference",
                             model_dir = MODEL_DIR,
                             config = config)

print(COCO_MODEL_PATH)

model.load_weights(COCO_MODEL_PATH, by_name=True)


file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))


from tensorflow.python.compiler.mlcompute import mlcompute
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
mlcompute.set_mlc_device(device_name='gpu') # Available options are 'cpu', 'gpu', and 'any'.
tf.config.run_functions_eagerly(False)
print(tf.executing_eagerly())

results = model.detect([image], verbose=1)