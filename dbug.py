# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:38:24 2017

@author: Phoebe
"""

import os
import time
import numpy as np
import pandas as pd

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".

#from pycocotools import mask as maskUtils
#%%
debugfile('ild.py', args='train --dataset=E:\lung_data --model=imagenet', wdir=r'C:\Users\Phoebe Chen\Desktop\CNNNNN\Mask_RCNN-master')
#%%
from config import Config
import utils
import model as modellib
ROOT_DIR = 'C:\\Users\\Phoebe Chen\\Desktop\\CNNNNN\\Mask_RCNN-master'

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
class InferenceConfig(ILDConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()

model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
model_path='C:\\Users\\Phoebe Chen\\Desktop\\CNNNNN\\Mask_RCNN-master\\mask_rcnn_coco.h5'
model.load_weights(model_path, by_name=True)
#%%
dataset='E:\lung_data'
dataset_train = ILDDataset()
dataset_train.load_ILD(dataset, "train")
#dataset_train.prepare()

# Validation dataset
dataset_val = ILDDataset()
dataset_train.load_ILD(dataset, "val")
#dataset_val.prepare()
#%%

print("Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=40,
            layers='heads')