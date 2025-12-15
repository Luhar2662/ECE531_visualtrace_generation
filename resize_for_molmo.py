# rrt_interface.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import open3d as o3d

# Optional: visualization of image/depth
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from PIL import Image, ImageDraw
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
from pathlib import Path
import hashlib
import cv2 as cv

# =============================
# Depth -> RGBD -> Point cloud
# =============================
def _resize_image(image: np.ndarray) -> np.ndarray:
    image = cv.resize(image, tuple([256,256]), interpolation=cv.INTER_AREA)
    return image


## Write code to open and image file resize it to 256x256 and save it back

def resize_image_file(input_path: str, output_path: str) -> None:
    image = cv.imread(input_path)
    resized_image = _resize_image(image)
    cv.imwrite(output_path, resized_image)  
    

# paths = ["pick_up_the_blue_can_obj_-0.1_-0.05_001_001.png", "pick_up_the_blue_can_obj_-0.1_0.45_001_001.png", "pick_up_the_blue_can_obj_-0.35_-0.05_001_001.png",
#          "pick_up_the_blue_can_obj_-0.35_0.45_001_001.png", "pick_up_the_blue_can_obj_-0.235_0.42_001_001.png", "pick_up_the_blue_can_obj_-0.22499999999999998_-0.05_001_001.png",
#          "pick_up_the_blue_can_obj_-0.22499999999999998_0.45_001_001.png"]

paths = ["clean_blue_can_obj_-0.1_-0.05001.png"]

for path in paths:
    resize_image_file(path, "resized_" + path)