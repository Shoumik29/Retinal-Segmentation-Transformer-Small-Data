"""
Preprocessing utilities for retinal fundus image segmentation.

Responsibilities:
    - Enhance fundus images (CLAHE, gamma, LAB, channel fusion, PCA, N4 bias correction)
    - Detect the circular field of view and neutralise the black border ring
"""

import os
from glob import glob
from typing import Callable, Dict, List, Tuple
import cv2
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from scipy.io import loadmat
from albumentations import (
    Compose,
    HorizontalFlip,
    HueSaturationValue,
    RandomBrightnessContrast,
    Rotate,
    VerticalFlip,
)
from sklearn.decomposition import PCA

