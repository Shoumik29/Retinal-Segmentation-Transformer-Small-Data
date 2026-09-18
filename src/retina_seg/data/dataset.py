"""
Responsibilities:
    - List image/mask file paths for train/test splits
    - Load the segmentation colormap (used for visualizing predictions)
    - Wrap numpy arrays of patches into a tf.data.Dataset
"""

import os
from glob import glob
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from scipy.io import loadmat
