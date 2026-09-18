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


def list_image_mask_paths(data_root: str, split: str) -> Tuple[List[str], List[str]]:

    """
    List of sorted image and mask paths training and testing split.

    Args:
        data_root: root directory of the dataset.
        split: "training" or "test".

    Returns:
        (image_paths, mask_paths) —> sorted, index-aligned.
    """

    img_paths = sorted(glob(os.path.join(data_root, split, "images", "*")))
    mask_paths = sorted(glob(os.path.join(data_root, split, "masks", "*")))

    if len(img_paths) != len(mask_paths):
        raise ValueError(
            f"Mismatched image/mask counts for split='{split}': "
            f"{len(img_paths)} images vs {len(mask_paths)} masks in {data_root}"
        )
    if len(img_paths) == 0:
        raise FileNotFoundError(
            f"No images found for split='{split}' under {data_root}. "
            f"Check the 'data.root' value in your config."
        )

    return img_paths, mask_paths


def prepare_data_paths(data_root: str) -> Tuple[List[str], List[str], List[str], List[str]]:

    """
    Wrapper returning train + test paths in one call.
 
    Returns:
        train_img_paths, train_mask_paths, test_img_paths, test_mask_paths
    """

    train_img_paths, train_mask_paths = list_image_mask_paths(data_root, "training")
    test_img_paths, test_mask_paths = list_image_mask_paths(data_root, "test")
    return train_img_paths, train_mask_paths, test_img_paths, test_mask_paths


def load_colormap(colormap_path: str) -> np.ndarray:

    """
    Load the .mat colormap to decode class indices into RGB for
    visualization.
 
    Args:
        colormap_path: path to .mat file containing a "color_map" key.
 
    Returns:
        np.ndarray of shape (num_classes, 3).
    """

    return loadmat(colormap_path)["color_map"]
