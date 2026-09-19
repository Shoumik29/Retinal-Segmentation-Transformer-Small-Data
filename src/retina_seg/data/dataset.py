"""
Responsibilities:
    - List image/mask file paths for train/test splits
    - Load the segmentation colormap (used for visualizing predictions)
    - Create train and test patches from the config
    - Wrap numpy arrays of patches into a tf.data.Dataset
"""


import os
from glob import glob
from typing import Any, Dict, List, Tuple
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from retina_seg.data.patches import create_patches


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


def prepare_dataset(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Create train and test patches from the config.

    Args:
        config: loaded yaml config.

    Returns:
        train_img_patches, train_mask_patches, test_img_patches, test_mask_patches
    """

    train_img_paths, train_mask_paths, test_img_paths, test_mask_paths = prepare_data_paths(config["data"]["root"])

    patch_size = config["preprocessing"]["patch_size"]
    res = config["preprocessing"]["resize"]
    step = tuple(config["preprocessing"]["step"])
    scale = config["preprocessing"]["scale"]
    method = config["preprocessing"]["method"]
    num_augmented = config["augmentation"]["num_augmented"]

    print('Creating training patches...')
    train_img_patches, train_mask_patches = create_patches(
        train_img_paths, train_mask_paths, patch_size, res, step, scale, method, num_augmented
    )

    print('Creating test patches...')
    test_img_patches, test_mask_patches = create_patches(
        test_img_paths, test_mask_paths, patch_size, res, step, scale, method
    )

    print("Train patches: ", len(train_img_patches))
    print("Test patches: ", len(test_img_patches))

    return train_img_patches, train_mask_patches, test_img_patches, test_mask_patches


def tf_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 1,
    shuffle: bool = False,
    drop_remainder: bool = True,
	) -> tf.data.Dataset:

    """
    Wrap image/mask patch arrays into a batched tf.data.Dataset.
 
    Args:
        X: image patches, shape (N, H, W, C).
        Y: mask patches, shape (N, H, W, num_classes) — (mask one-hot encoded).
        batch_size: batch size (cfg.data.batch_size).
        shuffle: (True for training).
        drop_remainder: If the dataset size isn't divisible by batch_size.
 
    Returns:
        A batched tf.data.Dataset of (image, mask) pairs.
    """

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
