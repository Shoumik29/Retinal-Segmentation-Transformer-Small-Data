"""
Responsibilities:
    - List image/mask file paths for train/test splits
    - Load the segmentation colormap (used for visualizing predictions)
    - Read, resize, preprocess, augment and patch images into training arrays
    - Wrap numpy arrays of patches into a tf.data.Dataset
"""


import os
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import imageio
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from tensorflow.keras.utils import to_categorical

from retina_seg.data import preprocessing
from retina_seg.data.augmentation import add_augmented_patches
from retina_seg.data.patches import sliding_window


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


def load_colormap(colormap_path: str, colormap_key: str = "color_map") -> np.ndarray:

    """
    Load the .mat colormap to decode class indices into RGB for
    visualization.

    Args:
        colormap_path: path to the .mat file.
        colormap_key: variable name of the colormap inside the .mat file
            (cfg.data.colormap_key).

    Returns:
        np.ndarray of shape (num_classes, 3).
    """

    return loadmat(colormap_path)[colormap_key]


def get_preprocess_fn(
    method: Optional[int],
) -> Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]]:

    """
    Resolve the `preprocessing.method` config value to a preprocessing function.
    A number n selects `preprocess_image_n`, except 6, which selects
    `preprocess_retinal_image`. None disables preprocessing.

    Args:
        method: config value — None or an experiment number from 0 to 7.

    Returns:
        The matching function from `preprocessing.py`, or None.

    Raises:
        ValueError: if no preprocessing function matches `method`.
    """

    if method is None:
        return None

    name = "preprocess_retinal_image" if method == 6 else f"preprocess_image_{method}"
    if type(method) is not int or not hasattr(preprocessing, name):
        raise ValueError(
            f"Unknown preprocessing method '{method}'. "
            f"Use a number from 0 to 7, or null, in 'preprocessing.method'."
        )

    return getattr(preprocessing, name)


def load_image_mask(
    image_path: str, mask_path: str, size: int
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Read an image/mask pair from disk and resize both to a square side length.
    The image is read as BGR; the mask is read as the first frame of the file,
    which also handles GIF masks, and resized with nearest-neighbour
    interpolation so it stays binary. Masks that decode with colour channels,
    as palette GIFs do, are reduced to their first channel.

    Args:
        image_path: path to the fundus image.
        mask_path: path to its vessel mask.
        size: side length both are resized to.

    Returns:
        (image, mask) —> BGR uint8 image of shape (size, size, 3) and uint8 mask
        of shape (size, size).

    Raises:
        FileNotFoundError: if the image cannot be read.
    """

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    mask = imageio.mimread(mask_path)[0]
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    image = cv2.resize(image, (size, size))
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    return image, mask


def create_patches(
    img_paths: List[str],
    mask_paths: List[str],
    method: Optional[int] = None,
    resize: int = 896,
    patch_size: int = 224,
    step: Tuple[int, int] = (224, 224),
    scale: float = 255.0,
    num_classes: int = 2,
    num_augmented: int = 0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Turn a list of image/mask files into normalised patch arrays ready to train on.
    Each image is resized, enhanced by the selected preprocessing method, and cut
    into patches on a sliding grid; its mask is cut on the same grid. Augmented
    patches are then appended if requested, and the masks are thresholded at 0.5
    and one-hot encoded. Single-channel preprocessing outputs are stacked to three
    channels so every method feeds the same model input shape.

    Args:
        img_paths: fundus image paths.
        mask_paths: vessel mask paths, index-aligned with `img_paths`.
        method: preprocessing experiment number (see `get_preprocess_fn`), or None.
        resize: side length images are resized to, rounded down to a multiple of
            `patch_size`.
        patch_size: side length of each square patch.
        step: (vertical_step, horizontal_step) between patch origins.
        scale: divisor mapping pixel values to [0, 1].
        num_classes: number of classes in the one-hot masks.
        num_augmented: augmented patches to append; use 0 for test data.
        seed: seed for choosing which patches to augment.

    Returns:
        (image_patches, mask_patches) —> arrays of shape (N, patch_size, patch_size, 3)
        and (N, patch_size, patch_size, num_classes).

    Raises:
        ValueError: if `img_paths` and `mask_paths` differ in length.
    """

    if len(img_paths) != len(mask_paths):
        raise ValueError(
            f"Mismatched path counts: {len(img_paths)} images vs {len(mask_paths)} masks"
        )

    preprocess_fn = get_preprocess_fn(method)
    size = (resize // patch_size) * patch_size
    img_patches = []
    mask_patches = []

    for image_path, mask_path in zip(img_paths, mask_paths):
        image, mask = load_image_mask(image_path, mask_path, size)

        if preprocess_fn is not None:
            _, image = preprocess_fn(image)
            if image.ndim == 2:
                image = cv2.merge([image, image, image])

        img_patches += sliding_window(image, patch_size, step, scale)
        mask_patches += sliding_window(mask, patch_size, step, scale)

    if num_augmented > 0:
        add_augmented_patches(img_patches, mask_patches, num_augmented, seed)

    img_patches = np.array(img_patches)
    mask_patches = to_categorical(
        (np.array(mask_patches) >= 0.5).astype(np.uint8), num_classes=num_classes
    )

    return img_patches, mask_patches


def prepare_dataset(
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Build train and test patch arrays from the settings in configs/default.yaml.
    Both splits get the same preprocessing; only the training split is augmented.

    Args:
        config: parsed config with "data", "preprocessing" and, optionally,
            "augmentation" sections.

    Returns:
        (train_img_patches, train_mask_patches, test_img_patches, test_mask_patches)
        —> arrays as returned by `create_patches`.
    """

    data_config = config["data"]
    preprocessing_config = config["preprocessing"]
    augmentation_config = config.get("augmentation", {})

    train_img_paths, train_mask_paths, test_img_paths, test_mask_paths = prepare_data_paths(
        data_config["root"]
    )
    patch_settings = {
        "method": preprocessing_config.get("method"),
        "resize": preprocessing_config["resize"],
        "patch_size": preprocessing_config["patch_size"],
        "step": tuple(preprocessing_config["step"]),
        "scale": preprocessing_config["scale"],
        "num_classes": data_config["num_classes"],
    }

    train_img_patches, train_mask_patches = create_patches(
        train_img_paths,
        train_mask_paths,
        num_augmented=augmentation_config.get("num_augmented", 0),
        seed=augmentation_config.get("seed"),
        **patch_settings,
    )
    test_img_patches, test_mask_patches = create_patches(
        test_img_paths, test_mask_paths, **patch_settings
    )

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
