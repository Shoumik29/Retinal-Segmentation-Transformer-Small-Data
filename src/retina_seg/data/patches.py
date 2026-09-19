"""
Patch extraction for retinal fundus images.
Responsibilities:
    - Split images and masks into patches
    - Random crop of image, mask and FOV patches
    - Create train/test patches from image and mask paths
    - Rebuild an image from its patches
"""

from typing import List, Optional, Tuple
import cv2
import imageio
import numpy as np
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from retina_seg.data.augmentation import add_augmented_patches
from retina_seg.data.preprocessing import get_preprocess_fn


def sliding_window(
    image: np.ndarray, patch_size: int, step: Tuple[int, int], scale: float = 255.0
) -> List[np.ndarray]:

    """
    Split an image into patches using a sliding window.

    Args:
        image: input image.
        patch_size: size of each patch.
        step: (vertical, horizontal) step.
        scale: value to divide patches by.

    Returns:
        List of patches.
    """

    patches = []
    height, width = image.shape[:2]
    patch_height = patch_size
    patch_width = patch_size
    vertical_step, horizontal_step = step
    for y in range(0, height - patch_height + 1, vertical_step):
        for x in range(0, width - patch_width + 1, horizontal_step):
            patch = image[y:y+patch_height, x:x+patch_width]
            patch = patch/scale

            patches.append(patch)

    return patches


def random_crop_image_mask_pairs(
    image: np.ndarray,
    mask: np.ndarray,
    fov: np.ndarray,
    patch_size: int,
    num_patches: int,
    seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

    """
    Crop image, mask and FOV patches at the same random positions.

    Args:
        image: input image (H, W, C).
        mask: mask of the image (H, W) or (H, W, 1).
        fov: FOV mask of the image.
        patch_size: size of each patch.
        num_patches: number of patches to crop.
        seed: random seed.

    Returns:
        (image_patches, mask_patches, fov_patches)
    """

    if seed is not None:
        np.random.seed(seed)

    height, width = image.shape[:2]
    image_patches = []
    mask_patches = []
    fov_patches = []

    for _ in range(num_patches):
        y = np.random.randint(0, height - patch_size + 1)
        x = np.random.randint(0, width - patch_size + 1)

        img_patch = image[y:y+patch_size, x:x+patch_size]
        mask_patch = mask[y:y+patch_size, x:x+patch_size]
        fov_patch = fov[y:y+patch_size, x:x+patch_size]

        img_patch = img_patch / 255.0
        mask_patch = mask_patch / 255.0
        fov_patch = fov_patch / 255.0

        image_patches.append(img_patch)
        mask_patches.append(mask_patch)
        fov_patches.append(fov_patch)

    return image_patches, mask_patches, fov_patches


def create_patches(
    img_paths: List[str],
    mask_paths: List[str],
    patch_size: int = 224,
    res: int = 896,
    step: Tuple[int, int] = (224, 224),
    scale: float = 255.0,
    method: Optional[int] = None,
    num_augmented: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Read images and masks, resize, preprocess and split them into patches.

    Args:
        img_paths: image paths.
        mask_paths: mask paths.
        patch_size: size of each patch.
        res: size images are resized to before patching.
        step: (vertical, horizontal) step.
        scale: value to divide pixels by.
        method: preprocessing method number, or None.
        num_augmented: augmented patches to add, 0 for test data.

    Returns:
        (img_patches, mask_patches) —> masks are one-hot encoded.
    """

    img_patches = []
    mask_patches = []
    preprocess_fn = get_preprocess_fn(method)

    for i in tqdm(range(len(img_paths))):
        image = cv2.imread(img_paths[i])
        mask = imageio.mimread(mask_paths[i])[0]
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        image = cv2.resize(image, (res, res))
        mask = cv2.resize(mask, (res, res), interpolation=cv2.INTER_NEAREST)

        SIZE_X = (image.shape[0] // patch_size) * patch_size
        SIZE_Y = (image.shape[1] // patch_size) * patch_size

        image = cv2.resize(image, (SIZE_Y, SIZE_X))
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)

        if preprocess_fn is not None:
            _, image = preprocess_fn(image)
            if image.ndim == 2:
                image = cv2.merge([image, image, image])

        image_patches = sliding_window(image, patch_size, step, scale)
        mask_patches_i = sliding_window(mask, patch_size, step, scale)

        img_patches += image_patches
        mask_patches += mask_patches_i

    if num_augmented > 0:
        img_patches, mask_patches = add_augmented_patches(img_patches, mask_patches, num_augmented)

    img_patches = np.array(img_patches)
    mask_patches = np.array(mask_patches)

    mask_patches = to_categorical((mask_patches >= 0.5).astype(np.uint8), num_classes=2)

    return img_patches, mask_patches


def reconstruct_image(
    patches: List[np.ndarray],
    image_shape: Tuple[int, int],
    patch_size: int,
    step: Tuple[int, int],
    ch: int,
    patch_index: int,
) -> np.ndarray:

    """
    Rebuild an image from its patches.

    Args:
        patches: list of patches.
        image_shape: (height, width) of the image.
        patch_size: size of each patch.
        step: (vertical, horizontal) step.
        ch: number of channels.
        patch_index: index of the first patch.

    Returns:
        Rebuilt image.
    """

    height, width = image_shape[:2]
    patch_height = patch_size
    patch_width = patch_size
    vertical_step, horizontal_step = step

    reconstructed_image = np.zeros((height, width, ch), dtype=np.float32)

    for y in range(0, height - patch_height + 1, vertical_step):
        for x in range(0, width - patch_width + 1, horizontal_step):
            if patch_index < len(patches):
                reconstructed_image[y:y+patch_height, x:x+patch_width] += patches[patch_index]
                patch_index += 1
            else:
                break

    return reconstructed_image
