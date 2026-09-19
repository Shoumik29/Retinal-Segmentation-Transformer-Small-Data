"""
Patch extraction and reconstruction for retinal fundus image segmentation.
Responsibilities:
    - Cut images and masks into normalised patches on a sliding grid
    - Randomly crop aligned image/mask/FOV patches
    - Reassemble a full image from its grid of patches
"""

from typing import List, Optional, Tuple
import numpy as np


def sliding_window(
    image: np.ndarray,
    patch_size: int,
    step: Tuple[int, int],
    scale: float = 255.0,
) -> List[np.ndarray]:

    """
    Cut an image into square patches on a regular grid, row by row.
    Patches that would run past the right or bottom edge are skipped, so an
    image whose sides are multiples of `patch_size` is covered exactly when
    `step` equals `(patch_size, patch_size)`.

    Args:
        image: image of shape (H, W) or (H, W, C).
        patch_size: side length of each square patch.
        step: (vertical_step, horizontal_step) between patch origins.
        scale: divisor applied to every patch, e.g. 255.0 to map uint8 to [0, 1].

    Returns:
        List of patches, each (patch_size, patch_size) or (patch_size, patch_size, C),
        ordered top-to-bottom then left-to-right.
    """

    height, width = image.shape[:2]
    vertical_step, horizontal_step = step
    patches = []

    for y in range(0, height - patch_size + 1, vertical_step):
        for x in range(0, width - patch_size + 1, horizontal_step):
            patches.append(image[y:y + patch_size, x:x + patch_size] / scale)

    return patches


def random_crop_image_mask_pairs(
    image: np.ndarray,
    mask: np.ndarray,
    fov: np.ndarray,
    patch_size: int,
    num_patches: int,
    seed: Optional[int] = None,
    scale: float = 255.0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

    """
    Randomly crop aligned image, mask and FOV patches at the same positions.

    Args:
        image: image of shape (H, W, C).
        mask: mask of shape (H, W) or (H, W, 1), aligned with `image`.
        fov: field-of-view mask of shape (H, W), aligned with `image`.
        patch_size: side length of each square patch.
        num_patches: number of patches to crop.
        seed: seeds NumPy's global RNG before sampling, if given.
        scale: divisor applied to every patch, e.g. 255.0 to map uint8 to [0, 1].

    Returns:
        (image_patches, mask_patches, fov_patches) —> index-aligned lists of
        `num_patches` patches each.
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

        image_patches.append(image[y:y + patch_size, x:x + patch_size] / scale)
        mask_patches.append(mask[y:y + patch_size, x:x + patch_size] / scale)
        fov_patches.append(fov[y:y + patch_size, x:x + patch_size] / scale)

    return image_patches, mask_patches, fov_patches


def reconstruct_image(
    patches: List[np.ndarray],
    image_shape: Tuple[int, int],
    patch_size: int,
    step: Tuple[int, int],
    ch: int,
    patch_index: int = 0,
) -> np.ndarray:

    """
    Reassemble one image from a grid of patches, the inverse of `sliding_window`.
    Patches are read from `patches` starting at `patch_index` and placed in the
    same top-to-bottom, left-to-right order `sliding_window` produced them.
    Overlapping patches are summed, not averaged, so `step` should equal
    `(patch_size, patch_size)` for an exact reconstruction.

    Args:
        patches: flat list of patches, possibly holding several images in sequence.
        image_shape: (height, width) of the image to rebuild.
        patch_size: side length of each square patch.
        step: (vertical_step, horizontal_step) used when the patches were cut.
        ch: number of channels in the rebuilt image.
        patch_index: position in `patches` of this image's first patch; for the
            k-th image of a sequence, k * patches_per_image.

    Returns:
        float32 image of shape (height, width, ch).
    """

    height, width = image_shape[:2]
    vertical_step, horizontal_step = step
    reconstructed_image = np.zeros((height, width, ch), dtype=np.float32)

    for y in range(0, height - patch_size + 1, vertical_step):
        for x in range(0, width - patch_size + 1, horizontal_step):
            if patch_index >= len(patches):
                return reconstructed_image
            reconstructed_image[y:y + patch_size, x:x + patch_size] += patches[patch_index]
            patch_index += 1

    return reconstructed_image
