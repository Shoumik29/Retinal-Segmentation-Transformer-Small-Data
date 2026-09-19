"""
Augmentation utilities for retinal fundus image segmentation.
Responsibilities:
    - Randomly jitter colour and gamma of fundus images
    - Apply random geometric and photometric augmentations to image/mask pairs
    - Expand image/mask pairs into fixed augmented variants and write them to disk
    - Append randomly augmented copies to lists of normalised patches
"""

import os
import random
from typing import List, Optional, Tuple
import albumentations as A
import cv2
import numpy as np

from retina_seg.data.preprocessing import gamma_correction


def random_hsv_gamma_adjustment(image: np.ndarray) -> np.ndarray:

    """
    Randomly perturb hue, saturation, value, and gamma.
    Uses Python's `random` module, as the original experiments did, so results
    reproduce under `random.seed(...)`.

    Args:
        image: BGR uint8 image.

    Returns:
        Colour-jittered BGR uint8 image.
    """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv_image)

    hue = np.mod(hue + random.randint(-10, 10), 180).astype(np.uint8)
    saturation = np.clip(saturation * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
    value = np.clip(value * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)

    adjusted = cv2.cvtColor(cv2.merge([hue, saturation, value]), cv2.COLOR_HSV2BGR)

    return gamma_correction(adjusted, random.uniform(0.8, 1.2))


def apply_augmentations(
    image: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Apply random geometric and photometric augmentations to an image/mask pair.
    Rotation and flips are applied jointly so the mask stays aligned with the
    image; brightness, contrast and hue/saturation shifts are applied to the
    image only.

    Args:
        image: BGR uint8 image.
        mask: single-channel uint8 mask, same height and width as `image`.

    Returns:
        (augmented_image, augmented_mask) —> index-aligned.
    """

    geometric = A.Compose([
        A.Rotate(limit=(-30, 30), p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])
    photometric = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
        A.HueSaturationValue(hue_shift_limit=(-5, 5), sat_shift_limit=(-20, 20), val_shift_limit=(-20, 20), p=0.5),
    ])

    augmented = geometric(image=image, mask=mask)
    augmented_image = photometric(image=augmented["image"])["image"]

    return augmented_image, augmented["mask"]


def _save_image_mask_pairs(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    save_path: str,
    name: str,
    extension: str,
) -> None:

    """
    Write index-aligned image/mask variants under `save_path`.
    A single pair is written as `<name><extension>`; several pairs are written
    as `<name>_<index><extension>`. Images go to `save_path/images` and masks
    to `save_path/masks`, both created if missing.

    Args:
        images: image variants to write.
        masks: mask variants to write, index-aligned with `images`.
        save_path: destination root directory.
        name: file stem shared by every variant.
        extension: output file extension, including the leading dot.

    Raises:
        ValueError: if an image or mask fails to encode.
    """

    image_dir = os.path.join(save_path, "images")
    mask_dir = os.path.join(save_path, "masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for index, (image, mask) in enumerate(zip(images, masks)):
        file_name = f"{name}{extension}" if len(images) == 1 else f"{name}_{index}{extension}"
        image_path = os.path.join(image_dir, file_name)
        mask_path = os.path.join(mask_dir, file_name)

        if not cv2.imwrite(image_path, image):
            raise ValueError(f"Failed to write image to {image_path}")
        if not cv2.imwrite(mask_path, mask):
            raise ValueError(f"Failed to write mask to {mask_path}")


def augment_data(
    images: np.ndarray,
    masks: np.ndarray,
    save_path: str,
    names: str,
    augment: bool = True,
    extension: str = ".jpg",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    """
    Expand one image/mask pair into five fixed variants and write them to disk.
    The variants are the original, a random rotation within ±60°, a horizontal
    flip, a vertical flip, and a hue/saturation/value shift. The colour shift
    reuses the original mask because it does not move any pixels.

    Args:
        images: a single BGR uint8 image.
        masks: its single-channel uint8 mask.
        save_path: destination root; "images" and "masks" subdirectories are created.
        names: path or filename of the source image; its stem names the outputs.
        augment: expand into the five variants; if False, write the pair unchanged.
        extension: output file extension. Defaults to ".jpg" to match the original
            experiments, but JPEG is lossy and blurs binary mask edges — use ".png"
            for new datasets.

    Returns:
        (image_variants, mask_variants) —> index-aligned lists as written to disk.
    """

    name = os.path.basename(names).split(".")[0]

    if augment:
        rotated = A.Rotate(limit=(-60, 60), p=1, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0))(image=images, mask=masks)
        h_flipped = A.HorizontalFlip(p=1)(image=images, mask=masks)
        v_flipped = A.VerticalFlip(p=1)(image=images, mask=masks)
        colour_shifted = A.HueSaturationValue(
            hue_shift_limit=(-5, 5), sat_shift_limit=(-20, 20), val_shift_limit=(-20, 20), p=1
        )(image=images)

        image_variants = [images, rotated["image"], h_flipped["image"], v_flipped["image"], colour_shifted["image"]]
        mask_variants = [masks, rotated["mask"], h_flipped["mask"], v_flipped["mask"], masks]
    else:
        image_variants = [images]
        mask_variants = [masks]

    _save_image_mask_pairs(image_variants, mask_variants, save_path, name, extension)

    return image_variants, mask_variants


def augment_data_1(
    images: np.ndarray,
    masks: np.ndarray,
    save_path: str,
    names: str,
    augment: bool = True,
    extension: str = ".jpg",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    """
    Expand one image/mask pair into seven fixed variants and write them to disk.
    The variants are the original, a random rotation within ±30°, a horizontal
    flip, a vertical flip, a brightness/contrast increase, a hue/saturation/value
    shift, and an elastic deformation. The brightness and colour variants reuse
    the original mask because they do not move any pixels.

    Args:
        images: a single BGR uint8 image.
        masks: its single-channel uint8 mask.
        save_path: destination root; "images" and "masks" subdirectories are created.
        names: path or filename of the source image; its stem names the outputs.
        augment: expand into the seven variants; if False, write the pair unchanged.
        extension: output file extension. Defaults to ".jpg" to match the original
            experiments, but JPEG is lossy and blurs binary mask edges — use ".png"
            for new datasets.

    Returns:
        (image_variants, mask_variants) —> index-aligned lists as written to disk.
    """

    name = os.path.basename(names).split(".")[0]

    if augment:
        rotated = A.Rotate(limit=(-30, 30), p=1, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0))(image=images, mask=masks)
        h_flipped = A.HorizontalFlip(p=1)(image=images, mask=masks)
        v_flipped = A.VerticalFlip(p=1)(image=images, mask=masks)
        brightened = A.RandomBrightnessContrast(brightness_limit=(0.1, 0.2), contrast_limit=(0.1, 0.2), p=1)(image=images)
        colour_shifted = A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=0.05, val_shift_limit=0.05, p=1)(image=images)
        elastic = A.ElasticTransform(alpha=1, sigma=8, alpha_affine=8, p=1)(image=images, mask=masks)

        image_variants = [
            images, rotated["image"], h_flipped["image"], v_flipped["image"],
            brightened["image"], colour_shifted["image"], elastic["image"],
        ]
        mask_variants = [
            masks, rotated["mask"], h_flipped["mask"], v_flipped["mask"],
            masks, masks, elastic["mask"],
        ]
    else:
        image_variants = [images]
        mask_variants = [masks]

    _save_image_mask_pairs(image_variants, mask_variants, save_path, name, extension)

    return image_variants, mask_variants


def add_augmented_patches(
    image: List[np.ndarray],
    mask: List[np.ndarray],
    num_augmented: int,
    seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    """
    Append strongly augmented copies of randomly chosen patches, in place.
    Each copy picks a random patch, applies flips, 90° rotations, shift/scale/
    rotate and elastic deformation jointly to image and mask (nearest-neighbour
    interpolation keeps the mask binary), then brightness/contrast and Gaussian
    noise to the image only.

    Args:
        image: image patches normalised to [0, 1]; extended in place.
        mask: mask patches normalised to [0, 1], index-aligned with `image`;
            extended in place.
        num_augmented: number of augmented pairs to append.
        seed: seeds NumPy's global RNG before sampling, if given.

    Returns:
        (image, mask) —> the same lists, now `num_augmented` entries longer.

    Raises:
        ValueError: if `image` and `mask` hold different numbers of patches.
    """

    if len(image) != len(mask):
        raise ValueError(
            f"Mismatched patch counts: {len(image)} images vs {len(mask)} masks"
        )
    if seed is not None:
        np.random.seed(seed)

    geometric = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.2, scale_limit=0.3, rotate_limit=45, p=0.8,
            interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT,
        ),
        A.ElasticTransform(p=0.2, interpolation=cv2.INTER_NEAREST),
    ])
    image_only = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    ])

    for _ in range(num_augmented):
        index = np.random.randint(len(image))
        image_uint8 = (image[index] * 255.0).astype(np.uint8)
        mask_uint8 = (mask[index] * 255.0).astype(np.uint8)

        augmented = geometric(image=image_uint8, mask=mask_uint8)
        augmented_image = image_only(image=augmented["image"])["image"]

        image.append(augmented_image.astype(np.float32) / 255.0)
        mask.append(augmented["mask"].astype(np.float32) / 255.0)

    return image, mask
