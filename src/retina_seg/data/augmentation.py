"""
Augmentation for retinal fundus images.
Responsibilities:
    - Random colour and gamma changes on one image
    - Random geometric and colour augmentation on an image/mask pair
    - Save five or seven fixed augmented versions of an image/mask pair
    - Add augmented copies to lists of patches
"""

import os
import random
from typing import List, Optional, Tuple
import albumentations as A
import cv2
import numpy as np
from albumentations import (
    ElasticTransform,
    HorizontalFlip,
    HueSaturationValue,
    RandomBrightnessContrast,
    Rotate,
    VerticalFlip,
)


def random_hsv_gamma_adjustment(image: np.ndarray) -> np.ndarray:

    """
    Randomly change hue, saturation, value and gamma.

    Args:
        image: BGR image.

    Returns:
        Adjusted image.
    """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    delta_h = random.randint(-10, 10)
    delta_s = random.uniform(0.8, 1.2)
    delta_v = random.uniform(0.8, 1.2)

    h = np.mod(h + delta_h, 180).astype(np.uint8)
    s = np.clip(s * delta_s, 0, 255).astype(np.uint8)
    v = np.clip(v * delta_v, 0, 255).astype(np.uint8)

    hsv_adjusted = cv2.merge([h, s, v])
    adjusted_image = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

    gamma = random.uniform(0.8, 1.2)
    gamma_corrected_image = np.uint8(cv2.pow(adjusted_image / 255.0, gamma) * 255)

    return gamma_corrected_image


def random_augment_image_mask(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    Random rotation and flips on image and mask, colour changes on image only.

    Args:
        image: BGR image.
        mask: mask of the image.

    Returns:
        (augmented_image, augmented_mask)
    """

    transform1 = A.Compose([
        A.Rotate(limit=(-30, 30), p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])

    transform2 = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
        A.HueSaturationValue(hue_shift_limit=(-5, 5), sat_shift_limit=(-20, 20), val_shift_limit=(-20, 20), p=0.5),
    ])

    augmented = transform1(image=image, mask=mask)
    augmented_image = augmented['image']
    augmented_mask = augmented['mask']

    image_aumentation = transform2(image=augmented_image)
    augmented_image = image_aumentation['image']

    return augmented_image, augmented_mask


def save_five_augmentations(images: np.ndarray, masks: np.ndarray, save_path: str, names: str, augment: bool = True) -> None:

    """
    Create 5 augmented versions of an image and mask and save them.

    Args:
        images: BGR image.
        masks: mask of the image.
        save_path: output directory with images/ and masks/ folders.
        names: source image path, used for file names.
        augment: save only the original if False.
    """

    name = names.split("/")[-1].split(".")[0]

    x = images
    y = masks

    if augment == True:

        aug = Rotate(limit=(-60, 60), p=1, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        augmented = aug(image=x, mask=y)
        x1 = augmented["image"]
        y1 = augmented["mask"]

        aug = HorizontalFlip(p=1)
        augmented = aug(image=x, mask=y)
        x2 = augmented["image"]
        y2 = augmented["mask"]

        aug = VerticalFlip(p=1)
        augmented = aug(image=x, mask=y)
        x3 = augmented["image"]
        y3 = augmented["mask"]

        aug = HueSaturationValue(hue_shift_limit=(-5, 5), sat_shift_limit=(-20, 20), val_shift_limit=(-20, 20), p=1)
        augmented = aug(image=x)
        x4 = augmented["image"]

        X = [x, x1, x2, x3, x4]
        Y = [y, y1, y2, y3, y]

    else:
        X = [x]
        Y = [y]

    index = 0
    for i, m in zip(X, Y):

        if len(X) == 1:
            tmp_image_name = f"{name}.jpg"
            tmp_mask_name = f"{name}.jpg"
        else:
            tmp_image_name = f"{name}_{index}.jpg"
            tmp_mask_name = f"{name}_{index}.jpg"

        image_path = os.path.join(save_path, "images", tmp_image_name)
        mask_path = os.path.join(save_path, "masks", tmp_mask_name)

        print(image_path)
        cv2.imwrite(image_path, i)
        cv2.imwrite(mask_path, m)

        index += 1


def save_seven_augmentations(
    images: np.ndarray, masks: np.ndarray, save_path: str, names: str, augment: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    """
    Create 7 augmented versions of an image and mask and save them.

    Args:
        images: BGR image.
        masks: mask of the image.
        save_path: output directory with images/ and masks/ folders.
        names: source image path, used for file names.
        augment: save only the original if False.

    Returns:
        (X, Y) —> saved images and masks.
    """

    name = names.split("/")[-1].split(".")[0]

    x = images
    y = masks

    if augment == True:

        aug = Rotate(limit=(-30, 30), p=1, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        augmented = aug(image=x, mask=y)
        x1 = augmented["image"]
        y1 = augmented["mask"]

        aug = HorizontalFlip(p=1)
        augmented = aug(image=x, mask=y)
        x2 = augmented["image"]
        y2 = augmented["mask"]

        aug = VerticalFlip(p=1)
        augmented = aug(image=x, mask=y)
        x3 = augmented["image"]
        y3 = augmented["mask"]

        aug = RandomBrightnessContrast(brightness_limit=(0.1, 0.2), contrast_limit=(0.1, 0.2), p=1)
        augmented = aug(image=x)
        x4 = augmented["image"]

        aug = HueSaturationValue(hue_shift_limit=5, sat_shift_limit=13, val_shift_limit=13, p=1)
        augmented = aug(image=x)
        x5 = augmented["image"]

        aug = ElasticTransform(alpha=1, sigma=8, alpha_affine=8, p=1)
        augmented = aug(image=x, mask=y)
        x6 = augmented["image"]
        y6 = augmented["mask"]

        X = [x, x1, x2, x3, x4, x5, x6]
        Y = [y, y1, y2, y3, y, y, y6]

    else:
        X = [x]
        Y = [y]

    index = 0
    for i, m in zip(X, Y):

        if len(X) == 1:
            tmp_image_name = f"{name}.jpg"
            tmp_mask_name = f"{name}.jpg"
        else:
            tmp_image_name = f"{name}_{index}.jpg"
            tmp_mask_name = f"{name}_{index}.jpg"

        image_path = os.path.join(save_path, "images", tmp_image_name)
        mask_path = os.path.join(save_path, "masks", tmp_mask_name)

        print(image_path)
        cv2.imwrite(image_path, i)
        cv2.imwrite(mask_path, m)

        index += 1

    return X, Y


def add_augmented_patches(
    image: List[np.ndarray], mask: List[np.ndarray], num_augmented: int, seed: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    """
    Add strongly augmented copies of random patches to the lists.

    Args:
        image: image patches in 0-1 range.
        mask: mask patches in 0-1 range.
        num_augmented: number of patches to add.
        seed: random seed.

    Returns:
        (image, mask) —> same lists with new patches added.
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    assert len(image) == len(mask), "Images and masks must be aligned"

    geo_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.2, scale_limit=0.3, rotate_limit=45, p=0.8,
            interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.ElasticTransform(p=0.2, interpolation=cv2.INTER_NEAREST),
    ], additional_targets={'mask': 'mask'})

    image_only_transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    ])

    for _ in range(num_augmented):
        idx = np.random.randint(len(image))
        image_np = (image[idx] * 255.).astype(np.uint8)
        mask_np = (mask[idx] * 255.).astype(np.uint8)

        geo_aug = geo_transform(image=image_np, mask=mask_np)
        aug_img = geo_aug['image']
        aug_mask = geo_aug['mask']

        img_aug = image_only_transform(image=aug_img)['image']

        img_aug = img_aug.astype(np.float32) / 255.
        mask_aug = aug_mask.astype(np.float32) / 255.

        image.append(img_aug)
        mask.append(mask_aug)

    return image, mask
