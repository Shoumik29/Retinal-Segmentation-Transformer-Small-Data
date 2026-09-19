"""
Preprocessing utilities for retinal fundus image segmentation.
Responsibilities:
    - Enhance fundus images (CLAHE, gamma, LAB, channel fusion, PCA, N4 bias correction)
    - Detect the circular field of view and neutralise the black border ring
    - Randomly jitter colour and gamma for augmentation
"""

import random
from typing import Callable, Dict, Tuple
import cv2
import numpy as np
import SimpleITK as sitk
from sklearn.decomposition import PCA


def clahe_equalized(
    image: np.ndarray,
    clip_limit: float = 1.5,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:

    """
    Apply Contrast-Limited Adaptive Histogram Equalization to a single channel.

    Args:
        image: single-channel uint8 image.
        clip_limit: contrast clipping threshold.
        tile_grid_size: (rows, cols) of the equalization tiles.

    Returns:
        Equalized single-channel uint8 image.
    """

    if image.ndim != 2:
        raise ValueError(
            f"clahe_equalized expects a single-channel image, got shape {image.shape}"
        )
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    return clahe.apply(image)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:

    """
    Apply gamma correction by raising normalized intensities to `gamma`.

    Args:
        image: uint8 image, any number of channels.
        gamma: exponent; < 1 brightens, > 1 darkens.

    Returns:
        Gamma-corrected uint8 image.
    """

    return np.uint8(cv2.pow(image / 255.0, gamma) * 255)


def gamma_correction_1(image: np.ndarray, gamma: float = 0.5) -> np.ndarray:

    """
    Apply gamma correction through a 256-entry lookup table.

    Faster than `gamma_correction` for large images, and uses the inverse
    exponent convention (< 1 darkens, > 1 brightens).

    Args:
        image: uint8 image, any number of channels.
        gamma: exponent; the LUT is built from 1 / gamma.

    Returns:
        Gamma-corrected uint8 image.
    """

    inv_gamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]
    ).astype(np.uint8)

    return cv2.LUT(image, table)


def normalize_image(image: np.ndarray) -> np.ndarray:

    """
    Min-max normalize an image and rescale it back to the 8-bit range.

    Args:
        image: image of any dtype.

    Returns:
        uint8 image spanning the full [0, 255] range.
    """

    norm_image = cv2.normalize(
        image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    return (255 * norm_image).astype(np.uint8)


def field_of_view_mask(image: np.ndarray) -> np.ndarray:

    """
    Locate the black border ring surrounding the circular field of view.

    Otsu-thresholds the red channel, which separates the illuminated retina
    from the unexposed corners more reliably than the green or blue channels.

    Args:
        image: BGR uint8 image.

    Returns:
        uint8 mask where 255 marks border (non-FOV) pixels and 0 marks retina.
    """

    red_channel = image[:, :, 2]
    _, mask = cv2.threshold(
        red_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return cv2.bitwise_not(mask)


def fill_border_ring(
    image: np.ndarray,
    mask: np.ndarray,
    smooth_edges: bool = True,
    blur_sigma: float = 7.0,
) -> np.ndarray:

    """
    Replace the black border ring with the mean retinal colour.

    Filling the ring before contrast enhancement stops the hard black/retina
    edge from dominating the CLAHE histograms near the FOV boundary.

    Args:
        image: BGR uint8 image.
        mask: border mask from `field_of_view_mask` (255 = border).
        smooth_edges: blend the filled ring with a Gaussian blur of the image.
        blur_sigma: Gaussian sigma used when `smooth_edges` is True.

    Returns:
        BGR uint8 copy with the border ring filled.
    """

    filled = image.copy()

    retina_pixels = np.where(mask == 0)
    average_color = np.mean(image[retina_pixels], axis=0)
    filled[mask == 255] = average_color

    if smooth_edges:
        blurred = cv2.GaussianBlur(
            filled, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma
        )
        filled[mask == 255] = blurred[mask == 255]

    return filled


def apply_morphological_operations(
    channel: np.ndarray, kernel: np.ndarray
) -> np.ndarray:

    """
    Emphasise vessel-like structures with a bottom-hat followed by a top-hat.

    Args:
        channel: single-channel uint8 image.
        kernel: structuring element, e.g. from `cv2.getStructuringElement`.

    Returns:
        Single-channel uint8 image with background illumination removed.
    """

    bottom_hat = cv2.morphologyEx(channel, cv2.MORPH_BLACKHAT, kernel)

    return cv2.morphologyEx(bottom_hat, cv2.MORPH_TOPHAT, kernel)


def apply_n4_bias_field_correction(
    image: np.ndarray, iterations: int = 25, levels: int = 5
) -> np.ndarray:

    """
    Correct slow-varying illumination bias with the N4 algorithm.

    Args:
        image: single-channel uint8 image.
        iterations: maximum iterations per resolution level.
        levels: number of multi-resolution levels.

    Returns:
        Bias-corrected single-channel uint8 image.
    """

    float_image = sitk.GetImageFromArray(image.astype(np.float32))

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([iterations] * levels)
    corrected = corrector.Execute(float_image)

    return sitk.GetArrayFromImage(corrected).astype(np.uint8)


def _lab_clahe_with_fov(
    image: np.ndarray, smooth_edges: bool, blur_sigma: float = 7.0
) -> np.ndarray:

    """
    Fill the FOV border, equalize the LAB lightness channel, restore the border.

    Shared body of `preprocess_image_2`, `preprocess_image_5` and
    `preprocess_image_7`, which differ only in whether the border is smoothed.

    Args:
        image: BGR uint8 image.
        smooth_edges: blend the filled border ring with a Gaussian blur.
        blur_sigma: Gaussian sigma used when `smooth_edges` is True.

    Returns:
        BGR uint8 image with a black border ring and an enhanced interior.
    """

    mask = field_of_view_mask(image)
    filled = fill_border_ring(
        image, mask, smooth_edges=smooth_edges, blur_sigma=blur_sigma
    )

    lab_image = cv2.cvtColor(filled, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    enhanced_lab = cv2.merge([clahe_equalized(l_channel), a_channel, b_channel])

    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced[mask == 255] = [0, 0, 0]

    return enhanced


def preprocess_image_0(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    CLAHE-equalized green channel on its own.

    The green channel carries the strongest vessel-to-background contrast in
    fundus photography, so this is the lightest-weight single-channel input.

    Args:
        image: BGR uint8 image.

    Returns:
        (image, enhanced) —> original BGR image, single-channel uint8 enhancement.
    """

    return image, clahe_equalized(image[:, :, 1])


def preprocess_image_1(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    CLAHE-equalized green channel merged back with the original B and R.

    Args:
        image: BGR uint8 image.

    Returns:
        (image, enhanced) —> original and enhanced BGR uint8 images.
    """

    enhanced_green = clahe_equalized(image[:, :, 1])

    return image, cv2.merge([image[:, :, 0], enhanced_green, image[:, :, 2]])


def preprocess_image_2(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    LAB lightness CLAHE after filling the FOV border ring, without smoothing.

    Args:
        image: BGR uint8 image.

    Returns:
        (image, enhanced) —> original and enhanced BGR uint8 images.
    """

    return image, _lab_clahe_with_fov(image, smooth_edges=False)


def preprocess_image_3(
    image: np.ndarray,
    gamma: float = 1.2,
    weights: Tuple[float, float, float] = (0.114, 0.299, 0.587),
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Fuse the BGR channels into one grayscale image, then equalize and gamma-correct.

    Args:
        image: BGR uint8 image.
        gamma: exponent passed to `gamma_correction`.
        weights: per-channel fusion weights, ordered (B, G, R). The defaults are
            the values used in the original experiments; the green and red
            weights are swapped relative to the standard ITU-R BT.601 luma
            weights (0.114, 0.587, 0.299).

    Returns:
        (image, enhanced) —> original BGR image, single-channel uint8 enhancement.
    """

    b_channel, g_channel, r_channel = cv2.split(image)
    weight_b, weight_g, weight_r = weights

    fused = (
        b_channel * weight_b + g_channel * weight_g + r_channel * weight_r
    ).astype(np.uint8)

    return image, gamma_correction(clahe_equalized(fused), gamma)


def preprocess_image_4(
    image: np.ndarray, kernel_size: Tuple[int, int] = (150, 150)
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Reduce the morphologically-filtered RGB channels to one channel via PCA.

    Each channel is background-suppressed and equalized independently, then PCA
    projects the three onto the axis of greatest variance — which is dominated
    by the vessel structure the morphological step isolated.

    Args:
        image: BGR uint8 image.
        kernel_size: ellipse size for the morphological structuring element.

    Returns:
        (image, enhanced) —> original BGR image, and BGR uint8 image with the
        PCA projection in the green channel.
    """

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    channel_r, channel_g, channel_b = cv2.split(rgb_image)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    filtered = [
        clahe_equalized(apply_morphological_operations(channel, kernel))
        for channel in (channel_r, channel_g, channel_b)
    ]

    stacked = np.vstack([channel.flatten() for channel in filtered]).T
    projected = PCA(n_components=1).fit_transform(stacked)
    vessel_map = projected[:, 0].reshape(filtered[0].shape)
    vessel_map = cv2.normalize(vessel_map, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    return image, cv2.merge([image[:, :, 0], vessel_map, image[:, :, 2]])


def preprocess_image_5(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    LAB lightness CLAHE after filling and Gaussian-smoothing the FOV border ring.

    Args:
        image: BGR uint8 image.

    Returns:
        (image, enhanced) —> original and enhanced BGR uint8 images.
    """

    return image, _lab_clahe_with_fov(image, smooth_edges=True)


def preprocess_image_7(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    LAB lightness CLAHE after filling and Gaussian-smoothing the FOV border ring.

    Produces the same output as `preprocess_image_5`; the N4 follow-up step
    from the original experiment was commented out there and is not applied.

    Args:
        image: BGR uint8 image.

    Returns:
        (image, enhanced) —> original and enhanced BGR uint8 images.
    """

    return image, _lab_clahe_with_fov(image, smooth_edges=True)


def preprocess_retinal_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    Bias-correct the green channel, equalize it, and merge it back into BGR.

    Args:
        image: BGR uint8 image.

    Returns:
        (image, enhanced) —> original and enhanced BGR uint8 images.
    """

    corrected_green = apply_n4_bias_field_correction(image[:, :, 1])

    return image, cv2.merge(
        [image[:, :, 0], clahe_equalized(corrected_green), image[:, :, 2]]
    )


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
