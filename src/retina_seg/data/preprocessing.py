"""
Preprocessing utilities for retinal fundus image segmentation.
Responsibilities:
    - Enhance fundus images (CLAHE, gamma, LAB, channel fusion, PCA, N4 bias correction)
    - Select the preprocessing method set in the config
"""

from typing import Callable, Optional, Tuple
import cv2
import numpy as np
import SimpleITK as sitk
from sklearn.decomposition import PCA


def clahe_equalized(imgs: np.ndarray) -> np.ndarray:

    """
    Apply CLAHE on a single channel image.

    Args:
        imgs: single channel uint8 image.

    Returns:
        Equalized image.
    """

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    imgs_equalized = clahe.apply(imgs)

    return imgs_equalized


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:

    """
    Apply gamma correction using power function.

    Args:
        img: uint8 image.
        gamma: gamma value, below 1 makes image brighter.

    Returns:
        Gamma corrected image.
    """

    gamma_corrected_image = np.uint8(cv2.pow(img / 255.0, gamma) * 255)

    return gamma_corrected_image


def gamma_correction_1(image: np.ndarray, gamma: float = 0.5) -> np.ndarray:

    """
    Apply gamma correction using a lookup table.

    Args:
        image: uint8 image.
        gamma: gamma value, table is built with 1 / gamma.

    Returns:
        Gamma corrected image.
    """

    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def normalize_image(image: np.ndarray) -> np.ndarray:

    """
    Min-max normalize an image to 0-255 range.

    Args:
        image: input image.

    Returns:
        Normalized uint8 image.
    """

    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = (255 * norm_image).astype(np.uint8)

    return norm_image


def preprocess_image_0(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    CLAHE on the green channel.

    Args:
        image: BGR image.

    Returns:
        (image, enhanced_green_channel) —> enhanced is single channel.
    """

    green_channel = image[:, :, 1]
    enhanced_green_channel = clahe_equalized(green_channel)

    return image, enhanced_green_channel


def preprocess_image_1(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    CLAHE on the green channel, merged back with blue and red.

    Args:
        image: BGR image.

    Returns:
        (image, enhanced_image) —> original and enhanced image.
    """

    green_channel = image[:, :, 1]
    enhanced_green_channel = clahe_equalized(green_channel)
    enhanced_image = cv2.merge([image[:, :, 0], enhanced_green_channel, image[:, :, 2]])

    return image, enhanced_image


def preprocess_image_2(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    Fill the black border with the average colour, then CLAHE on the L channel of LAB.

    Args:
        image: BGR image.

    Returns:
        (image, enhanced_rgb_image) —> original and enhanced image.
    """

    img = image.copy()
    red_channel = img[:, :, 2]

    _, mask = cv2.threshold(red_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(mask)

    non_black_pixels = np.where(mask == 0)
    average_color = np.mean(image[non_black_pixels], axis=0)

    img[mask == 255] = average_color

    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    enhanced_l_channel = clahe_equalized(l_channel)
    enhanced_lab_image = cv2.merge([enhanced_l_channel, a_channel, b_channel])
    enhanced_rgb_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

    enhanced_rgb_image[mask == 255] = [0, 0, 0]

    return image, enhanced_rgb_image


def preprocess_image_3(image: np.ndarray, gamma: float = 1.2) -> Tuple[np.ndarray, np.ndarray]:

    """
    Weighted channel fusion, then CLAHE and gamma correction.

    Args:
        image: BGR image.
        gamma: gamma value.

    Returns:
        (image, gamma_corrected_image) —> enhanced is single channel.
    """

    b_channel, g_channel, r_channel = cv2.split(image)

    b_channel_normalized = b_channel * 0.114
    g_channel_normalized = g_channel * 0.299
    r_channel_normalized = r_channel * 0.587

    fused_image = (b_channel_normalized + g_channel_normalized + r_channel_normalized).astype(np.uint8)
    clahe_equalized_image = clahe_equalized(fused_image)
    gamma_corrected_image = gamma_correction(clahe_equalized_image, gamma)

    return image, gamma_corrected_image


def preprocess_image_4(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    Morphological filtering on each channel, combined using PCA.

    Args:
        image: BGR image.

    Returns:
        (image, enhanced_image) —> PCA output is placed in the green channel.
    """

    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    I_R, I_G, I_B = cv2.split(input_image)
    I_R_gray = cv2.cvtColor(cv2.merge([I_R, I_R, I_R]), cv2.COLOR_BGR2GRAY)
    I_G_gray = cv2.cvtColor(cv2.merge([I_G, I_G, I_G]), cv2.COLOR_BGR2GRAY)
    I_B_gray = cv2.cvtColor(cv2.merge([I_B, I_B, I_B]), cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (150, 150))

    I_UR = apply_morphological_operations(I_R_gray, kernel)
    I_UG = apply_morphological_operations(I_G_gray, kernel)
    I_UB = apply_morphological_operations(I_B_gray, kernel)

    I_UR = clahe_equalized(I_UR)
    I_UG = clahe_equalized(I_UG)
    I_UB = clahe_equalized(I_UB)

    vstacked_images = np.vstack((I_UR.flatten(), I_UG.flatten(), I_UB.flatten())).T
    pca = PCA(n_components=1)
    transformed_images = pca.fit_transform(vstacked_images)

    I_fv = transformed_images[:, 0].reshape(I_UR.shape)
    I_fv = cv2.normalize(I_fv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    enhanced_image = cv2.merge([image[:, :, 0], I_fv, image[:, :, 2]])

    return image, enhanced_image


def preprocess_image_5(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    Fill and blur the black border, then CLAHE on the L channel of LAB.

    Args:
        image: BGR image.

    Returns:
        (image, enhanced_rgb_image) —> original and enhanced image.
    """

    img = image.copy()
    red_channel = img[:, :, 2]

    _, mask = cv2.threshold(red_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(mask)

    non_black_pixels = np.where(mask == 0)
    average_color_r = np.mean(img[non_black_pixels][:, 2])
    average_color_g = np.mean(img[non_black_pixels][:, 1])
    average_color_b = np.mean(img[non_black_pixels][:, 0])
    average_color = [average_color_b, average_color_g, average_color_r]

    img[mask == 255] = average_color

    blurred_img = cv2.GaussianBlur(img, (0, 0), sigmaX=7, sigmaY=7)
    img[mask == 255] = blurred_img[mask == 255]

    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    enhanced_l_channel = clahe_equalized(l_channel)
    enhanced_lab_image = cv2.merge([enhanced_l_channel, a_channel, b_channel])
    enhanced_rgb_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

    enhanced_rgb_image[mask == 255] = [0, 0, 0]

    return image, enhanced_rgb_image


def apply_morphological_operations(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    """
    Highlight vessels using black-hat followed by top-hat.

    Args:
        channel: single channel image.
        kernel: structuring element.

    Returns:
        Filtered channel.
    """

    bottom_hat = cv2.morphologyEx(channel, cv2.MORPH_BLACKHAT, kernel)
    top_hat = cv2.morphologyEx(bottom_hat, cv2.MORPH_TOPHAT, kernel)

    return top_hat


def apply_n4_bias_field_correction(image: np.ndarray) -> np.ndarray:

    """
    Remove uneven illumination using N4 bias field correction.

    Args:
        image: single channel image.

    Returns:
        Corrected image.
    """

    float_image = sitk.GetImageFromArray(image.astype(np.float32))

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([25] * 5)
    corrected_image = corrector.Execute(float_image)
    corrected_image = sitk.GetArrayFromImage(corrected_image).astype(np.uint8)

    return corrected_image


def preprocess_image_7(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    Same steps as preprocess_image_5.

    Args:
        image: BGR image.

    Returns:
        (image, enhanced_rgb_image) —> original and enhanced image.
    """

    img = image.copy()
    red_channel = img[:, :, 2]

    _, mask = cv2.threshold(red_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(mask)

    non_black_pixels = np.where(mask == 0)
    average_color_r = np.mean(img[non_black_pixels][:, 2])
    average_color_g = np.mean(img[non_black_pixels][:, 1])
    average_color_b = np.mean(img[non_black_pixels][:, 0])
    average_color = [average_color_b, average_color_g, average_color_r]

    img[mask == 255] = average_color

    blurred_img = cv2.GaussianBlur(img, (0, 0), sigmaX=7, sigmaY=7)
    img[mask == 255] = blurred_img[mask == 255]

    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    enhanced_l_channel = clahe_equalized(l_channel)
    enhanced_lab_image = cv2.merge([enhanced_l_channel, a_channel, b_channel])
    enhanced_rgb_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

    enhanced_rgb_image[mask == 255] = [0, 0, 0]

    return image, enhanced_rgb_image


def preprocess_retinal_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    N4 bias correction and CLAHE on the green channel.

    Args:
        image: BGR image.

    Returns:
        (image, enhanced_image) —> original and enhanced image.
    """

    green_channel = image[:, :, 1]
    corrected_image = apply_n4_bias_field_correction(green_channel)
    clahe_image = clahe_equalized(corrected_image)
    enhanced_image = cv2.merge([image[:, :, 0], clahe_image, image[:, :, 2]])

    return image, enhanced_image


def get_preprocess_fn(
    method: Optional[int],
) -> Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]]:

    """
    Get the preprocessing function from the method number in the config.

    Args:
        method: 0 to 7, or None for no preprocessing. 6 is preprocess_retinal_image.

    Returns:
        Preprocessing function, or None.
    """

    if method is None:
        return None

    name = "preprocess_retinal_image" if method == 6 else f"preprocess_image_{method}"
    if type(method) is not int or name not in globals():
        raise ValueError(
            f"Unknown preprocessing method '{method}'. "
            f"Use a number from 0 to 7, or null, in 'preprocessing.method'."
        )

    return globals()[name]
