"""
Tests for the data pipeline: preprocessing, augmentation, patches, dataset and config.
Responsibilities:
    - Read real images and masks from the dataset named in configs/default.yaml
    - Check the intensity helpers and every preprocessing method
    - Check the augmentations keep shapes, follow the seed and save the right files
    - Check patches cover the image, rebuild it, and come back one-hot encoded
    - Check prepare_dataset builds train and test patches from the config
"""

import copy
import os
import random
import shutil
import cv2
import imageio
import numpy as np
import pytest
import yaml
from retina_seg.config import load_config, merge_configs
from retina_seg.data import augmentation, dataset, patches, preprocessing


def read_image(image_path: str, size: int) -> np.ndarray:

    """
    Read a fundus image and resize it.

    Args:
        image_path: path of the image.
        size: output size.

    Returns:
        BGR image.
    """

    return cv2.resize(cv2.imread(image_path), (size, size))


def read_mask(mask_path: str, size: int) -> np.ndarray:

    """
    Read a vessel mask and resize it, keeping one channel.

    Args:
        mask_path: path of the mask.
        size: output size.

    Returns:
        Single channel mask.
    """

    mask = imageio.mimread(mask_path)[0]
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def make_save_path(tmp_path) -> str:

    """
    Create an output folder with the images and masks subfolders.

    Args:
        tmp_path: folder given by pytest.

    Returns:
        Path of the output folder.
    """

    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()

    return str(tmp_path)


def link_pairs(img_paths: list, mask_paths: list, split_dir) -> None:

    """
    Link image and mask files into a split folder, so tests reuse the real files.

    Args:
        img_paths: image paths to link.
        mask_paths: mask paths to link.
        split_dir: folder to create images/ and masks/ in.
    """

    (split_dir / "images").mkdir(parents=True)
    (split_dir / "masks").mkdir(parents=True)

    for image_path, mask_path in zip(img_paths, mask_paths):
        os.symlink(os.path.abspath(image_path), split_dir / "images" / os.path.basename(image_path))
        os.symlink(os.path.abspath(mask_path), split_dir / "masks" / os.path.basename(mask_path))


@pytest.fixture
def config() -> dict:

    """
    The config for the dataset named in configs/default.yaml.
    """

    return copy.deepcopy(load_config())


@pytest.fixture
def resize(config: dict) -> int:

    """
    The resize value for this dataset.
    """

    return config["preprocessing"]["resize"]


@pytest.fixture
def patch_size(config: dict) -> int:

    """
    The patch size for this dataset.
    """

    return config["preprocessing"]["patch_size"]


@pytest.fixture
def dataset_root(config: dict) -> str:

    """
    The dataset folder, skipping the test if it is not there.
    """

    root = config["data"]["root"]
    if not os.path.isdir(root):
        pytest.skip(f"dataset folder '{root}' not found, check data.dataset in configs/default.yaml")

    return root


@pytest.fixture
def training_paths(dataset_root: str) -> tuple:

    """
    Image and mask paths of the training split.
    """

    return dataset.list_image_mask_paths(dataset_root, "training")


@pytest.fixture
def fundus_image(training_paths: tuple, resize: int) -> np.ndarray:

    """
    The first training image, resized as the pipeline resizes it.
    """

    return read_image(training_paths[0][0], resize)


@pytest.fixture
def vessel_mask(training_paths: tuple, resize: int) -> np.ndarray:

    """
    The mask of the first training image, resized to match.
    """

    return read_mask(training_paths[1][0], resize)


@pytest.fixture
def image_patches(fundus_image: np.ndarray, patch_size: int) -> list:

    """
    Patches of the first training image.
    """

    return patches.sliding_window(fundus_image, patch_size, (patch_size, patch_size))


@pytest.fixture
def mask_patches(vessel_mask: np.ndarray, patch_size: int) -> list:

    """
    Patches of the first training mask.
    """

    return patches.sliding_window(vessel_mask, patch_size, (patch_size, patch_size))


@pytest.fixture
def subset_config(config: dict, dataset_root: str, tmp_path) -> dict:

    """
    A config pointed at 2 training and 1 test pair, linked from the real dataset.
    Keeps the pipeline tests quick without copying whole datasets.
    """

    train_images, train_masks, test_images, test_masks = dataset.prepare_data_paths(dataset_root)
    link_pairs(train_images[:2], train_masks[:2], tmp_path / "training")
    link_pairs(test_images[:1], test_masks[:1], tmp_path / "test")

    config["data"]["root"] = str(tmp_path)

    return config


def test_clahe_equalized_keeps_shape_and_type(fundus_image):

    """
    CLAHE returns a single channel image of the same size.
    """

    green = fundus_image[:, :, 1]
    equalized = preprocessing.clahe_equalized(green)

    assert equalized.shape == green.shape
    assert equalized.dtype == np.uint8


def test_clahe_equalized_raises_on_colour_image(fundus_image):

    """
    CLAHE only accepts one channel.
    """

    with pytest.raises(cv2.error):
        preprocessing.clahe_equalized(fundus_image)


def test_gamma_correction_brightens_below_one(fundus_image):

    """
    A gamma below 1 makes the image brighter.
    """

    brighter = preprocessing.gamma_correction(fundus_image, 0.5)

    assert brighter.mean() > fundus_image.mean()
    assert brighter.dtype == np.uint8


def test_gamma_correction_1_darkens_below_one(fundus_image):

    """
    The lookup table version uses the inverse convention, so it darkens.
    """

    darker = preprocessing.gamma_correction_1(fundus_image, 0.5)

    assert darker.mean() < fundus_image.mean()


def test_normalize_image_spans_full_range(fundus_image):

    """
    Normalizing stretches the values to 0 and 255.
    """

    normalized = preprocessing.normalize_image(fundus_image[:, :, 1])

    assert normalized.min() == 0
    assert normalized.max() == 255


def test_apply_morphological_operations_keeps_shape(fundus_image):

    """
    The morphological filter returns one channel of the same size.
    """

    green = fundus_image[:, :, 1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    filtered = preprocessing.apply_morphological_operations(green, kernel)

    assert filtered.shape == green.shape


def test_apply_n4_bias_field_correction_keeps_shape(fundus_image):

    """
    N4 correction returns one channel of the same size.
    """

    green = fundus_image[:, :, 1]
    corrected = preprocessing.apply_n4_bias_field_correction(green)

    assert corrected.shape == green.shape
    assert corrected.dtype == np.uint8


@pytest.mark.parametrize("method", [0, 1, 2, 3, 4, 5, 6, 7])
def test_method_enhances_the_image(fundus_image, method):

    """
    Every method returns the untouched image plus an enhanced one that really changed.
    Methods 0 and 3 give one channel, the rest give three.
    """

    original, enhanced = preprocessing.get_preprocess_fn(method)(fundus_image)
    expected_dimensions = 2 if method in (0, 3) else 3

    assert np.array_equal(original, fundus_image)
    assert enhanced.shape[:2] == fundus_image.shape[:2]
    assert enhanced.dtype == np.uint8
    assert enhanced.ndim == expected_dimensions
    assert not np.array_equal(enhanced, fundus_image[:, :, 1] if enhanced.ndim == 2 else fundus_image)


@pytest.mark.parametrize("method, name", [
    (0, "preprocess_image_0"),
    (1, "preprocess_image_1"),
    (2, "preprocess_image_2"),
    (3, "preprocess_image_3"),
    (4, "preprocess_image_4"),
    (5, "preprocess_image_5"),
    (6, "preprocess_retinal_image"),
    (7, "preprocess_image_7"),
])
def test_get_preprocess_fn_maps_number_to_function(method, name):

    """
    The number in the config picks the matching function, and 6 is the retinal one.
    """

    assert preprocessing.get_preprocess_fn(method).__name__ == name


def test_get_preprocess_fn_returns_none_for_none():

    """
    No method means no preprocessing.
    """

    assert preprocessing.get_preprocess_fn(None) is None


@pytest.mark.parametrize("method", [8, -1, "5", 5.0, True])
def test_get_preprocess_fn_rejects_bad_values(method):

    """
    Anything that is not a valid method number fails with a clear message.
    """

    with pytest.raises(ValueError, match="Unknown preprocessing method"):
        preprocessing.get_preprocess_fn(method)


def test_random_hsv_gamma_adjustment_keeps_shape(fundus_image):

    """
    The colour jitter returns an image of the same size and type.
    """

    adjusted = augmentation.random_hsv_gamma_adjustment(fundus_image)

    assert adjusted.shape == fundus_image.shape
    assert adjusted.dtype == np.uint8


def test_random_hsv_gamma_adjustment_follows_the_seed(fundus_image):

    """
    The same seed gives the same result.
    """

    random.seed(1)
    first = augmentation.random_hsv_gamma_adjustment(fundus_image)
    random.seed(1)
    second = augmentation.random_hsv_gamma_adjustment(fundus_image)

    assert np.array_equal(first, second)


def test_random_augment_image_mask_keeps_shapes(fundus_image, vessel_mask):

    """
    Image and mask come back with their own shapes unchanged.
    """

    random.seed(2)
    np.random.seed(2)
    image, mask = augmentation.random_augment_image_mask(fundus_image, vessel_mask)

    assert image.shape == fundus_image.shape
    assert mask.shape == vessel_mask.shape


def test_random_augment_image_mask_keeps_mask_values(fundus_image, vessel_mask):

    """
    Augmenting the mask does not invent new class values.
    """

    random.seed(3)
    np.random.seed(3)
    _, mask = augmentation.random_augment_image_mask(fundus_image, vessel_mask)

    assert set(np.unique(mask)).issubset(set(np.unique(vessel_mask)) | {0})


def test_save_five_augmentations_writes_five_pairs(tmp_path, fundus_image, vessel_mask, training_paths):

    """
    Five versions are written to images/ and masks/ with matching names.
    """

    save_path = make_save_path(tmp_path)
    source = training_paths[0][0]
    name = os.path.basename(source).split(".")[0]
    augmentation.save_five_augmentations(fundus_image, vessel_mask, save_path, source)

    images = sorted(os.listdir(os.path.join(save_path, "images")))
    masks = sorted(os.listdir(os.path.join(save_path, "masks")))

    assert images == [f"{name}_{index}.jpg" for index in range(5)]
    assert images == masks


def test_save_five_augmentations_writes_one_pair_without_augment(tmp_path, fundus_image, vessel_mask, training_paths):

    """
    With augment off, only the original pair is written, named after the source.
    """

    save_path = make_save_path(tmp_path)
    source = training_paths[0][0]
    name = os.path.basename(source).split(".")[0]
    augmentation.save_five_augmentations(fundus_image, vessel_mask, save_path, source, augment=False)

    assert os.listdir(os.path.join(save_path, "images")) == [f"{name}.jpg"]


def test_save_seven_augmentations_writes_seven_pairs(tmp_path, fundus_image, vessel_mask, training_paths):

    """
    Seven versions are written and returned.
    """

    save_path = make_save_path(tmp_path)
    images, masks = augmentation.save_seven_augmentations(
        fundus_image, vessel_mask, save_path, training_paths[0][0]
    )

    assert len(images) == 7
    assert len(masks) == 7
    assert len(os.listdir(os.path.join(save_path, "images"))) == 7


def test_add_augmented_patches_appends_to_the_lists(image_patches, mask_patches):

    """
    The lists grow by the number asked for, keeping the first patches unchanged.
    """

    first = image_patches[0].copy()
    grown_images, grown_masks = augmentation.add_augmented_patches(image_patches, mask_patches, 3, seed=42)

    assert len(grown_images) == len(grown_masks)
    assert len(grown_images) == len(mask_patches)
    assert np.array_equal(grown_images[0], first)


def test_add_augmented_patches_keeps_patch_shape(image_patches, mask_patches):

    """
    Augmented patches have the same shape as the ones they came from.
    """

    image_shape = image_patches[0].shape
    mask_shape = mask_patches[0].shape
    grown_images, grown_masks = augmentation.add_augmented_patches(image_patches, mask_patches, 2, seed=42)

    assert grown_images[-1].shape == image_shape
    assert grown_masks[-1].shape == mask_shape


def test_add_augmented_patches_follows_the_seed(image_patches, mask_patches):

    """
    The same seed gives the same augmented patches.
    """

    first_images = [patch.copy() for patch in image_patches]
    first_masks = [patch.copy() for patch in mask_patches]
    second_images = [patch.copy() for patch in image_patches]
    second_masks = [patch.copy() for patch in mask_patches]

    augmentation.add_augmented_patches(first_images, first_masks, 3, seed=7)
    augmentation.add_augmented_patches(second_images, second_masks, 3, seed=7)

    assert all(np.array_equal(a, b) for a, b in zip(first_images, second_images))
    assert all(np.array_equal(a, b) for a, b in zip(first_masks, second_masks))


def test_add_augmented_patches_rejects_mismatched_lists(image_patches, mask_patches):

    """
    Images and masks must be aligned.
    """

    with pytest.raises(AssertionError):
        augmentation.add_augmented_patches(image_patches, mask_patches[:1], 1)


def test_sliding_window_covers_the_image(fundus_image, patch_size, resize):

    """
    The whole image is covered by patches of the configured size.
    """

    result = patches.sliding_window(fundus_image, patch_size, (patch_size, patch_size))
    expected = (resize // patch_size) ** 2

    assert len(result) == expected
    assert result[0].shape == (patch_size, patch_size, 3)


def test_sliding_window_scales_the_pixels(fundus_image, patch_size):

    """
    Patches are divided by the scale, so the default puts them in the 0-1 range.
    """

    scaled = patches.sliding_window(fundus_image, patch_size, (patch_size, patch_size))
    unscaled = patches.sliding_window(fundus_image, patch_size, (patch_size, patch_size), scale=1.0)

    assert scaled[0].max() <= 1.0
    assert unscaled[0].max() > 1.0
    assert np.allclose(scaled[0] * 255.0, unscaled[0])


def test_sliding_window_overlaps_with_a_smaller_step(fundus_image, patch_size, resize):

    """
    Halving the step gives more, overlapping patches.
    """

    result = patches.sliding_window(fundus_image, patch_size, (patch_size // 2, patch_size // 2))
    expected = ((resize - patch_size) // (patch_size // 2) + 1) ** 2

    assert len(result) == expected


def test_sliding_window_drops_the_remainder(fundus_image, patch_size):

    """
    A size that is not a multiple of the patch size loses the leftover strip.
    """

    cropped = fundus_image[: patch_size + 10, : patch_size + 10]
    result = patches.sliding_window(cropped, patch_size, (patch_size, patch_size))

    assert len(result) == 1


def test_random_crop_returns_aligned_patches(fundus_image, vessel_mask, patch_size):

    """
    Image, mask and FOV are cropped at the same positions.
    """

    images, masks, fovs = patches.random_crop_image_mask_pairs(
        fundus_image, vessel_mask, vessel_mask, patch_size, 5, seed=42
    )

    assert len(images) == len(masks) == len(fovs) == 5
    assert images[0].shape == (patch_size, patch_size, 3)
    assert all(np.array_equal(mask, fov) for mask, fov in zip(masks, fovs))


def test_random_crop_follows_the_seed(fundus_image, vessel_mask, patch_size):

    """
    The same seed crops the same positions.
    """

    first = patches.random_crop_image_mask_pairs(fundus_image, vessel_mask, vessel_mask, patch_size, 5, seed=1)
    second = patches.random_crop_image_mask_pairs(fundus_image, vessel_mask, vessel_mask, patch_size, 5, seed=1)

    assert all(np.array_equal(a, b) for a, b in zip(first[0], second[0]))


def test_reconstruct_image_rebuilds_the_original(fundus_image, image_patches, patch_size, resize):

    """
    Cutting an image into patches and rebuilding it gives the image back.
    """

    rebuilt = patches.reconstruct_image(image_patches, (resize, resize), patch_size, (patch_size, patch_size), 3, 0)

    assert np.allclose(rebuilt, fundus_image / 255.0)


def test_reconstruct_image_starts_at_patch_index(training_paths, image_patches, patch_size, resize):

    """
    With two images in one list, patch_index picks which one is rebuilt.
    """

    second_image = read_image(training_paths[0][1], resize)
    both = image_patches + patches.sliding_window(second_image, patch_size, (patch_size, patch_size))
    rebuilt = patches.reconstruct_image(
        both, (resize, resize), patch_size, (patch_size, patch_size), 3, len(image_patches)
    )

    assert np.allclose(rebuilt, second_image / 255.0)


def test_create_patches_returns_images_and_one_hot_masks(training_paths, patch_size, resize, config):

    """
    Two images give twice the patches of one, with masks one-hot encoded.
    """

    img_paths, mask_paths = training_paths
    images, masks = patches.create_patches(img_paths[:2], mask_paths[:2], patch_size, resize)
    expected = 2 * (resize // patch_size) ** 2

    assert images.shape == (expected, patch_size, patch_size, 3)
    assert masks.shape == (expected, patch_size, patch_size, config["data"]["num_classes"])
    assert np.all(masks.sum(axis=-1) == 1)
    assert images.max() <= 1.0


@pytest.mark.parametrize("method", [0, 1, 3, 5])
def test_create_patches_always_gives_three_channels(training_paths, patch_size, resize, method):

    """
    Single channel methods are stacked, so every method feeds the same input shape.
    """

    img_paths, mask_paths = training_paths
    images, _ = patches.create_patches(img_paths[:1], mask_paths[:1], patch_size, resize, method=method)
    expected = (resize // patch_size) ** 2

    assert images.shape == (expected, patch_size, patch_size, 3)


def test_create_patches_adds_augmented_patches(training_paths, patch_size, resize):

    """
    Asking for augmented patches makes the set bigger.
    """

    img_paths, mask_paths = training_paths
    plain, _ = patches.create_patches(img_paths[:1], mask_paths[:1], patch_size, resize)
    augmented, augmented_masks = patches.create_patches(
        img_paths[:1], mask_paths[:1], patch_size, resize, num_augmented=3
    )

    assert augmented.shape[0] == plain.shape[0] + 3
    assert augmented_masks.shape[0] == plain.shape[0] + 3


def test_list_image_mask_paths_returns_aligned_paths(training_paths):

    """
    Images and masks come back sorted, in the same order and the same number.
    """

    img_paths, mask_paths = training_paths

    assert len(img_paths) == len(mask_paths)
    assert img_paths == sorted(img_paths)
    assert [os.path.basename(p).split(".")[0] for p in img_paths] == [
        os.path.basename(p).split(".")[0] for p in mask_paths
    ]


def test_list_image_mask_paths_raises_for_missing_split(dataset_root):

    """
    A split that does not exist says so instead of returning nothing.
    """

    with pytest.raises(FileNotFoundError, match="No images found"):
        dataset.list_image_mask_paths(dataset_root, "validation")


def test_list_image_mask_paths_raises_on_count_mismatch(training_paths, tmp_path):

    """
    A missing mask is caught before training starts.
    """

    img_paths, mask_paths = training_paths
    link_pairs(img_paths[:2], mask_paths[:2], tmp_path / "training")
    next((tmp_path / "training" / "masks").iterdir()).unlink()

    with pytest.raises(ValueError, match="Mismatched image/mask counts"):
        dataset.list_image_mask_paths(str(tmp_path), "training")


def test_prepare_data_paths_returns_both_splits(dataset_root):

    """
    Train and test paths come back in one call, each split aligned.
    """

    train_images, train_masks, test_images, test_masks = dataset.prepare_data_paths(dataset_root)

    assert len(train_images) == len(train_masks) > 0
    assert len(test_images) == len(test_masks) > 0


def test_load_colormap_returns_one_row_per_class(dataset_root, config):

    """
    The colormap is read from the .mat file named in the config.
    """

    colormap_path = os.path.join(dataset_root, config["data"]["colormap"])
    if not os.path.exists(colormap_path):
        pytest.skip(f"no colormap at '{colormap_path}'")

    colormap = dataset.load_colormap(colormap_path)

    assert colormap.shape == (config["data"]["num_classes"], 3)


def test_prepare_dataset_returns_train_and_test_patches(subset_config):

    """
    Two training images and one test image give patches of the configured size.
    """

    patch_size = subset_config["preprocessing"]["patch_size"]
    per_image = (subset_config["preprocessing"]["resize"] // patch_size) ** 2
    classes = subset_config["data"]["num_classes"]
    train_images, train_masks, test_images, test_masks = dataset.prepare_dataset(subset_config)

    assert train_images.shape == (2 * per_image, patch_size, patch_size, 3)
    assert train_masks.shape == (2 * per_image, patch_size, patch_size, classes)
    assert test_images.shape == (per_image, patch_size, patch_size, 3)
    assert test_masks.shape == (per_image, patch_size, patch_size, classes)


def test_prepare_dataset_masks_are_one_hot(subset_config):

    """
    Every mask pixel belongs to exactly one class.
    """

    _, train_masks, _, test_masks = dataset.prepare_dataset(subset_config)

    assert np.all(train_masks.sum(axis=-1) == 1)
    assert np.all(test_masks.sum(axis=-1) == 1)


def test_prepare_dataset_augments_only_the_training_split(subset_config):

    """
    Augmented patches are added to training and never to test.
    """

    plain_train, _, plain_test, _ = dataset.prepare_dataset(subset_config)
    subset_config["augmentation"]["num_augmented"] = 5
    augmented_train, _, augmented_test, _ = dataset.prepare_dataset(subset_config)

    assert augmented_train.shape[0] == plain_train.shape[0] + 5
    assert augmented_test.shape[0] == plain_test.shape[0]


def test_prepare_dataset_applies_the_method_from_the_config(subset_config):

    """
    Setting a method changes the patches, and the shapes stay the same.
    """

    plain_train, _, _, _ = dataset.prepare_dataset(subset_config)
    subset_config["preprocessing"]["method"] = 5
    processed_train, _, _, _ = dataset.prepare_dataset(subset_config)

    assert processed_train.shape == plain_train.shape
    assert not np.array_equal(processed_train, plain_train)


def test_prepare_dataset_follows_the_resize_value(subset_config):

    """
    A resize of one patch gives one patch per image.
    """

    patch_size = subset_config["preprocessing"]["patch_size"]
    subset_config["preprocessing"]["resize"] = patch_size
    train_images, _, _, _ = dataset.prepare_dataset(subset_config)

    assert train_images.shape == (2, patch_size, patch_size, 3)


def test_tf_dataset_batches_the_patches(subset_config):

    """
    The tf.data wrapper yields batches of the size asked for.
    """

    patch_size = subset_config["preprocessing"]["patch_size"]
    train_images, train_masks, _, _ = dataset.prepare_dataset(subset_config)
    batched = dataset.tf_dataset(train_images, train_masks, batch_size=4)
    images, masks = next(iter(batched))

    assert tuple(images.shape) == (4, patch_size, patch_size, 3)
    assert len(list(batched)) == train_images.shape[0] // 4


def test_load_config_merges_the_dataset_file(config):

    """
    The dataset named in default.yaml brings in its own root and resize.
    """

    assert config["data"]["root"]
    assert config["preprocessing"]["resize"]
    assert config["preprocessing"]["patch_size"]


def test_load_config_raises_for_unknown_dataset(tmp_path):

    """
    A dataset with no config file fails and lists the ones that exist.
    """

    shutil.copytree("configs", tmp_path / "configs")
    default_path = tmp_path / "configs" / "default.yaml"
    default_config = yaml.safe_load(default_path.read_text())
    default_config["data"]["dataset"] = "not_a_dataset"
    default_path.write_text(yaml.safe_dump(default_config))

    with pytest.raises(FileNotFoundError, match="No config for dataset"):
        load_config(str(default_path))


def test_merge_configs_keeps_untouched_sections():

    """
    Merging only replaces the keys given, section by section.
    """

    merged = merge_configs(
        {"data": {"root": "a", "batch_size": 1}, "augmentation": {"num_augmented": 0}},
        {"data": {"root": "b"}},
    )

    assert merged == {"data": {"root": "b", "batch_size": 1}, "augmentation": {"num_augmented": 0}}
