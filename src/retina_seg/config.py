"""
Config loading for retinal fundus image segmentation.
Responsibilities:
    - Load the default config and the dataset config named in it
    - Merge both into one config dictionary
"""

import os
from typing import Any, Dict
import yaml


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:

    """
    Merge one config into another, section by section.

    Args:
        base: config to merge into.
        override: config whose values win.

    Returns:
        Merged config.
    """

    merged = dict(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:

    """
    Load the default config and the dataset config it names.

    Args:
        config_path: path of the default config.

    Returns:
        Config with the dataset values merged in.
    """

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    dataset = config["data"]["dataset"]
    dataset_path = os.path.join(os.path.dirname(config_path), "datasets", f"{dataset}.yaml")

    if not os.path.exists(dataset_path):
        available = sorted(f[:-5] for f in os.listdir(os.path.dirname(dataset_path)) if f.endswith(".yaml"))
        raise FileNotFoundError(
            f"No config for dataset '{dataset}' at {dataset_path}. "
            f"Available: {available}. Check the 'data.dataset' value in your config."
        )

    with open(dataset_path) as dataset_file:
        dataset_config = yaml.safe_load(dataset_file)

    return merge_configs(config, dataset_config)
