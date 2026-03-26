from typing import Tuple

import numpy as np

DEFAULT_IMAGE_SIZE = 512

MODEL_FAMILY_RESIZE_SETTINGS = {
    "da3": {"max_dimension": 518, "divisor": 14},
    "must3r": {"max_dimension": 512, "divisor": 16},
}

EVAL_OUTPUT_RESOLUTIONS = {
    ("kitti", "da3"): (518, 168),
    ("kitti", "must3r"): (512, 160),
    ("nuscenes", "da3"): (518, 294),
    ("nuscenes", "must3r"): (512, 288),
}


def normalize_model_family(model_family: str) -> str:
    normalized = model_family.lower()
    if "da3" in normalized:
        return "da3"
    if "must3r" in normalized:
        return "must3r"
    raise ValueError(f"Unsupported model family '{model_family}'")


def round_to_nearest_divisible(value: float, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    return max(divisor, int(np.floor((value / divisor) + 0.5)) * divisor)


def get_output_resolution(image_size: Tuple[int, int], model_family: str) -> Tuple[int, int]:
    normalized_model_family = normalize_model_family(model_family)
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image_size}")

    resize_settings = MODEL_FAMILY_RESIZE_SETTINGS[normalized_model_family]
    max_dimension = resize_settings["max_dimension"]
    divisor = resize_settings["divisor"]

    if width >= height:
        scaled_short_side = height * max_dimension / width
        return max_dimension, round_to_nearest_divisible(scaled_short_side, divisor)

    scaled_short_side = width * max_dimension / height
    return round_to_nearest_divisible(scaled_short_side, divisor), max_dimension


def get_eval_output_resolution(dataset: str, model_family: str) -> Tuple[int, int]:
    normalized_model_family = normalize_model_family(model_family)
    resolution = EVAL_OUTPUT_RESOLUTIONS.get((dataset, normalized_model_family))
    if resolution is None:
        raise ValueError(
            f"Unsupported dataset/model_family combination: dataset={dataset}, "
            f"model_family={normalized_model_family}"
        )
    return resolution
