"""Shared AVA-VLA inference preprocessing contracts."""

from __future__ import annotations

import math
from typing import Union

import numpy as np
from PIL import Image


def center_crop_for_augmentation(
    image: Union[np.ndarray, Image.Image],
    area_scale: float = 0.9,
) -> Image.Image:
    """Apply the deterministic center counterpart of training's 90% random crop.

    The crop is resized back to the original image size; the vision backbone's
    own image transform remains responsible for the final 384px resize.
    """
    if not 0.0 < area_scale <= 1.0:
        raise ValueError(f"area_scale must be in (0, 1], got {area_scale}")
    pil_image = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
    width, height = pil_image.size
    side_scale = math.sqrt(area_scale)
    crop_width = max(1, int(round(width * side_scale)))
    crop_height = max(1, int(round(height * side_scale)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)
