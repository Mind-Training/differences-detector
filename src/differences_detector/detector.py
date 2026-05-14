"""Detect gray circular difference markers in images."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np

ImageInput: TypeAlias = str | PathLike[str] | np.ndarray


def detect_difference_points(
    image: ImageInput,
    *,
    gray_min: int = 120,
    gray_max: int = 215,
    area_min: int = 500,
    area_max: int = 50000,
) -> list[tuple[int, int]]:
    """Return the center coordinates of gray circular markers in an image.

    Parameters
    ----------
    image:
        Path to an image file or an image already loaded as a NumPy array.
    gray_min:
        Lower bound of the gray pixel range, from 0 to 255.
    gray_max:
        Upper bound of the gray pixel range, from 0 to 255.
    area_min:
        Minimum contour area to consider.
    area_max:
        Maximum contour area to consider.

    Returns
    -------
    list[tuple[int, int]]
        Center coordinates as ``(x, y)`` tuples, sorted top-to-bottom and then
        left-to-right.
    """
    img = _load_image(image)
    gray = _to_grayscale(img)

    mask = ((gray > gray_min) & (gray < gray_max)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points: list[tuple[int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area_min < area < area_max:
            x, y, w, h = cv2.boundingRect(contour)
            points.append((x + w // 2, y + h // 2))

    points.sort(key=lambda point: (point[1], point[0]))
    return points


def detect_gray_circles(image: ImageInput, **kwargs: object) -> list[tuple[int, int]]:
    """Backward-compatible alias for ``detect_difference_points``."""
    return detect_difference_points(image, **kwargs)


def draw_detected_points(
    image: ImageInput,
    points: list[tuple[int, int]],
    output_path: str | PathLike[str],
    *,
    radius: int = 45,
    thickness: int = 4,
    color: tuple[int, int, int] = (0, 0, 255),
) -> None:
    """Draw detected points as circles and save the resulting image."""
    img = _load_image(image).copy()
    for x, y in points:
        cv2.circle(img, (x, y), radius, color, thickness)

    if not cv2.imwrite(str(output_path), img):
        raise OSError(f"Could not write output image: {output_path}")


def _load_image(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Image array is empty.")
        return image

    image_path = Path(image)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(
        "Expected a grayscale, BGR, or BGRA image array; "
        f"received shape {image.shape}."
    )
