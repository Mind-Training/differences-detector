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
    radius_min: int = 30,
    radius_max: int = 65,
    ring_width: int = 10,
    score_threshold: float = 0.65,
    min_distance: int | None = None,
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
        Minimum ring area to consider.
    area_max:
        Maximum ring area to consider.
    radius_min:
        Minimum marker radius to consider.
    radius_max:
        Maximum marker radius to consider.
    ring_width:
        Expected width of the gray ring.
    score_threshold:
        Minimum normalized ring score, from 0.0 to 1.0.
    min_distance:
        Minimum pixel distance between detected centers. Defaults to
        ``radius_min``.

    Returns
    -------
    list[tuple[int, int]]
        Center coordinates as ``(x, y)`` tuples, sorted top-to-bottom and then
        left-to-right.
    """
    _validate_detection_params(
        gray_min=gray_min,
        gray_max=gray_max,
        area_min=area_min,
        area_max=area_max,
        radius_min=radius_min,
        radius_max=radius_max,
        ring_width=ring_width,
        score_threshold=score_threshold,
        min_distance=min_distance,
    )

    img = _load_image(image)
    gray = _to_grayscale(img)

    mask = ((gray > gray_min) & (gray < gray_max)).astype(np.float32)
    if not np.any(mask):
        return []

    if min_distance is None:
        min_distance = radius_min

    candidates = _find_ring_candidates(
        mask,
        area_min=area_min,
        area_max=area_max,
        radius_min=radius_min,
        radius_max=radius_max,
        ring_width=ring_width,
        score_threshold=score_threshold,
        min_distance=min_distance,
    )

    points = _suppress_nearby_candidates(candidates, min_distance=min_distance)
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


def _find_ring_candidates(
    mask: np.ndarray,
    *,
    area_min: int,
    area_max: int,
    radius_min: int,
    radius_max: int,
    ring_width: int,
    score_threshold: float,
    min_distance: int,
) -> list[tuple[float, int, int]]:
    candidates: list[tuple[float, int, int]] = []
    local_kernel_size = _odd_kernel_size(min_distance)

    for radius in range(radius_min, radius_max + 1, 2):
        kernel = _annulus_kernel(radius, ring_width)
        ring_area = float(kernel.sum())
        if not area_min < ring_area < area_max:
            continue

        response = cv2.filter2D(
            mask,
            ddepth=-1,
            kernel=kernel / ring_area,
            borderType=cv2.BORDER_CONSTANT,
        )
        local_max = cv2.dilate(
            response,
            np.ones((local_kernel_size, local_kernel_size), dtype=np.uint8),
        )
        ys, xs = np.where((response == local_max) & (response >= score_threshold))
        candidates.extend(
            (float(response[y, x]), int(x), int(y)) for x, y in zip(xs, ys)
        )

    return candidates


def _suppress_nearby_candidates(
    candidates: list[tuple[float, int, int]],
    *,
    min_distance: int,
) -> list[tuple[int, int]]:
    selected: list[tuple[float, int, int]] = []
    min_distance_squared = min_distance * min_distance

    for score, x, y in sorted(candidates, reverse=True):
        if all(
            (x - selected_x) ** 2 + (y - selected_y) ** 2 > min_distance_squared
            for _, selected_x, selected_y in selected
        ):
            selected.append((score, x, y))

    return [(x, y) for _, x, y in selected]


def _annulus_kernel(radius: int, ring_width: int) -> np.ndarray:
    inner_radius = max(1, radius - ring_width)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    distance_squared = xx * xx + yy * yy

    return (
        (distance_squared <= radius * radius)
        & (distance_squared >= inner_radius * inner_radius)
    ).astype(np.float32)


def _odd_kernel_size(size: int) -> int:
    return max(3, size if size % 2 else size + 1)


def _validate_detection_params(
    *,
    gray_min: int,
    gray_max: int,
    area_min: int,
    area_max: int,
    radius_min: int,
    radius_max: int,
    ring_width: int,
    score_threshold: float,
    min_distance: int | None,
) -> None:
    if not 0 <= gray_min < gray_max <= 255:
        raise ValueError("Expected 0 <= gray_min < gray_max <= 255.")
    if area_min < 0 or area_max <= area_min:
        raise ValueError("Expected 0 <= area_min < area_max.")
    if radius_min <= 0 or radius_max < radius_min:
        raise ValueError("Expected 0 < radius_min <= radius_max.")
    if ring_width <= 0 or ring_width >= radius_max:
        raise ValueError("Expected 0 < ring_width < radius_max.")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("Expected 0.0 <= score_threshold <= 1.0.")
    if min_distance is not None and min_distance <= 0:
        raise ValueError("Expected min_distance to be positive.")
