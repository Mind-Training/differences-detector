"""Detect gray translucent difference markers in images."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np

ImageInput: TypeAlias = str | PathLike[str] | np.ndarray


@dataclass(frozen=True)
class _Component:
    score: float
    x: int
    y: int
    w: int
    h: int
    area: int

    # Pixel centroid from connected components.
    centroid_x: float
    centroid_y: float

    # Geometric center from the component contour.
    # This is more stable for broken/uneven translucent circles.
    circle_x: float
    circle_y: float


def detect_difference_points(
    image: ImageInput,
    *,
    gray_min: int = 85,
    gray_max: int = 170,
    area_min: int | None = None,
    area_max: int | None = None,
    radius_min: int | None = None,
    radius_max: int | None = None,
    ring_width: int | None = None,
    score_threshold: float = 0.0,
    min_distance: int | None = None,
    expected_count: int = 8,
) -> list[tuple[int, int]]:
    """Return the center coordinates of gray translucent difference markers."""
    img = _load_image(image)
    gray = _to_grayscale(img)

    height, width = gray.shape[:2]
    scale = min(width, height) / 1063.0

    if radius_min is None:
        radius_min = max(10, round(30 * scale))

    if radius_max is None:
        radius_max = max(radius_min + 4, round(65 * scale))

    if min_distance is None:
        min_distance = max(14, round(radius_min * 1.25))

    if area_min is None:
        area_min = max(25, round(radius_min * radius_min * 0.30))

    if area_max is None:
        area_max = max(area_min + 1, round(radius_max * radius_max * 3.5))

    mask = _build_marker_mask(gray, gray_min=gray_min, gray_max=gray_max)

    components = _extract_components(
        mask,
        area_min=area_min,
        area_max=area_max,
        radius_min=radius_min,
        radius_max=radius_max,
    )

    return _select_points(
        components,
        expected_count=expected_count,
        min_distance=min_distance,
    )


def detect_gray_circles(image: ImageInput, **kwargs: object) -> list[tuple[int, int]]:
    """Backward-compatible alias for ``detect_difference_points``."""
    return detect_difference_points(image, **kwargs)


def draw_detected_points(
    image: ImageInput,
    points: list[tuple[int, int]],
    output_path: str | PathLike[str],
    *,
    radius: int | None = None,
    thickness: int | None = None,
    color: tuple[int, int, int] = (0, 0, 255),
) -> None:
    """Draw detected points as circles and save the resulting image."""
    img = _load_image(image).copy()

    height, width = img.shape[:2]
    scale = min(width, height) / 1063.0

    if radius is None:
        radius = max(10, round(42 * scale))

    if thickness is None:
        thickness = max(2, round(4 * scale))

    for x, y in points:
        cv2.circle(img, (x, y), radius, color, thickness)

    if not cv2.imwrite(str(output_path), img):
        raise OSError(f"Could not write output image: {output_path}")


def _build_marker_mask(
    gray: np.ndarray,
    *,
    gray_min: int,
    gray_max: int,
) -> np.ndarray:
    """Build a binary mask for mid-gray translucent markers."""
    mask = ((gray >= gray_min) & (gray <= gray_max)).astype(np.uint8) * 255

    # Removes thin drawing/grid lines while preserving marker blobs.
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def _extract_components(
    mask: np.ndarray,
    *,
    area_min: int,
    area_max: int,
    radius_min: int,
    radius_max: int,
) -> list[_Component]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    components: list[_Component] = []

    min_size = max(6, round(radius_min * 0.70))
    max_size = max(min_size + 1, round(radius_max * 2.6))

    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        centroid_x, centroid_y = centroids[i]

        if not area_min <= area <= area_max:
            continue

        if w < min_size or h < min_size:
            continue

        if w > max_size or h > max_size:
            continue

        aspect = max(w, h) / max(1, min(w, h))
        if aspect > 2.35:
            continue

        fill_ratio = area / max(1, w * h)
        if not 0.10 <= fill_ratio <= 0.90:
            continue

        component_mask = (labels == i).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        (circle_x, circle_y), _ = cv2.minEnclosingCircle(contour)

        compactness_score = 1.0 / aspect
        score = float(area) * compactness_score

        components.append(
            _Component(
                score=score,
                x=x,
                y=y,
                w=w,
                h=h,
                area=area,
                centroid_x=float(centroid_x),
                centroid_y=float(centroid_y),
                circle_x=float(circle_x),
                circle_y=float(circle_y),
            )
        )

    return components


def _select_points(
    components: list[_Component],
    *,
    expected_count: int,
    min_distance: int,
) -> list[tuple[int, int]]:
    """Select the best component centers using simple NMS."""
    components = sorted(components, key=lambda component: component.score, reverse=True)

    selected: list[tuple[int, int]] = []

    for component in components:
        # Use the geometric contour center, not the pixel centroid.
        x = int(round(component.circle_x))
        y = int(round(component.circle_y))

        is_duplicate = False

        for selected_x, selected_y in selected:
            dx = x - selected_x
            dy = y - selected_y

            if dx * dx + dy * dy < min_distance * min_distance:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        selected.append((x, y))

        if len(selected) == expected_count:
            break

    selected.sort(key=lambda point: (point[1], point[0]))
    return selected


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
