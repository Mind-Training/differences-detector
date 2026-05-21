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
    area_min: int | None = None,
    area_max: int | None = None,
    radius_min: int | None = None,
    radius_max: int | None = None,
    ring_width: int | None = None,
    score_threshold: float = 0.20,
    min_distance: int | None = None,
) -> list[tuple[int, int]]:
    """Return the center coordinates of gray circular markers in an image."""
    img = _load_image(image)
    gray = _to_grayscale(img)

    height, width = gray.shape[:2]
    scale = min(width, height) / 1063.0

    if radius_min is None:
        radius_min = max(10, round(30 * scale))

    if radius_max is None:
        radius_max = max(radius_min + 4, round(65 * scale))

    if ring_width is None:
        ring_width = max(4, round(10 * scale))

    if min_distance is None:
        min_distance = max(10, round(radius_min * 0.85))

    if area_min is None:
        area_min = max(40, round(500 * scale * scale))

    if area_max is None:
        area_max = max(area_min + 1, round(50000 * scale * scale))

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

    signal = _marker_signal_from_local_contrast(
        gray,
        gray_min=gray_min,
        gray_max=gray_max,
    )

    if not np.any(signal):
        return []

    candidates = _find_ring_candidates(
        signal,
        area_min=area_min,
        area_max=area_max,
        radius_min=radius_min,
        radius_max=radius_max,
        ring_width=ring_width,
        score_threshold=score_threshold,
        min_distance=min_distance,
    )

    points = _select_best_candidates(
        candidates,
        signal=signal,
        gray=gray,
        width=width,
        height=height,
        expected_count=8,
        min_distance=max(min_distance, radius_min),
        radius_min=radius_min,
        radius_max=radius_max,
        ring_width=ring_width,
    )

    return points


def _marker_signal_from_local_contrast(
    gray: np.ndarray,
    *,
    gray_min: int,
    gray_max: int,
) -> np.ndarray:
    """Return marker likelihood for darker mid-gray rings."""
    gray_f = gray.astype(np.float32)

    height, width = gray.shape[:2]
    scale = min(width, height) / 1063.0

    sigma = max(6.0, 22.0 * scale)

    background = cv2.GaussianBlur(
        gray_f,
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )

    # Only pixels darker than their local background.
    dark_difference = background - gray_f

    # IMPORTANT:
    # The markers are mid-gray. Exclude black line art and light gray background.
    marker_gray_mask = ((gray_f >= 125) & (gray_f <= 178)).astype(np.float32)

    contrast_min = max(3.0, 7.0 * scale)
    contrast_max = max(14.0, 34.0 * scale)

    signal = (dark_difference - contrast_min) / (contrast_max - contrast_min)
    signal = np.clip(signal, 0.0, 1.0)

    signal *= marker_gray_mask

    signal = cv2.GaussianBlur(
        signal,
        ksize=(0, 0),
        sigmaX=max(0.6, 1.1 * scale),
        sigmaY=max(0.6, 1.1 * scale),
        borderType=cv2.BORDER_REFLECT,
    )

    return signal.astype(np.float32)


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
    signal: np.ndarray,
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
    valid_pixels = np.ones_like(signal, dtype=np.float32)

    for radius in range(radius_min, radius_max + 1, 2):
        ring_kernel = _annulus_kernel(radius, ring_width)
        full_ring_area = float(ring_kernel.sum())

        if not area_min < full_ring_area < area_max:
            continue

        inner_radius = max(2, radius - ring_width * 2)
        inner_kernel = _disk_kernel(inner_radius)
        full_inner_area = float(inner_kernel.sum())

        outer_radius = radius + max(3, ring_width)
        outer_kernel = _annulus_kernel_between(
            inner_radius=radius + 1,
            outer_radius=outer_radius,
        )
        full_outer_area = float(outer_kernel.sum())

        ring_sum = cv2.filter2D(
            signal,
            ddepth=cv2.CV_32F,
            kernel=ring_kernel,
            borderType=cv2.BORDER_CONSTANT,
        )

        visible_ring_area = cv2.filter2D(
            valid_pixels,
            ddepth=cv2.CV_32F,
            kernel=ring_kernel,
            borderType=cv2.BORDER_CONSTANT,
        )

        inner_sum = cv2.filter2D(
            signal,
            ddepth=cv2.CV_32F,
            kernel=inner_kernel,
            borderType=cv2.BORDER_CONSTANT,
        )

        visible_inner_area = cv2.filter2D(
            valid_pixels,
            ddepth=cv2.CV_32F,
            kernel=inner_kernel,
            borderType=cv2.BORDER_CONSTANT,
        )

        outer_sum = cv2.filter2D(
            signal,
            ddepth=cv2.CV_32F,
            kernel=outer_kernel,
            borderType=cv2.BORDER_CONSTANT,
        )

        visible_outer_area = cv2.filter2D(
            valid_pixels,
            ddepth=cv2.CV_32F,
            kernel=outer_kernel,
            borderType=cv2.BORDER_CONSTANT,
        )

        ring_score = np.zeros_like(signal, dtype=np.float32)
        inner_score = np.zeros_like(signal, dtype=np.float32)
        outer_score = np.zeros_like(signal, dtype=np.float32)

        valid_ring = visible_ring_area >= full_ring_area * 0.30
        valid_inner = visible_inner_area >= max(1.0, full_inner_area * 0.30)
        valid_outer = visible_outer_area >= max(1.0, full_outer_area * 0.30)

        ring_score[valid_ring] = ring_sum[valid_ring] / visible_ring_area[valid_ring]
        inner_score[valid_inner] = (
            inner_sum[valid_inner] / visible_inner_area[valid_inner]
        )
        outer_score[valid_outer] = (
            outer_sum[valid_outer] / visible_outer_area[valid_outer]
        )

        # A real marker should have local contrast mainly on the ring.
        # Uniform gray background has almost zero local contrast.
        response = ring_score - (inner_score * 0.25) - (outer_score * 0.15)

        local_max = cv2.dilate(
            response,
            np.ones((local_kernel_size, local_kernel_size), dtype=np.uint8),
        )

        ys, xs = np.where(
            (response == local_max)
            & (response >= score_threshold)
            & (ring_score >= score_threshold)
        )

        candidates.extend(
            (float(response[y, x]), int(x), int(y)) for x, y in zip(xs, ys)
        )

    return candidates


def _select_best_candidates(
    candidates: list[tuple[float, int, int]],
    *,
    signal: np.ndarray,
    gray: np.ndarray,
    width: int,
    height: int,
    expected_count: int,
    min_distance: int,
    radius_min: int,
    radius_max: int,
    ring_width: int,
) -> list[tuple[int, int]]:
    """Select the best marker centers using score, ring coverage and NMS."""
    if not candidates:
        return []

    ranked_candidates: list[tuple[float, int, int]] = []

    for score, x, y in candidates:
        ring_quality = _candidate_ring_quality(
            signal,
            gray,
            x=x,
            y=y,
            radius_min=radius_min,
            radius_max=radius_max,
            ring_width=ring_width,
        )

        # Reject candidates that do not look like a circular marker.
        # This removes most gray-background/border artifacts in 226.png.
        if ring_quality < 0.38:
            continue

        adjusted_score = score * (0.45 + ring_quality)

        # Strong penalty for exact convolution artifacts at the image border.
        if x <= 1 or y <= 1 or x >= width - 2 or y >= height - 2:
            adjusted_score *= 0.15

        # Mild border penalty. Real clipped circles are still allowed.
        border_margin = max(6, min_distance // 2)

        if (
            x < border_margin
            or y < border_margin
            or x > width - border_margin
            or y > height - border_margin
        ):
            adjusted_score *= 0.75

        ranked_candidates.append((adjusted_score, x, y))

    ranked_candidates.sort(reverse=True, key=lambda item: item[0])

    selected: list[tuple[int, int]] = []

    nms_distance = max(min_distance, 24)

    for _, x, y in ranked_candidates:
        is_too_close = False

        for selected_x, selected_y in selected:
            dx = x - selected_x
            dy = y - selected_y

            if dx * dx + dy * dy < nms_distance * nms_distance:
                is_too_close = True
                break

        if is_too_close:
            continue

        selected.append((x, y))

        if len(selected) == expected_count:
            break

    selected.sort(key=lambda point: (point[1], point[0]))
    return selected


def _candidate_ring_quality(
    signal: np.ndarray,
    gray: np.ndarray,
    *,
    x: int,
    y: int,
    radius_min: int,
    radius_max: int,
    ring_width: int,
) -> float:
    """Estimate whether a candidate looks like a real mid-gray circular ring."""
    height, width = signal.shape[:2]

    best_quality = 0.0

    angle_count = 64
    radial_offsets = range(
        -max(1, ring_width // 2),
        max(1, ring_width // 2) + 1,
    )

    for radius in range(radius_min, radius_max + 1):
        active_sectors = 0
        visible_sectors = 0
        sector_values: list[float] = []
        ring_gray_values: list[float] = []

        for angle_index in range(angle_count):
            angle = (2.0 * np.pi * angle_index) / angle_count

            values: list[float] = []
            gray_values: list[float] = []

            for offset in radial_offsets:
                sample_radius = radius + offset

                px = int(round(x + np.cos(angle) * sample_radius))
                py = int(round(y + np.sin(angle) * sample_radius))

                if 0 <= px < width and 0 <= py < height:
                    values.append(float(signal[py, px]))
                    gray_values.append(float(gray[py, px]))

            if not values:
                continue

            visible_sectors += 1

            sector_value = max(values)
            sector_values.append(sector_value)

            # Use the strongest sample in this sector.
            best_index = int(np.argmax(values))
            ring_gray_values.append(gray_values[best_index])

            if sector_value >= 0.16:
                active_sectors += 1

        if visible_sectors == 0 or not ring_gray_values:
            continue

        coverage = active_sectors / visible_sectors
        mean_strength = float(np.mean(sector_values))

        ring_gray_mean = float(np.mean(ring_gray_values))
        ring_gray_std = float(np.std(ring_gray_values))

        # Real markers are mid-gray.
        if not 120 <= ring_gray_mean <= 182:
            continue

        # Reject candidates made from mixed black/white drawing edges.
        if ring_gray_std > 42:
            continue

        quality = (coverage * 0.70) + (mean_strength * 0.30)

        # Prefer rings with the expected marker tone.
        if 135 <= ring_gray_mean <= 170:
            quality *= 1.20

        best_quality = max(best_quality, quality)

    return best_quality


def _annulus_kernel(radius: int, ring_width: int) -> np.ndarray:
    inner_radius = max(1, radius - ring_width)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    distance_squared = xx * xx + yy * yy

    return (
        (distance_squared <= radius * radius)
        & (distance_squared >= inner_radius * inner_radius)
    ).astype(np.float32)


def _disk_kernel(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    distance_squared = xx * xx + yy * yy

    return (distance_squared <= radius * radius).astype(np.float32)


def _annulus_kernel_between(*, inner_radius: int, outer_radius: int) -> np.ndarray:
    yy, xx = np.ogrid[
        -outer_radius : outer_radius + 1, -outer_radius : outer_radius + 1
    ]
    distance_squared = xx * xx + yy * yy

    return (
        (distance_squared <= outer_radius * outer_radius)
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
