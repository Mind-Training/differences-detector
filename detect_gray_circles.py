"""
detect_gray_circles.py
----------------------
Detects gray circle markers in puzzle images and returns their center coordinates.

Usage:
    python detect_gray_circles.py <image_path>

Returns:
    List of (x, y) tuples with the center of each detected gray circle.
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from differences_detector import detect_difference_points


def detect_gray_circles(
    image_path: str,
    gray_min: int = 120,
    gray_max: int = 215,
    area_min: int = 500,
    area_max: int = 50000,
) -> list[tuple[int, int]]:
    """
    Detect gray circle markers in an image and return their center coordinates.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    gray_min : int
        Lower bound of the gray pixel range (0-255). Default: 120.
    gray_max : int
        Upper bound of the gray pixel range (0-255). Default: 215.
    area_min : int
        Minimum contour area to consider. Default: 500.
    area_max : int
        Maximum contour area to consider. Default: 50000.

    Returns
    -------
    list of (x, y) tuples
        Center coordinates of each detected gray circle,
        sorted top-to-bottom, left-to-right.
    """
    return detect_difference_points(
        image_path,
        gray_min=gray_min,
        gray_max=gray_max,
        area_min=area_min,
        area_max=area_max,
    )


def visualize(image_path: str, circles: list[tuple[int, int]], output_path: str) -> None:
    """Draw detected circles on the image and save the result."""
    img = cv2.imread(image_path)
    for i, (cx, cy) in enumerate(circles):
        cv2.circle(img, (cx, cy), 50, (0, 0, 255), 3)
        cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)
        cv2.putText(
            img, str(i + 1), (cx + 12, cy - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
        )
    cv2.imwrite(output_path, img)
    print(f"Visualization saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_gray_circles.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "detected_circles.png"

    centers = detect_gray_circles(image_path)

    print(f"\nDetected {len(centers)} gray circle(s):")
    for i, (x, y) in enumerate(centers):
        print(f"  Circle {i + 1}: ({x}, {y})")

    print(f"\nAs array: {centers}")

    visualize(image_path, centers, output_path)
