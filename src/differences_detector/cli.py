"""Command-line interface for differences-detector."""

from __future__ import annotations

import argparse

from .detector import detect_difference_points, draw_detected_points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect gray circular difference markers in an image."
    )
    parser.add_argument("image", help="Input image path.")
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output image path with detected points drawn in red.",
    )
    parser.add_argument("--gray-min", type=int, default=120)
    parser.add_argument("--gray-max", type=int, default=215)
    parser.add_argument("--area-min", type=int, default=500)
    parser.add_argument("--area-max", type=int, default=50000)
    args = parser.parse_args()

    points = detect_difference_points(
        args.image,
        gray_min=args.gray_min,
        gray_max=args.gray_max,
        area_min=args.area_min,
        area_max=args.area_max,
    )

    print(points)

    if args.output:
        draw_detected_points(args.image, points, args.output)
