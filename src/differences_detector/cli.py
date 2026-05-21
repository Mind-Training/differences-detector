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

    parser.add_argument("--gray-min", type=int, default=85)
    parser.add_argument("--gray-max", type=int, default=170)

    # Important:
    # These default to None so detector.py can use its adaptive scaling.
    parser.add_argument("--area-min", type=int, default=25)
    parser.add_argument("--area-max", type=int, default=5000)
    parser.add_argument("--radius-min", type=int, default=10)
    parser.add_argument("--radius-max", type=int, default=35)
    parser.add_argument("--ring-width", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--min-distance", type=int, default=24)
    parser.add_argument("--expected-count", type=int, default=8)

    args = parser.parse_args()

    points = detect_difference_points(
        args.image,
        gray_min=args.gray_min,
        gray_max=args.gray_max,
        area_min=args.area_min,
        area_max=args.area_max,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        ring_width=args.ring_width,
        score_threshold=args.score_threshold,
        min_distance=args.min_distance,
        expected_count=args.expected_count,
    )

    print(points)
    print(f"Detected {len(points)} differences.")

    if args.output:
        draw_detected_points(args.image, points, args.output)


if __name__ == "__main__":
    main()
