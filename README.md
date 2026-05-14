# differences-detector

Python library for detecting gray circular difference markers in puzzle images.

The main function receives an image and returns a list of detected points as
pixel coordinates.

## Installation

For local development, install the package from this repository:

```bash
pip install -e .
```

If you use Poetry:

```bash
poetry install
```

## Basic Usage

```python
from differences_detector import detect_difference_points

points = detect_difference_points("image.png")
print(points)
```

Example output:

```python
[(301, 250), (80, 270), (448, 596)]
```

Each point is returned as an `(x, y)` tuple:

- `x`: horizontal pixel coordinate, counted from the left side of the image.
- `y`: vertical pixel coordinate, counted from the top of the image.

The points are returned sorted from top to bottom and then from left to right.

## Input Formats

The library accepts an image path:

```python
from differences_detector import detect_difference_points

points = detect_difference_points("image.png")
```

It also accepts an image already loaded as a NumPy array, for example with
OpenCV:

```python
import cv2
from differences_detector import detect_difference_points

image = cv2.imread("image.png")
points = detect_difference_points(image)
```

Supported array formats:

- Grayscale image: shape `(height, width)`.
- BGR image: shape `(height, width, 3)`.
- BGRA image: shape `(height, width, 4)`.

## API Reference

```python
detect_difference_points(
    image,
    *,
    gray_min=120,
    gray_max=215,
    area_min=500,
    area_max=50000,
) -> list[tuple[int, int]]
```

Parameters:

- `image`: image path or NumPy image array.
- `gray_min`: minimum grayscale value used to isolate the markers.
- `gray_max`: maximum grayscale value used to isolate the markers.
- `area_min`: minimum contour area accepted as a marker.
- `area_max`: maximum contour area accepted as a marker.

Return value:

```python
list[tuple[int, int]]
```

The list contains the center coordinate of each detected marker.

## Integration Example

```python
from differences_detector import detect_difference_points


def process_puzzle_image(image_path: str) -> dict:
    points = detect_difference_points(image_path)

    return {
        "count": len(points),
        "points": [{"x": x, "y": y} for x, y in points],
    }


result = process_puzzle_image("puzzle.png")
print(result)
```

Example output:

```python
{
    "count": 3,
    "points": [
        {"x": 301, "y": 250},
        {"x": 80, "y": 270},
        {"x": 448, "y": 596},
    ],
}
```

## Drawing Detected Points

The package includes an optional helper for saving a visualization with circles
drawn over the detected points:

```python
from differences_detector import detect_difference_points, draw_detected_points

image_path = "puzzle.png"
points = detect_difference_points(image_path)

draw_detected_points(image_path, points, "result.png")
```

This helper is only intended for debugging or visualization. The main library
output remains the coordinate list.

## Command Line Usage

Detect points and print the coordinate list:

```bash
detect-differences puzzle.png
```

Detect points and save a visualization image:

```bash
detect-differences puzzle.png --output result.png
```

Optional tuning parameters:

```bash
detect-differences puzzle.png \
  --gray-min 120 \
  --gray-max 215 \
  --area-min 500 \
  --area-max 50000
```

## Error Handling

If the image path cannot be loaded, the library raises:

```python
FileNotFoundError
```

If an empty or unsupported NumPy array is passed, the library raises:

```python
ValueError
```

Example:

```python
from differences_detector import detect_difference_points

try:
    points = detect_difference_points("missing.png")
except FileNotFoundError as error:
    print(error)
```

## Recommended Usage in Another Project

In another Python project, import only the public API:

```python
from differences_detector import detect_difference_points
```

Avoid importing internal helpers directly, for example:

```python
from differences_detector.detector import _load_image
```

Private helpers may change without notice.
