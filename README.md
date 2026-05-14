# differences-detector

Small Python package for detecting gray circular difference markers in images.

## Install

```bash
pip install -e .
```

## Python API

```python
from differences_detector import detect_difference_points

points = detect_difference_points("test_S.png")
print(points)
```

The function returns a list of `(x, y)` coordinate tuples, sorted from top to
bottom and then left to right.

You can also pass an image already loaded with OpenCV:

```python
import cv2
from differences_detector import detect_difference_points

image = cv2.imread("test_S.png")
points = detect_difference_points(image)
```

## CLI

```bash
detect-differences test_S.png --output resultado.png
```
# differences-detector
