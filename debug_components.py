from pathlib import Path

import cv2
import numpy as np


def debug(image_path: str, gray_min: int = 85, gray_max: int = 170) -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = ((gray >= gray_min) & (gray <= gray_max)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    components = []

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]

        if area < 25:
            continue

        components.append((area, x, y, w, h, round(cx), round(cy)))

    components.sort(reverse=True)

    print(f"\n{image_path}")
    print(f"components={len(components)}")
    print("area, x, y, w, h, cx, cy")

    for row in components[:30]:
        print(row)

    debug_img = img.copy()

    for area, x, y, w, h, cx, cy in components[:30]:
        color = (0, 0, 255)

        if 25 <= w <= 70 and 25 <= h <= 70 and 0.45 <= w / max(1, h) <= 2.2:
            color = (0, 255, 0)

        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
        cv2.circle(debug_img, (cx, cy), 3, (255, 0, 0), -1)
        cv2.putText(
            debug_img,
            str(area),
            (x, max(0, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    stem = Path(image_path).stem
    cv2.imwrite(f"debug-mask-{stem}.png", mask)
    cv2.imwrite(f"debug-components-{stem}.png", debug_img)


for image_id in [225, 226, 227, 237, 238, 239]:
    debug(f"{image_id}.png")
