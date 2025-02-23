import numpy as np
from typing import List, Tuple


def draw_pixel_blend(image: np.ndarray, x: int, y: int, color: Tuple[int, int, int], alpha: float):
    """Blends a pixel with a given alpha."""
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Ensure inside bounds
        bg = image[y, x]
        image[y, x] = (1 - alpha) * bg + alpha * np.array(color, dtype=np.uint8)


def draw_wu_line(
    image: np.ndarray, x0: float, y0: float, x1: float, y1: float, color: Tuple[int, int, int]
):
    """Draws an anti-aliased line using Wu's algorithm."""
    steep = abs(y1 - y0) > abs(x1 - x0)

    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx if dx != 0 else 1

    # First endpoint
    x_end = round(x0)
    y_end = y0 + gradient * (x_end - x0)
    x_gap = 1 - (x0 + 0.5 - int(x0 + 0.5))
    x_pxl1 = x_end
    y_pxl1 = int(y_end)

    if steep:
        draw_pixel_blend(image, y_pxl1, x_pxl1, color, 1 - (y_end - y_pxl1))
        draw_pixel_blend(image, y_pxl1 + 1, x_pxl1, color, y_end - y_pxl1)
    else:
        draw_pixel_blend(image, x_pxl1, y_pxl1, color, 1 - (y_end - y_pxl1))
        draw_pixel_blend(image, x_pxl1, y_pxl1 + 1, color, y_end - y_pxl1)

    y_inter = y_end + gradient  # First y-intersection

    # Main loop
    for x in range(x_pxl1 + 1, round(x1)):
        y = int(y_inter)
        if steep:
            draw_pixel_blend(image, y, x, color, 1 - (y_inter - y))
            draw_pixel_blend(image, y + 1, x, color, y_inter - y)
        else:
            draw_pixel_blend(image, x, y, color, 1 - (y_inter - y))
            draw_pixel_blend(image, x, y + 1, color, y_inter - y)
        y_inter += gradient


def draw_thin_polygon(
    image: np.ndarray, points: List[Tuple[int, int]], color: Tuple[int, int, int]
):
    """Draws an anti-aliased polygon by drawing anti-aliased lines between points."""
    num_points = len(points)
    for i in range(num_points):
        p1, p2 = points[i], points[(i + 1) % num_points]
        draw_wu_line(image, p1[0], p1[1], p2[0], p2[1], color)
