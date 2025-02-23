import numpy as np
import cv2


def simplify_contours(cv2_contours: list[np.ndarray], simplification: float) -> list[np.ndarray]:
    """Simplification can range from 0 (nothing) to 100 (very strong)."""
    if simplification != 0:
        # line_simplification = 0 -> epsilon = 10^-6
        # line_simplification = 100 -> epsilon = 10^-3
        epsilon = 10 ** (-6 + 3 * simplification)
        return [
            cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
            for contour in cv2_contours
        ]
    else:
        return cv2_contours


def filter_contours_by_length(
    cv2_contours: list[np.ndarray], min_length=0, max_length=float("inf")
) -> list[np.ndarray]:
    if max_length != 0:
        return [
            contour
            for contour in cv2_contours
            if (min_length <= cv2.arcLength(contour, False) <= max_length)
        ]
    else:
        return cv2_contours
