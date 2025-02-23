import cv2
import numpy as np


def find_white_space_topleft(
    binary_image: np.ndarray, rectangle: tuple[int, int]
) -> tuple[int, int] | None:
    """
    Finds the first available white space in the mask using an integral image for efficiency.
    Returns the top-left (x, y) position or None if no space is found.
    """
    image_height, image_width = binary_image.shape
    rect_width, rect_height = rectangle

    # Compute the integral image
    integral_image = cv2.integral(binary_image)

    for y in range(image_height - rect_height + 1):
        for x in range(image_width - rect_width + 1):
            # Sum of pixels in the window using the integral image
            total_white = (
                integral_image[y + rect_height, x + rect_width]
                - integral_image[y, x + rect_width]
                - integral_image[y + rect_height, x]
                + integral_image[y, x]
            )

            # If the window is fully white, return position
            if total_white == rect_width * rect_height:
                return (x, y)

    return None  # No suitable space found


def find_white_space_topleft_spiral(
    binary_image: np.ndarray, rectangle: tuple[int, int], start: tuple[int, int], step: int = 1
) -> tuple[int, int] | None:
    """
    Finds the first available white space in the mask, expanding outward in a spiral from (start_x, start_y).
    Returns the top-left (x, y) position or None if no space is found.
    """
    image_height, image_width = binary_image.shape
    rect_width, rect_height = rectangle
    start_x, start_y = start

    # Compute the integral image
    integral_image = cv2.integral(binary_image)

    # Spiral search setup
    x, y = start_x, start_y
    dx, dy = step, 0  # Initial movement direction (right)
    segment_length = 1  # Steps in the current direction before turning
    steps_taken = 0  # Steps in the current segment
    segment_passes = 0  # Number of times segment length has been used (every 2 times, it increases)

    while 0 <= x < image_width and 0 <= y < image_height:
        # Check if the rectangle fits inside the image bounds
        if x + rect_width <= image_width and y + rect_height <= image_height:
            # Compute sum of pixels in the window using the integral image
            total_white = (
                integral_image[y + rect_height, x + rect_width]
                - integral_image[y, x + rect_width]
                - integral_image[y + rect_height, x]
                + integral_image[y, x]
            )

            # If the window is fully white, return position
            if total_white == rect_width * rect_height:
                return (x, y)

        # Move in the current direction
        x += dx
        y += dy
        steps_taken += 1

        # If we've reached the end of the current segment length
        if steps_taken == segment_length:
            steps_taken = 0  # Reset step counter
            dx, dy = -dy, dx  # Rotate direction (right → down → left → up)
            segment_passes += 1  # Count how many segments we've completed

            # Every two turns, increase the segment length
            if segment_passes % 2 == 0:
                segment_length += step

    return None  # No suitable space found
