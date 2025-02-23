# Image Preprocessing Utilities
# Lei Mao
# University of Chicago
# 3/1/2018

from typing import Optional, Tuple
import cv2
import numpy as np


def inside_rect(rect, num_cols, num_rows):
    # Determine if the four corners of the rectangle are inside the rectangle with width and height
    # rect tuple
    # center (x,y), (width, height), angle of rotation (to the row)
    # center  The rectangle mass center.
    # center tuple (x, y): x is regarding to the width (number of columns) of the image, y is regarding to the height (number of rows) of the image.
    # size    Width and height of the rectangle.
    # angle   The rotation angle in a clockwise direction. When the angle is 0, 90, 180, 270 etc., the rectangle becomes an up-right rectangle.
    # Return:
    # True: if the rotated sub rectangle is side the up-right rectange
    # False: else

    rect_center = rect[0]
    rect_center_x = rect_center[0]
    rect_center_y = rect_center[1]

    if (rect_center_x < 0) or (rect_center_x > num_cols):
        return False
    if (rect_center_y < 0) or (rect_center_y > num_rows):
        return False

    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
    box = cv2.boxPoints(rect)

    x_max = int(np.max(box[:, 0]))
    x_min = int(np.min(box[:, 0]))
    y_max = int(np.max(box[:, 1]))
    y_min = int(np.min(box[:, 1]))

    if (x_max <= num_cols) and (x_min >= 0) and (y_max <= num_rows) and (y_min >= 0):
        return True
    else:
        return False


def rect_bbx(rect):
    # Rectangle bounding box for rotated rectangle
    # Example:
    # rotated rectangle: height 4, width 4, center (10, 10), angle 45 degree
    # bounding box for this rotated rectangle, height 4*sqrt(2), width 4*sqrt(2), center (10, 10), angle 0 degree

    box = cv2.boxPoints(rect)

    x_max = int(np.max(box[:, 0]))
    x_min = int(np.min(box[:, 0]))
    y_max = int(np.max(box[:, 1]))
    y_min = int(np.min(box[:, 1]))

    center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
    width = int(x_max - x_min)
    height = int(y_max - y_min)
    angle = 0

    return (center, (width, height), angle)


def image_rotate_without_crop(mat, angle):
    # https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
    # angle in degrees

    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    return rotated_mat


def get_croped_image_using_rectangle(image, rect: Tuple[Tuple[int, int], Tuple[int, int], float]):
    rect_center_x = rect[0][0]
    rect_center_y = rect[0][1]
    rect_width = rect[1][0]
    rect_height = rect[1][1]

    return image[
        rect_center_y - rect_height // 2 : rect_center_y + rect_height - rect_height // 2,
        rect_center_x - rect_width // 2 : rect_center_x + rect_width - rect_width // 2,
    ]


def crop_rotated_rectangle(
    image: np.ndarray, rect: Tuple[Tuple[int, int], Tuple[int, int], float]
) -> Optional[tuple[float, np.ndarray]]:
    # Crop a rotated rectangle from a image

    num_rows, num_cols = image.shape[:2]

    center, size, angle = rect
    center_x, center_y = center
    width, height = size

    # Check if the rectangle is inside bounds
    if not inside_rect(rect=rect, num_cols=num_cols, num_rows=num_rows):
        # Calculate the enlargement size: half the diagonal of the rectangle's max dimension
        max_dim = max(width, height)
        enlargement = int(np.ceil(np.sqrt(2) * max_dim))

        # Create a new image with black padding
        new_height = num_rows + 2 * enlargement
        new_width = num_cols + 2 * enlargement
        enlarged_image = np.zeros((new_height, new_width, *image.shape[2:]), dtype=image.dtype)

        # Place the original image in the center
        enlarged_image[
            enlargement : enlargement + num_rows, enlargement : enlargement + num_cols
        ] = image

        # Update the rectangle's center coordinates
        new_center = (center_x + enlargement, center_y + enlargement)
        rect = (new_center, size, angle)
        image = enlarged_image

    rotated_angle = rect[2]

    rect_bbx_upright = rect_bbx(rect=rect)
    rect_bbx_upright_image = get_croped_image_using_rectangle(image=image, rect=rect_bbx_upright)

    try:
        rotated_rect_bbx_upright_image = image_rotate_without_crop(
            mat=rect_bbx_upright_image, angle=rotated_angle
        )
    except:
        print("Could not rotate image.")
        return None

    rect_width = int(rect[1][0])
    rect_height = int(rect[1][1])

    crop_center = (
        rotated_rect_bbx_upright_image.shape[1] // 2,
        rotated_rect_bbx_upright_image.shape[0] // 2,
    )

    point = np.array(
        [center_x - width / 2, center_y - height / 2, 1]
    )  # Add 1 for homogeneous coordinates
    matrix = cv2.getRotationMatrix2D(
        (center_x, center_y), angle, 1
    )  # Get the transformation matrix
    top_left = np.dot(matrix, point)  # Apply the transformation

    return rotated_rect_bbx_upright_image[
        crop_center[1]
        - int(rect_height // 2) : crop_center[1]
        + (rect_height - int(rect_height // 2)),
        crop_center[0]
        - int(rect_width // 2) : crop_center[0]
        + (rect_width - int(rect_width // 2)),
    ]
