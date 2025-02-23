import matlike_utils
from drawing import cv2, np
from prototyping.plotting import cv2, np
from skin_detect_config import SkinDetectConfig
from tuple_helper import np


import numpy as np


def skincolor_threshold_image(
    bgr_image: np.ndarray,
    config: SkinDetectConfig,
    inverse: bool = False,
) -> np.ndarray:
    """
    Thanks to: https://github.com/MelikaRad/CV_SkinDetection
    A function to detect skin in an input image using the provided configuration.
    Returns a grayscale mask.

    Required time by scaling factor (including downscaling with cv2.INTER_AREA):
    700 ms for full width
    200 ms for 1/2 width
    50 ms for 1/4 width
    25 ms for 1/10 width
    """
    # Noise removal
    image = cv2.medianBlur(
        bgr_image, matlike_utils.kernel_size_from_radius(config.blurr_kernel_size)
    )

    # Converting image to HSV and YCrCb color spaces
    bgr = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Defining the kernel for morphology tasks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.kernel_size, config.kernel_size))

    # Creating BGR mask
    bgr_mask = cv2.inRange(
        bgr,
        (config.lower_blu, config.lower_grn, config.lower_red),
        (255, 255, 255),
    )
    bgr_mask = cv2.morphologyEx(bgr_mask, cv2.MORPH_OPEN, kernel)
    bgr_mask = cv2.morphologyEx(bgr_mask, cv2.MORPH_CLOSE, kernel)

    red_greater_green = bgr[:, :, 2] > bgr[:, :, 1]  # red greater than
    red_greater_blue = bgr[:, :, 2] > bgr[:, :, 0]
    red_far_from_green = (bgr[:, :, 2] - bgr[:, :, 1]) > config.red_green_distance
    red_distance_mask = (red_greater_green & red_greater_blue & red_far_from_green).astype("uint8")

    bgr_mask = cv2.bitwise_and(bgr_mask, red_distance_mask)

    # Creating HSV mask
    hsv_mask = cv2.inRange(
        hsv,
        (config.lower_hue, config.lower_sat, config.lower_val),
        (config.upper_hue, config.upper_sat, config.upper_val),
    )
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)

    hsv_based_mask = cv2.bitwise_and(bgr_mask, hsv_mask)

    # Creating YCrCb mask
    chrominance_mask = cv2.inRange(
        ycrcb,
        (config.lower_luminance, config.lower_redchrome, config.lower_bluechrome),
        (config.upper_luminance, config.upper_redchrome, config.upper_bluechrome),
    )
    chrominance_mask = cv2.morphologyEx(chrominance_mask, cv2.MORPH_OPEN, kernel)
    chrominance_mask = cv2.morphologyEx(chrominance_mask, cv2.MORPH_CLOSE, kernel)

    Cr = ycrcb[:, :, 1]
    Cb = ycrcb[:, :, 2]

    chroninance_mask = (
        (Cr <= config.cr_cb_upper_1 * Cb + config.cr_cb_offset_1)
        & (Cr >= config.cr_cb_lower_1 * Cb + config.cr_cb_offset_2)
        & (Cr >= config.cr_cb_lower_2 * Cb + config.cr_cb_offset_3)
        & (Cr <= config.cr_cb_upper_2 * Cb + config.cr_cb_offset_4)
        & (Cr <= config.cr_cb_upper_3 * Cb + config.cr_cb_offset_5)
    ).astype("uint8")

    chrominance_mask = cv2.bitwise_and(chrominance_mask, chroninance_mask)
    chrominance_based_mask = cv2.bitwise_and(bgr_mask, chrominance_mask)

    # Either the hsv or the ycrcb mask needs to be true
    final_mask = cv2.bitwise_or(hsv_based_mask, chrominance_based_mask)

    # Calculate the mask of all skintones (non-skintones are black)
    skincolor_and_black = cv2.bitwise_and(bgr_image, bgr_image, mask=final_mask)

    # Apply thresholding (skintones become white)
    _, mask = cv2.threshold(
        skincolor_and_black, 1, 255, cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    )

    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
