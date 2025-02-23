from typing import List

import cv2
import numpy as np
from parameter_config import ParameterConfig


from param import Integer, Number


class SkinDetectConfig(ParameterConfig):
    category_hsv = True
    lower_hue = Integer(0, bounds=(0, 255))
    upper_hue = Integer(50, bounds=(0, 255))
    lower_sat = Integer(58, bounds=(0, 255))
    upper_sat = Integer(174, bounds=(0, 255))
    lower_val = Integer(0, bounds=(0, 255))
    upper_val = Integer(255, bounds=(0, 255))

    category_rgb = True
    lower_red = Integer(95, bounds=(0, 255))
    lower_grn = Integer(40, bounds=(0, 255))
    lower_blu = Integer(20, bounds=(0, 255))
    red_green_distance = Integer(15, bounds=(0, 255))

    category_ycrcb = True
    lower_luminance = Integer(80, bounds=(0, 255))
    upper_luminance = Integer(255, bounds=(0, 255))
    lower_redchrome = Integer(135, bounds=(-180, 180))
    upper_redchrome = Integer(180, bounds=(-180, 180))
    lower_bluechrome = Integer(85, bounds=(-180, 180))
    upper_bluechrome = Integer(180, bounds=(-180, 180))
    cr_cb_lower_1 = Number(0.3448, bounds=(0, 2.0))
    cr_cb_upper_1 = Number(1.5862, bounds=(0, 2.0))
    cr_cb_lower_2 = Number(-4.5652, bounds=(-10.0, 0))
    cr_cb_upper_2 = Number(-1.15, bounds=(-10.0, 0))
    cr_cb_upper_3 = Number(-2.2857, bounds=(-10.0, 0))
    cr_cb_offset_1 = Integer(20, bounds=(0, 500))
    cr_cb_offset_2 = Number(76.2069, bounds=(0, 500))
    cr_cb_offset_3 = Number(234.5652, bounds=(0, 500))
    cr_cb_offset_4 = Number(301.75, bounds=(0, 500))
    cr_cb_offset_5 = Number(432.85, bounds=(0, 500))

    category_other = True
    blurr_kernel_size = Integer(5, bounds=(1, 50))
    kernel_size = Integer(3, bounds=(1, 50))

    @staticmethod
    def create_from_min_and_max_of_colors(rgb_colors: List[tuple]) -> "SkinDetectConfig":
        config = SkinDetectConfig()

        # Convert RGB list to NumPy array
        rgb_array = np.array(rgb_colors, dtype=np.uint8)

        # Extract R, G, B channels
        r, g, b = rgb_array[:, 0], rgb_array[:, 1], rgb_array[:, 2]

        # Convert RGB to HSV
        hsv_array = cv2.cvtColor(rgb_array.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        h, s, v = hsv_array[:, 0], hsv_array[:, 1], hsv_array[:, 2]

        # Convert RGB to YCbCr
        ycbcr_array = cv2.cvtColor(rgb_array.reshape(-1, 1, 3), cv2.COLOR_RGB2YCrCb).reshape(-1, 3)
        y, cb, cr = ycbcr_array[:, 0], ycbcr_array[:, 1], ycbcr_array[:, 2]

        # Update config directly
        old_lower_hue = config.lower_hue
        config.lower_hue = int(h.min())

        old_upper_hue = config.upper_hue
        config.upper_hue = int(h.max())

        print("hue")
        print(f"[{old_lower_hue}, {old_upper_hue}]")
        print(f"[{config.lower_hue}, {config.upper_hue}]")

        old_value = config.lower_sat
        config.lower_sat = int(s.min())
        print(f"{old_value} -> {config.lower_sat} lower_sat")

        old_value = config.upper_sat
        config.upper_sat = int(s.max())
        print(f"{old_value} -> {config.upper_sat} upper_sat")

        old_value = config.lower_val
        config.lower_val = int(v.min())
        print(f"{old_value} -> {config.lower_val} lower_val")

        old_value = config.upper_val
        config.upper_val = int(v.max())
        print(f"{old_value} -> {config.upper_val} upper_val")

        old_value = config.lower_red
        config.lower_red = int(r.min())
        print(f"{old_value} -> {config.lower_red} lower_red")

        old_value = config.upper_red
        config.upper_red = int(r.max())
        print(f"{old_value} -> {config.upper_red} upper_red")

        old_value = config.lower_grn
        config.lower_grn = int(g.min())
        print(f"{old_value} -> {config.lower_grn} lower_grn")

        old_value = config.upper_grn
        config.upper_grn = int(g.max())
        print(f"{old_value} -> {config.upper_grn} upper_grn")

        old_value = config.lower_blu
        config.lower_blu = int(b.min())
        print(f"{old_value} -> {config.lower_blu} lower_blu")

        old_value = config.upper_blu
        config.upper_blu = int(b.max())
        print(f"{old_value} -> {config.upper_blu} upper_blu")

        old_value = config.lower_luminance
        config.lower_luminance = int(y.min())
        print(f"{old_value} -> {config.lower_luminance} lower_luminance")

        old_value = config.upper_luminance
        config.upper_luminance = int(y.max())
        print(f"{old_value} -> {config.upper_luminance} upper_luminance")

        old_value = config.lower_bluechrome
        config.lower_bluechrome = int(cb.min())
        print(f"{old_value} -> {config.lower_bluechrome} lower_bluechrome")

        old_value = config.upper_bluechrome
        config.upper_bluechrome = int(cb.max())
        print(f"{old_value} -> {config.upper_bluechrome} upper_bluechrome")

        old_value = config.lower_redchrome
        config.lower_redchrome = int(cr.min())
        print(f"{old_value} -> {config.lower_redchrome} lower_redchrome")

        old_value = config.upper_redchrome
        config.upper_redchrome = int(cr.max())
        print(f"{old_value} -> {config.upper_redchrome} upper_redchrome")

        return config
