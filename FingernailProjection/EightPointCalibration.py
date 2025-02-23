from typing import Tuple
import cv2
from helper import picke_load, pickle_save, join_string
import numpy as np
import cv2


class EightPointCalibration:
    """Calibration using 4 points for the projection and 4 points for the screen."""

    def __init__(self):
        self.calibration_points_screen: list[Tuple[int, int]] = []
        self.calibration_points_projector: list[Tuple[int, int]] = []
        self.calibration_transform = picke_load("calibration_transform.pkl")

    def calibrate_if_all_points_are_set(self):
        if len(self.calibration_points_screen) == 4 and len(self.calibration_points_projector) == 4:
            self.calibration_transform, _ = cv2.findHomography(
                np.array(self.calibration_points_screen, dtype=np.float32),
                np.array(self.calibration_points_projector, dtype=np.float32),
            )
            pickle_save("calibration_transform.pkl", self.calibration_transform)
            print(
                f"Calibrated camera\n{join_string(self.calibration_points_screen)} (screen)\n{join_string(self.calibration_points_projector)} (projector)"
            )

    def add_calibration_point(self, points, position):
        if len(points) == 4:
            points.clear()

        points.append(position)
        print(f"Added camera calibration point {len(points)}: {position[0]}|{position[1]}")

        self.calibrate_if_all_points_are_set()

    def add_screen_point(self, position):
        self.add_calibration_point(self.calibration_points_screen, position)

    def add_projector_point(self, position):
        self.add_calibration_point(self.calibration_points_projector, position)
