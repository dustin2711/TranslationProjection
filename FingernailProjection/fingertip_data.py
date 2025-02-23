from dataclasses import dataclass
from math import sqrt
import math
import sys
import os
from typing import Optional
from shapely import MultiPoint
from skincolor_threshold_image import skincolor_threshold_image

sys.path.append(os.path.abspath(".."))
import linestring_utils
from config import Config
from prototyping.plotting import *
from drawing import *
import matlike_utils
from tuple_helper import *
from config import Config
import numpy as np
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from stopwatch import Stopwatch


def get_enclosing_contour(image: np.ndarray, point: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Finds and returns the contour enclosing a given point. If no contour encloses the point,
    returns the closest contour.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    closest_contour = None
    min_distance = float("inf")

    for contour in contours:
        distance = cv2.pointPolygonTest(contour, point, measureDist=True)
        if distance >= 0:  # Point is inside the contour
            return contour
        # Update closest contour if this one is closer
        if abs(distance) < min_distance:
            min_distance = abs(distance)
            closest_contour = contour

    return closest_contour


@dataclass
class FingertipData:
    mask: np.ndarray
    """ The thresholded image."""
    mask_opened: np.ndarray
    """ The thresholded image after opening."""
    mask_opened_closed: np.ndarray
    """ The thresholded image after opening and closing."""
    linestring: LineString
    """ The linestring of the finger."""
    fingertip_center: Tuple2
    """ The center of the fingertip. """
    finger_apex: Tuple2
    """ The finger-approximating halfcircle's center. """
    half_finger_width: float
    """ The finger-approximating halfcircle's radius. """
    most_upper_position: Tuple2

    @staticmethod
    def create_from_mask(
        mask: np.ndarray,
        config: Config,
        fingertip_landmark: Tuple[int, int],
    ) -> "FingertipData":
        """
        Processes the image to compute masks, linestrings, and circle parameters.
        Premise: The finger points straight up from the bottom, like a clock showing 12.
        """

        finger_width, height = mask.shape

        stopwatch = Stopwatch().restart(config.log_fingertip_data_retriever)

        # Create the skin color masks
        mask_opened = matlike_utils.open_image(mask, config.skincolor_open_radius)
        mask_opened_closed = matlike_utils.close(mask_opened, config.skincolor_close_radius)
        stopwatch.log("Opening and closing")

        data = FingertipData(mask, mask_opened, mask_opened_closed, None, (0, 0), (0, 0), 0, (0, 0))

        # Find and process the longest contour
        contour = get_enclosing_contour(mask_opened_closed, fingertip_landmark)
        stopwatch.log("Get enclosing contour")
        if contour is None:
            print("No contours found.")
            return data

        data.linestring = matlike_utils.convert_contour_to_closed_linestring(contour)
        stopwatch.log("Convert to linestring")

        if config.linestring_smoothing != 0:
            resampled_linestring = linestring_utils.resample_linestring(
                data.linestring, config.linestring_resample_distance
            )
            data.linestring = matlike_utils.smooth_linestring(
                resampled_linestring, config.linestring_smoothing
            )
            stopwatch.log("Resampling and smoothing")

        # Use the finger widths and the most upper contour position to create a circle
        data.most_upper_position = Tuple2(min(data.linestring.coords, key=lambda p: p[1])).as_int

        # Scan the width of the contour starting at the height of the landmark
        # The finger width is approx. the median of these withs
        finger_widths = []
        center_positions_x = []
        for height_of_line in range(
            int(fingertip_landmark[1]),
            height,
            config.finger_width_sample_rate_px,
        ):
            intersection: BaseGeometry = data.linestring.intersection(
                LineString([(0, height_of_line), (finger_width, height_of_line)])
            )
            if isinstance(intersection, MultiPoint) and len(intersection.geoms) == 2:
                first, second = intersection.geoms  # The order is arbitrary
                finger_widths.append(abs(second.x - first.x))
                center_positions_x.append(0.5 * (second.x + first.x))

        if len(finger_widths) == 0:
            print("No finger widths could be sampled!")
        else:
            finger_center_x = np.median(center_positions_x)
            finger_width = np.median(finger_widths)

            data.fingertip_center = Tuple2(finger_center_x, data.most_upper_position[1]).as_int
            data.half_finger_width = 0.5 * finger_width
            data.finger_apex = Tuple2(
                finger_center_x,
                data.most_upper_position[1],
            ).as_int
            stopwatch.log("Getting radius and center")

        return data
