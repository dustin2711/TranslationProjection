from config import Config
from text_detection import TextDetection
from matlike_utils import *
from cv2.typing import MatLike

from enums import *
from text_detection import TextDetection
from cv2.typing import MatLike
from color_helper import *
from abc import ABC, abstractmethod
from stopwatch import Stopwatch
from image_display import ImageDisplay

_stopwatch = Stopwatch(logging_enabled=False)


class HandContourDetector(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.display = ImageDisplay(
            "HandContourDetector",
            Anchor.TOP_RIGHT,
            lambda: self.config.debug_screen_index,
            ScaleMode.FILL_HALF_SCREEN_HEIGHT,
        )

        self.reference_image = None

    @abstractmethod
    def threshold_image(self, image: MatLike) -> MatLike:
        """Returns the the hand contour the thresholded image."""
        pass

    def select_hand_contour(self, image: MatLike, detections: list[TextDetection]) -> LineString:

        # Ensure reference image is set
        if self.reference_image is None or not are_tuple_lengths_equal(image, self.reference_image):
            self.reference_image = image.copy()
            _stopwatch.logandrestart("Set reference image")

        thresholded_image = self.threshold_image(image)

        _stopwatch.restart(self.config.log_contour_detection)

        """Finds the hand contour in the thresholded image."""
        cv2_contours, _ = cv2.findContours(
            thresholded_image,
            ContourMode.EXTERNAL.value,
            cv2.CHAIN_APPROX_NONE,
        )

        _stopwatch.logandrestart("findContours")

        contours: list[LineString] = [
            convert_contour_to_closed_linestring(it) for it in cv2_contours
        ]

        if self.config.minimum_contour_length > 0:
            contours = [it for it in contours if it.length > self.config.minimum_contour_length]
            _stopwatch.logandrestart("filtered too short contours")

        # Filter out points in each line string outside the text
        if self.config.clip_contour_outside_of_text:

            box = TextDetection.get_total_axis_bounding_box(detections)
            _stopwatch.logandrestart("Got total bounding box")

            if box:
                for index, contour in enumerate(contours):
                    # If bounds are left or above the top left text position, then filter these points
                    left = box.top_left[0]
                    right = box.bottom_right[0]
                    top = box.top_left[1]
                    bot = box.bottom_right[1]
                    if (
                        contour.bounds[0] < left
                        or contour.bounds[1] < top
                        or contour.bounds[2] > right
                        or contour.bounds[3] > bot
                    ):
                        previous_point_count = len(contour.coords)
                        contours[index] = filter_linestring(
                            contour,
                            lambda point: left <= point[0] <= right and top <= point[1] <= bot,
                        )
                        _stopwatch.log(
                            f"Truncated contour {index} from {previous_point_count} to {len(contour.coords)}"
                        )
                contours = [it for it in contours if it.length > 0]

        _stopwatch.logandrestart("Truncated contours")

        match self.config.contour_selection_mode:
            case ContourSelectionMode.LONGEST:
                contour = max(contours, key=lambda string: string.length, default=None)
            case ContourSelectionMode.TOP_LEFT:
                contour = max(
                    contours,
                    key=lambda string: max(
                        Point(coord).distance(Point(9999, 9999)) for coord in string.coords
                    ),  # Calculate max distance for each contour
                    default=None,
                )

        _stopwatch.logandrestart("selected contour")

        if contour is None:
            pass

        # Check if the line string of the "hand" has a minimum length
        if contour is not None and len(contour.coords) >= 2:
            contour = smooth_linestring(contour, self.config.curve_averaging)
            image = image.copy()
            draw_linestring(image, contour, GREEN, 2)
            point(image, contour.centroid, GREEN, 5)

        _stopwatch.logandrestart("smooth_linestring and draw")

        if self.config.show_hand_contour:
            self.display.show(
                [
                    ensure_rgb(self.reference_image),
                    ensure_rgb(image),
                    # ensure_rgb(thresholded_image),
                ]
            )

        _stopwatch.logandrestart("show_hand_contour")

        return contour
