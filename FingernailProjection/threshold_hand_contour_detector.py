from hand_contour_detector import HandContourDetector
from matlike_utils import *
from cv2.typing import MatLike
from stopwatch import Stopwatch


class ThresholdHandContourDetector(HandContourDetector):
    """Uses filtering and thresholding to detect fingertips."""

    def threshold_image(self, image: MatLike) -> LineString:

        stopwatch = Stopwatch(logging_enabled=self.config.log_thresholding)
        stopwatch.restart()

        image = apply_gaussian(
            image,
            self.config.gaussian_kernel_radius,
            0,
        )
        image = ensure_grayscale(image)

        stopwatch.logandrestart("Blurred and greyscaled")

        match self.config.hand_contour_threshold_mode:
            case ThresholdMode.ADAPTIVE:
                image = apply_adaptive_threshold(
                    image,
                    self.config.adaptive_threshold_kernel_radius,
                    self.config.adaptive_threshold_offset,
                )
            case ThresholdMode.OTSU:
                image = apply_otsu_threshold(image)
            case ThresholdMode.MANUAL:
                image = apply_threshold(image, self.config.manual_threshold_value)

        stopwatch.logandrestart("Thresholded")

        if self.config.dilate_and_erode_kernel_radius > 0:
            # Remove text
            image = apply_morpholocial_operation(
                image,
                MorphOperation.CLOSE,
                self.config.hand_kernel_shape,
                self.config.dilate_and_erode_kernel_radius,
            )
            image = apply_morpholocial_operation(
                image,
                MorphOperation.OPEN,
                self.config.hand_kernel_shape,
                self.config.dilate_and_erode_kernel_radius,
            )

        stopwatch.logandrestart("Applied morphological operations")

        # if self.config.hand_contour_threshold_mode in [ThresholdMode.MANUAL, ThresholdMode.OTSU]:
        #     # Fill the small white areas of the image with black
        #     fill_contours(image, find_contours(image), 0, self.config.max_area_to_fill)

        return image
