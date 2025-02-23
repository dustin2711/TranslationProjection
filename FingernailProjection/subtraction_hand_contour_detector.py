from hand_contour_detector import HandContourDetector
from matlike_utils import *
from skimage.feature import local_binary_pattern
from cv2.typing import MatLike
from stopwatch import Stopwatch

stopwatch = Stopwatch()


class SubtractionHandContourDetector(HandContourDetector):
    """Gets the hand contour by subtracting a reference image from the current image."""

    def __init__(self, config: Config):
        super().__init__(config)

    def threshold_image(self, image: MatLike) -> MatLike:
        stopwatch.restart(self.config.log_subtraction_contour_detection)

        if self.config.difference_mode == DifferenceMode.NO_PREPROCESSING:
            # Get delta image by subtracting the reference image from the current image
            delta_image = cv2.absdiff(self.reference_image, image)

        elif self.config.difference_mode == DifferenceMode.CANNY:
            # Preprocess first
            image = apply_gaussian(image, self.config.canny_blurr_size)
            self.reference_image = apply_gaussian(
                self.reference_image, self.config.canny_blurr_size
            )
            edges_reference = cv2.Canny(self.reference_image, 0.2, 0.5)
            edges_current = cv2.Canny(image, 0.2, 0.5)

            edges_reference = cv2.Canny(
                self.reference_image, self.config.low_threshold, self.config.high_threshold
            )
            edges_current = cv2.Canny(image, self.config.low_threshold, self.config.high_threshold)

            delta_image = cv2.absdiff(edges_reference, edges_current)

        elif self.config.difference_mode == DifferenceMode.BINARY_PATTERN:
            lbp_reference = local_binary_pattern(
                cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY), P=8, R=1, method="uniform"
            )
            lbp_image = local_binary_pattern(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), P=8, R=1, method="uniform"
            )

            delta_image = cv2.absdiff(lbp_reference.astype("uint8"), lbp_image.astype("uint8"))

        stopwatch.logandrestart("Generated delta image")

        delta_image = ensure_grayscale(delta_image)

        stopwatch.logandrestart("Generated greay delta image")

        if self.config.binarization_threshold == -1:
            thresholded_delta_image = apply_otsu_threshold(delta_image)
        else:
            _, thresholded_delta_image = cv2.threshold(
                delta_image,
                self.config.binarization_threshold,
                255,
                cv2.THRESH_BINARY,
            )

        stopwatch.logandrestart("Applied threshold")

        if self.config.difference_mode == DifferenceMode.NO_PREPROCESSING:
            if self.config.opening_kernel_size > 0:
                thresholded_delta_image = cv2.morphologyEx(
                    thresholded_delta_image,
                    cv2.MORPH_OPEN,
                    get_kernel(self.config.opening_kernel_size),
                )

            if self.config.closing_kernel_size > 0:
                thresholded_delta_image = cv2.morphologyEx(
                    thresholded_delta_image,
                    cv2.MORPH_CLOSE,
                    get_kernel(self.config.closing_kernel_size),
                )
            stopwatch.logandrestart("Applied morphological operations")

        return ensure_grayscale(thresholded_delta_image)
