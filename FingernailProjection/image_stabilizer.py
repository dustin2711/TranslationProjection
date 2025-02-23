from enums import *
from config import Config
from drawing import cv2, np
import cv2
from matlike_utils import *
from cv2.typing import MatLike
from drawing import *
from stopwatch import Stopwatch

stopwatch = Stopwatch()


class ImageStabilizer:

    def __init__(self, config: Config):
        self.config = config
        self.reference_image = None
        self.display = ImageDisplay(
            "ImageStabilizer",
            Anchor.TOP_RIGHT,
            self.config.debug_screen_index,
            ScaleMode.FILL_HALF_SCREEN_HEIGHT,
        )

    @staticmethod
    def _draw_keypoints(
        image_to_draw: MatLike, keypoints, color: tuple = (0, 255, 0), radius: int = 3
    ) -> list[Callable]:
        """Local func to draw all the keypoints."""
        for keypoint in keypoints:
            cv2.circle(image_to_draw, tuple(map(int, keypoint.pt)), radius, color, -1)

    def stabilize(self, image: MatLike) -> MatLike:
        """Stabilizes the given image to match the reference image."""
        stopwatch.logging_enabled = self.config.log_stabilization
        stopwatch.restart()

        # Ensure reference image is set
        if self.reference_image is None or not are_tuple_lengths_equal(image, self.reference_image):
            self.reference_image = image.copy()
            stopwatch.logandrestart("Set reference image")

        # Use CUDA
        # gpu_image = cv2.cuda_GpuMat()
        # gpu_image.upload(image)
        # gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, -1, (5, 5), 1.5)
        # blurred_gpu = gaussian_filter.apply(gpu_image)
        # orb_cuda = cv2.cuda_ORB_create(nfeatures=500)
        # keypoints_gpu, descriptors_gpu = orb_cuda.detectAndComputeAsync(blurred_gpu, None)
        # keypoints_current = orb_cuda.convert(keypoints_gpu)
        # descriptor_current = descriptors_gpu.download()
        # stopwatch.logandstart("Get keypoints cuda")

        # Feature detection and description
        orb = cv2.ORB_create(
            nfeatures=self.config.nfeatures,
            scaleFactor=self.config.scale_factor,
            fastThreshold=self.config.fast_threshold,
        )  # ORB = Oriented FAST and Rotated BRIEF
        keypoints_reference, descriptor_reference = orb.detectAndCompute(
            apply_gaussian(self.reference_image, self.config.blurr_size), None
        )
        keypoints_current, descriptor_current = orb.detectAndCompute(
            apply_gaussian(image, self.config.blurr_size), None
        )

        stopwatch.logandrestart("Get keypoints reference")

        if descriptor_reference is None or descriptor_current is None:
            print("No features found.")
            return image

        # Match features
        brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = brute_force_matcher.match(descriptor_reference, descriptor_current)
        matches = sorted(matches, key=lambda it: it.distance)

        stopwatch.logandrestart("Get matches")

        source_points = np.float32(
            [keypoints_reference[match.queryIdx].pt for match in matches]
        ).reshape(-1, 1, 2)
        target_points = np.float32(
            [keypoints_current[match.trainIdx].pt for match in matches]
        ).reshape(-1, 1, 2)

        stopwatch.logandrestart("Get source and target points")

        if self.config.show_stabilization:
            reference_copy = ensure_rgb(self.reference_image.copy())
            ImageStabilizer._draw_keypoints(reference_copy, keypoints_reference)

            current_copy = ensure_rgb(image.copy())
            ImageStabilizer._draw_keypoints(current_copy, keypoints_current)

            self.display.show([reference_copy, current_copy])
            stopwatch.logandrestart("Showing stabilization images")

        shape = (self.reference_image.shape[1], self.reference_image.shape[0])

        transform, _ = (
            cv2.estimateAffinePartial2D(target_points, source_points, method=cv2.RANSAC)
            if self.config.use_affine
            else cv2.findHomography(
                target_points, source_points, cv2.RANSAC, self.config.ransac_value
            )
        )

        stopwatch.logandrestart("Found stabilization transform")

        if transform is None:
            print("Transform could not be determined.")
            return image

        return_image = (
            cv2.warpAffine(image, transform, shape)
            if self.config.use_affine
            else cv2.warpPerspective(image, transform, shape)
        )

        stopwatch.logandrestart("Found stabilization transform")

        return return_image
