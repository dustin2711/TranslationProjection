from abc import ABC, abstractmethod
from text_detection import TextDetection
from config import Config
from drawing import *
from matlike_utils import *
import cv2
from text_detection import TextDetection
from cv2.typing import MatLike
from shapely.geometry import LineString, Point
from image_display import ImageDisplay


class FingertipExtractor(ABC):

    def __init__(self, config: Config):
        self.config = config
        self.display = ImageDisplay(
            "FingertipExtractor",
            Anchor.BOTTOM_RIGHT,
            lambda: self.config.debug_screen_index,
            ScaleMode.FILL_HALF_SCREEN_HEIGHT,
        )

    def get_fingertip(
        self,
        image: MatLike,
        contour: LineString,
        detections: list[TextDetection],
    ) -> tuple[int, int]:
        """Get the fingertip position from the given image."""

        # Attempt to get the fingertip and base point
        if tip_and_base := self.get_fingertip_and_basepoint(image, contour, detections):
            fingertip, base = tip_and_base

        if self.config.show_fingertip_extraction:
            if tip_and_base:
                point(image, fingertip, RED, 5)
                point(image, base, RED, 5)
            self.display.show(image)

        if not tip_and_base:
            return None

        # Calculate direction as a numpy array
        direction = np.array(
            [
                fingertip[0] - base[0],
                fingertip[1] - base[1],
            ]
        )

        # Normalize the direction vector
        if magnitude := np.linalg.norm(direction):
            # Offset the fingertip point
            offset = self.config.offset_away_from_centroid * direction / magnitude
            return (fingertip[0] + offset[0], fingertip[1] + offset[1])
        else:
            print("Could not shift the fingertip away from the centroid.")
            return fingertip

    @abstractmethod
    def get_fingertip_and_basepoint(
        self,
        image: MatLike,
        contour: LineString,
        detections: list[TextDetection],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Extracts the fingertip position and the point of the hand/finger to move away from."""
        pass
