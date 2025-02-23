import cv2
from setting_models import Enum
from enum import Enum


class CursorMode(Enum):
    Mouse = 0
    Finger = 1


class Anchor(Enum):
    CENTER = 0
    TOP = 1
    BOTTOM = 2
    LEFT = 4
    RIGHT = 8
    TOP_LEFT = TOP | LEFT
    TOP_RIGHT = TOP | RIGHT
    BOTTOM_LEFT = BOTTOM | LEFT
    BOTTOM_RIGHT = BOTTOM | RIGHT

    @staticmethod
    def get_opposite_anchor(anchor: "Anchor") -> "Anchor":
        """Return the opposite anchor of the given anchor."""
        opposite_map = {
            Anchor.CENTER: Anchor.CENTER,
            Anchor.TOP: Anchor.BOTTOM,
            Anchor.BOTTOM: Anchor.TOP,
            Anchor.LEFT: Anchor.RIGHT,
            Anchor.RIGHT: Anchor.LEFT,
            Anchor.TOP_LEFT: Anchor.BOTTOM_RIGHT,
            Anchor.TOP_RIGHT: Anchor.BOTTOM_LEFT,
            Anchor.BOTTOM_LEFT: Anchor.TOP_RIGHT,
            Anchor.BOTTOM_RIGHT: Anchor.TOP_LEFT,
        }

        return opposite_map.get(anchor, None)

    @staticmethod
    def calculate_offset(anchor: "Anchor", width: int, height: int) -> tuple[int, int]:
        factors = {
            Anchor.CENTER: (0.5, 0.5),
            Anchor.TOP: (0.5, 0),
            Anchor.BOTTOM: (0.5, 1),
            Anchor.LEFT: (0, 0.5),
            Anchor.RIGHT: (1, 0.5),
            Anchor.TOP_LEFT: (0, 0),
            Anchor.TOP_RIGHT: (1, 0),
            Anchor.BOTTOM_LEFT: (0, 1),
            Anchor.BOTTOM_RIGHT: (1, 1),
        }
        factor_x, factor_y = factors[anchor]
        return int(factor_x * width), int(factor_y * height)


class ProjectionTarget(Enum):
    FINGERNAIL = 0
    FIRST_JOINT = 1


class ProjectionTarget(Enum):
    FINGER = 0
    HAND = 1
    PAPER = 2


class TranslationDirection(Enum):
    JAPANESE_TO_ENGLISH = 0
    ENGLISH_TO_JAPANESE = 1


class TextflowMode(Enum):
    VERTICAL_KANJI = 0
    HORIZONTAL = 1
    VERTICAL = 2


class Orientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class ScaleMode(Enum):
    NATIVE = 0
    DOUBLE = 1
    QUADRUPLE = 2
    FILL_SCREEN_HEIGHT = 3
    FILL_HALF_SCREEN_HEIGHT = 4
    FILL = 5
    FILL_SCREEN_WIDTH = 6


class ImageRetrievingMode(Enum):
    FROM_FILE = 0
    CAMERA = 1


class ContourSelectionMode(Enum):
    LONGEST = 0
    TOP_LEFT = 1


class HandCountourDetectionMode(Enum):
    """How is the contour of the hand detected?"""

    THRESHOLD = 0
    SUBTRACTION = 1
    SKINCOLOR = 3
    # ADAPTIVE_SUBTRACTION = 2


class FingertipExtractionMode(Enum):
    TOP_LEFT_OF_COUNTOUR = 0
    MEDIAPIPE = 1


class DifferenceMode(Enum):
    NO_PREPROCESSING = 0
    CANNY = 1
    # BINARY_PATTERN = 2


class ContourMode(Enum):
    TREE = cv2.RETR_TREE
    EXTERNAL = cv2.RETR_EXTERNAL


class MorphOperation(Enum):
    ERODE = cv2.MORPH_ERODE
    DILATE = cv2.MORPH_DILATE
    """
    EXPAND BLACK areas and shrinks them back (Erosion => Dilation).
    Small structures will be destroyed but bigger ones will stay.
    """
    OPEN = cv2.MORPH_OPEN
    """
    EXPAND WHITE areas and shrinks them back (Erosion => Dilation).
    """
    CLOSE = cv2.MORPH_CLOSE
    GRADIENT = cv2.MORPH_GRADIENT
    TOPHAT = cv2.MORPH_TOPHAT
    BLACKHAT = cv2.MORPH_BLACKHAT
    HITMISS = cv2.MORPH_HITMISS  # Only works on binary images


class BackgroundSubtractorMode(Enum):
    MOG2 = 0
    KNN = 1


class KernelShape(Enum):
    RECT = cv2.MORPH_RECT
    ELLIPSE = cv2.MORPH_ELLIPSE
    CROSS = cv2.MORPH_CROSS


class ThresholdMode(Enum):
    ADAPTIVE_THRESHOLD = 0
    OTSU = 1


class OcrEngine(Enum):
    TESSERACT = 0
    PADDLE = 1
    MANGA = 2


class ThresholdMode(Enum):
    # NONE = 0
    MANUAL = 3
    OTSU = 2
    ADAPTIVE = 1


class Rotation(Enum):
    Rot0 = 0
    Rot90 = 1
    Rot180 = 2
    Rot270 = 3
