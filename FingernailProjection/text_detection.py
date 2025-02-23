from typing import Optional
from enums import *
from axis_bounding_box import AxisBoundingBox, QuadBoundingBox


from dataclasses import dataclass, field
from drawing import *
from matlike_utils import draw_text
from cv2.typing import MatLike

from config import Config, make_tuple_color


@dataclass
class TextDetection:
    text: str
    boundingbox: QuadBoundingBox
    confidence: int
    parent: Optional["TextDetection"] = None
    children: list["TextDetection"] = field(default_factory=list)

    @property
    def child_index(self):
        """Returns the index in the parent.children list."""
        return self.parent.children.index(self)

    @property
    def text_index(self) -> int:
        """Returns the starting index of text inside the parent text."""
        predecessor = self.predecessor
        length = 0
        while predecessor != None:
            length += len(predecessor.text) + 1  # Add text length plus space
            predecessor = predecessor.predecessor
        return length

    @property
    def text_length(self) -> int:
        """Returns the length of text."""
        return len(self.text)

    @property
    def predecessor(self):
        """Returns the predecessor in the parent.children list."""
        previous_index = self.child_index - 1
        return self.parent.children[previous_index] if previous_index >= 0 else None

    @property
    def successor(self):
        """Returns the predecessor in the parent.children list."""
        next_index = self.child_index + 1
        return self.parent.children[next_index] if next_index >= len(self.parent.children) else None

    def __hash__(self):
        # Use a tuple of attributes that uniquely identify this object
        return hash((self.text, self.boundingbox, self.confidence))

    def __eq__(self, other):
        if isinstance(other, TextDetection):
            return (self.text, self.boundingbox, self.confidence) == (
                other.text,
                other.boundingbox,
                other.confidence,
            )
        return False

    @staticmethod
    def get_total_axis_bounding_box(detections: list["TextDetection"]) -> AxisBoundingBox:
        raise NotImplementedError()

    @staticmethod
    def draw_bounding_boxes(
        image: np.array,
        detections: list["TextDetection"],
        highlighted_detection: Optional["TextDetection"],
        config: Config,
        color: tuple[int, int, int],
    ):
        for detection in detections:
            thickness = (
                config.bounding_box_thickness
                if detection != highlighted_detection
                else config.bounding_box_thickness_on_hover
            )
            if thickness <= 0:
                continue

            draw_polygon(image, detection.boundingbox.points, color, thickness, is_closed=True)
