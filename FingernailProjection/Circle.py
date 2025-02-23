from hand_contour_detector import dataclass
from image_display import dataclass
from matlike_utils import dataclass


@dataclass
class Circle:
    center: tuple[float, float] = (0, 0)
    radius: float = 10
