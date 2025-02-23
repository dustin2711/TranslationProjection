from dataclasses import dataclass
import math

from shapely import Point, Polygon
from enums import *
from drawing import rect

from typing import List, NamedTuple, Tuple


@dataclass
class QuadBoundingBox:
    """A bounding box defined by four arbitrary corner points."""

    points: List[Tuple[int, int]]
    _polygon: Polygon = None

    def __post_init__(self):
        if len(self.points) != 4:
            raise ValueError("QuadBoundingBox must be initialized with exactly 4 points.")

    @property
    def area(self) -> int:
        return self.polygon.area

    @property
    def polygon(self) -> Polygon:
        if not self._polygon:
            self._polygon = Polygon(self.points)
        return self._polygon

    def distance_to_point(self, position: Tuple[int, int]) -> float:
        return self.polygon.distance(Point(position))

    @property
    def center(self) -> Tuple[int, int]:
        return (
            self.polygon.centroid.x,
            self.polygon.centroid.y,
        )

    def __hash__(self):
        return hash((self.points[0], self.points[1], self.points[2], self.points[3]))


@dataclass
class AxisBoundingBox:
    xmin: int
    xmax: int
    ymin: int
    ymax: int

    def add_offset(self, offset: Tuple[int, int]):
        self.xmin += offset[0]
        self.xmax += offset[0]
        self.ymin += offset[1]
        self.ymax += offset[1]

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def center(self) -> Tuple[int, int]:
        return (self.xmax + self.xmin) // 2, (self.ymax + self.ymin) // 2

    @property
    def top_left(self) -> Tuple[int, int]:
        return (self.xmin, self.ymin)

    @property
    def bottom_right(self) -> Tuple[int, int]:
        return (self.xmax, self.ymax)

    def get_corner_position(self, anchor: Anchor) -> Tuple[int, int]:
        """Get the position of the bounding box based on the given anchor."""
        if anchor == Anchor.CENTER:
            return self.center
        elif anchor == Anchor.TOP:
            return (self.center[0], self.ymin)
        elif anchor == Anchor.BOTTOM:
            return (self.center[0], self.ymax)
        elif anchor == Anchor.LEFT:
            return (self.xmin, self.center[1])
        elif anchor == Anchor.RIGHT:
            return (self.xmax, self.center[1])
        elif anchor == Anchor.TOP_LEFT:
            return self.top_left
        elif anchor == Anchor.TOP_RIGHT:
            return (self.xmax, self.ymin)
        elif anchor == Anchor.BOTTOM_LEFT:
            return (self.xmin, self.ymax)
        elif anchor == Anchor.BOTTOM_RIGHT:
            return self.bottom_right
        else:
            raise ValueError(f"Unsupported anchor: {anchor}")

    def distance_to_point(self, point: tuple[int, int]) -> float:
        px, py = point

        # Calculate the horizontal distance
        if px < self.xmin:
            x_dist = self.xmin - px
        elif px > self.xmax:
            x_dist = px - self.xmax
        else:
            x_dist = 0

        # Calculate the vertical distance
        if py < self.ymin:
            y_dist = self.ymin - py
        elif py > self.ymax:
            y_dist = py - self.ymax
        else:
            y_dist = 0

        # Return Euclidean distance
        return math.sqrt(x_dist**2 + y_dist**2)

    def distance_to(self, other: "AxisBoundingBox") -> float:
        # Calculate the horizontal distance
        if self.xmax < other.xmin:
            x_dist = other.xmin - self.xmax
        elif other.xmax < self.xmin:
            x_dist = self.xmin - other.xmax
        else:
            x_dist = -min(self.xmax - other.xmin, other.xmax - self.xmin)

        # Calculate the vertical distance
        if self.ymax < other.ymin:
            y_dist = other.ymin - self.ymax
        elif other.ymax < self.ymin:
            y_dist = self.ymin - other.ymax
        else:
            y_dist = -min(self.ymax - other.ymin, other.ymax - self.ymin)

        # If both distances are negative, boxes are overlapping; return the max distance between them
        if x_dist < 0 and y_dist < 0:
            return max(x_dist, y_dist)

        # Otherwise, return Euclidean distance
        return math.sqrt(x_dist**2 + y_dist**2)

    def draw(self, image, color=(0, 0, 0), thickness=2):
        rect(image, self.top_left, self.bottom_right, color, thickness)

    def get_intersection_area_with(self, other: "AxisBoundingBox") -> int:
        # Determine the overlap coordinates
        x_left = max(self.xmin, other.xmin)
        y_top = max(self.ymin, other.ymin)
        x_right = min(self.xmax, other.xmax)
        y_bottom = min(self.ymax, other.ymax)

        # Check if there's an intersection
        if x_right < x_left or y_bottom < y_top:
            return 0  # No intersection

        # Calculate intersection area
        intersection_width = x_right - x_left
        intersection_height = y_bottom - y_top
        return intersection_width * intersection_height

    @staticmethod
    def create_from_size(x_min: int, y_min: int, width: int, height: int) -> "AxisBoundingBox":
        return AxisBoundingBox(x_min, x_min + width, y_min, y_min + height)

    @staticmethod
    def create_from_points(points: list[Tuple[int, int]]) -> "AxisBoundingBox":
        # Initialize min and max values to the first point
        x_min, y_min = points[0]
        x_max, y_max = points[0]

        # Iterate over the points once to determine min and max values
        for x, y in points:
            if x < x_min:
                x_min = x
            if y < y_min:
                y_min = y
            if x > x_max:
                x_max = x
            if y > y_max:
                y_max = y

        return AxisBoundingBox(x_min, x_max, y_min, y_max)

    @staticmethod
    def create_from_ordered_points(
        topleft: Tuple[int, int],
        topright: Tuple[int, int],
        bottomright: Tuple[int, int],
        bottomleft: Tuple[int, int],
    ) -> "AxisBoundingBox":
        return AxisBoundingBox(
            min(topleft[0], bottomleft[0]),
            max(topright[0], bottomright[0]),
            min(topleft[1], topright[1]),
            max(bottomleft[1], bottomright[1]),
        )

    @staticmethod
    def create_from_corners(
        topleft: Tuple[float, float],
        topright: Tuple[float, float],
        bottomright: Tuple[float, float],
        bottomleft: Tuple[float, float],
    ) -> "AxisBoundingBox":
        return AxisBoundingBox(
            int(min(topleft[0], bottomleft[0])),
            int(max(topright[0], bottomright[0])),
            int(min(topleft[1], topright[1])),
            int(max(bottomleft[1], bottomright[1])),
        )

    @staticmethod
    def create_merged(box1: "AxisBoundingBox", box2: "AxisBoundingBox") -> "AxisBoundingBox":
        xmin = min(box1.xmin, box2.xmin)
        ymin = min(box1.ymin, box2.ymin)
        xmax = max(box1.xmax, box2.xmax)
        ymax = max(box1.ymax, box2.ymax)
        return AxisBoundingBox(xmin, xmax, ymin, ymax)

    def get_relation(self, other: "AxisBoundingBox") -> tuple[float, float]:
        """Calculates the relationship (distance or overlap) along x and y axes.

        Returns:
            A tuple (x_relation, y_relation), where:
            - Positive values represent the distance between the boxes.
            - Negative values represent the overlap extent along each axis.
        """
        return (
            AxisBoundingBox.range_distance(self.xmin, self.xmax, other.xmin, other.xmax),
            AxisBoundingBox.range_distance(self.ymin, self.ymax, other.ymin, other.ymax),
        )

    @staticmethod
    def range_distance(startA, endA, startB, endB) -> int:
        if endA < startB:  # Non-overlapping, range1 ends before range2 starts
            return startB - endA
        elif endB < startA:  # Non-overlapping, range2 ends before range1 starts
            return startA - endB
        else:  # Overlapping
            overlap = min(endA, endB) - max(startA, startB)
            return -overlap
