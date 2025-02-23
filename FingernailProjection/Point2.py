import math
from dataclasses import dataclass
from shapely.geometry import LineString, Point


@dataclass
class Vector2:
    x: float
    y: float

    @property
    @staticmethod
    def zero() -> "Vector2":
        return Vector2(0, 0)

    UP: "Vector2" = None
    RIGHT: "Vector2" = None

    @staticmethod
    def from_tuple(tup: tuple[float, float]) -> "Vector2":
        return Vector2(tup[0], tup[1])

    def tupled(self) -> tuple[float, float]:
        return self.x, self.y

    def tupled_int(self) -> tuple[int, int]:
        return int(self.x), int(self.y)

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2":
        return Vector2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector2":
        return self * scalar

    def __truediv__(self, scalar: float) -> "Vector2":
        return Vector2(self.x / scalar, self.y / scalar)

    def dot(self, other: "Vector2") -> float:
        return self.x * other.x + self.y * other.y

    @property
    def orientation(self) -> float:
        return math.atan2(self.y, self.x)

    @property
    def orientation_degrees(self) -> float:
        return math.degrees(self.orientation)

    @property
    def length_squared(self) -> float:
        return self.x**2 + self.y**2

    @property
    def length(self) -> float:
        return math.sqrt(self.length_squared)

    def with_length(self, new_length: float) -> "Vector2":
        scale = new_length / self.length
        return Vector2(self.x * scale, self.y * scale)

    def distance(self, other: "Vector2") -> float:
        return (self - other).length

    def __repr__(self):
        return f"Point2({self.x}, {self.y})"

    def topoint(self) -> Point:
        return Point(self.x, self.y)


Vector2.UP = Vector2(1, 0)
Vector2.RIGHT = Vector2(0, 1)
