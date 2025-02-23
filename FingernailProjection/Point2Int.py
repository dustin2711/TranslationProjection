from dataclasses import dataclass
import math


@dataclass
class Point2Int:
    x: int
    y: int

    @staticmethod
    def from_tuple(tup: tuple[int, int]) -> "Point2Int":
        return Point2Int(tup[0], tup[1])

    def tupled(self) -> tuple[int, int]:
        return self.x, self.y

    def __add__(self, other: "Point2Int") -> "Point2Int":
        return Point2Int(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point2Int") -> "Point2Int":
        return Point2Int(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: int) -> "Point2Int":
        return Point2Int(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: int) -> "Point2Int":
        return self * scalar

    def dot(self, other: "Point2Int") -> int:
        return self.x * other.x + self.y * other.y

    def with_length(self, new_length: float) -> "Point2Int":
        scale = new_length / math.sqrt(self.x**2 + self.y**2)
        return Point2Int(int(self.x * scale), int(self.y * scale))

    def __repr__(self):
        return f"Point2d({self.x}, {self.y})"
