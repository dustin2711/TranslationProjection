from math import *
from typing import Tuple
import numpy as np

from numpy import rad2deg


class Tuple2(tuple):
    """Two-dimensional tuples of numbers (float or integer)."""

    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == 2:
            values = args[0]  # Tuple input case: Tuple2((x, y))
        elif len(args) == 2:
            values = args  # Separate arguments case: Tuple2(x, y)
        else:
            raise TypeError("Tuple2 must be initialized with (x, y) or ((x, y),)")
        return super().__new__(cls, values)

    def __add__(self, other: "Tuple2") -> "Tuple2":
        return Tuple2((self[0] + other[0], self[1] + other[1]))

    def __sub__(self, other: "Tuple2") -> "Tuple2":
        return Tuple2((self[0] - other[0], self[1] - other[1]))

    def __mul__(self, scale: float) -> "Tuple2":
        return Tuple2((self[0] * scale, self[1] * scale))

    def __truediv__(self, scale: float) -> "Tuple2":
        return Tuple2((self[0] / scale, self[1] / scale))

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    @property
    def length(self) -> float:
        """Returns the Euclidean length of the vector."""
        return sqrt(self[0] ** 2 + self[1] ** 2)

    @property
    def orientation_rad(self) -> float:
        """Returns the angle in radians from the x-axis."""
        return atan2(self[1], self[0])

    @property
    def orientation_degrees(self) -> float:
        """Returns the angle in degrees from the x-axis."""
        return degrees(self.orientation_rad)

    @property
    def as_int(self) -> "Tuple2":
        return Tuple2(int(self[0]), int(self[1]))

    def mean(self, other: "Tuple2") -> "Tuple2":
        return (self + other) * 0.5

    @property
    def normalized(self) -> "Tuple2":
        return self / self.length

    def move_in_direction(self, direction: "Tuple2", distance: float) -> "Tuple2":
        return self + direction.normalized * distance

    def rotate(vector: "Tuple2", degrees: float) -> "Tuple2":
        angle = radians(degrees)
        cos_a = cos(angle)
        sin_a = sin(angle)
        return Tuple2(
            vector[0] * cos_a - vector[1] * sin_a,
            vector[0] * sin_a + vector[1] * cos_a,
        )


def subtract(point1: Tuple[int, int], point2: Tuple[int, int]) -> Tuple[int, int]:
    return (point1[0] - point2[0], point1[1] - point2[1])


def multiply(point: Tuple[int, int], scale: float) -> Tuple[int, int]:
    return (point[0] * scale, point[1] * scale)


def divide(point: Tuple[int, int], scale: float) -> Tuple[int, int]:
    return (point[0] / scale, point[1] / scale)


def add(point1: Tuple[int, int], point2: Tuple[int, int]) -> Tuple[int, int]:
    return (point1[0] + point2[0], point1[1] + point2[1])


def get_orientation_rad(point1: Tuple[int, int]) -> Tuple[int, int]:
    return atan2(point1[1], point1[0])


def get_orientation_degrees(point1: Tuple[int, int]) -> Tuple[int, int]:
    return rad2deg(atan2(point1[1], point1[0]))


def length(point1: Tuple[int, int]) -> float:
    return sqrt(point1[0] * point1[0] + point1[1] * point1[1])


def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    return length(subtract(point1, point2))


def rotate_vector(vector: np.ndarray, degrees: float) -> np.ndarray:
    radians = np.radians(degrees)
    rotation_matrix = np.array(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
    )
    return np.dot(rotation_matrix, vector)


def get_mean_position_and_deviation(
    positions: list[tuple[float, float]]
) -> tuple[tuple[float, float], float]:
    """Returns mean and deviation of the list of positions."""
    # Calculate the mean position
    mean_position = (
        np.mean([pos[0] for pos in positions]),
        np.mean([pos[1] for pos in positions]),
    )

    # Calculate the distance from each position to the mean position
    distances = [euclidean_distance(it, mean_position) for it in positions]

    return mean_position, np.mean(distances)


def to_int_tuple(position: tuple[float, float]):
    return (int(position[0]), int(position[1]))


def mean(point1: Tuple[int, int], point2: Tuple[int, int]) -> Tuple[int, int]:
    return (0.5 * (point1[0] + point2[0]), 0.5 * (point1[1] + point2[1]))


def normalize(direction: Tuple[int, int]) -> Tuple[int, int]:
    magnitude = length(direction)
    return (direction[0] / magnitude, direction[1] / magnitude)


def move_in_direction(
    position: tuple[float, float], direction: tuple[float, float], distance: float
) -> tuple[float, float]:
    normalized_direction = normalize(direction)
    new_position = (
        position[0] + normalized_direction[0] * distance,
        position[1] + normalized_direction[1] * distance,
    )

    return new_position


def is_inbetween(point, a, b):
    """Returns true if the point is inside the box spaned a and b."""
    return min(a[0], b[0]) <= point[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= point[1] <= max(
        a[1], b[1]
    )


def is_point_behind_line(
    point: Tuple[float, float],
    point_on_line: Tuple[float, float],
    behind_defining_direction: Tuple[float, float],
) -> bool:
    """
    Checks if a point is behind a line defined by a point on the line and a normal vector using NumPy.
    """
    vector = np.array(point) - np.array(point_on_line)
    normal_vec = np.array(behind_defining_direction)
    dot_product = np.dot(vector, normal_vec)
    return dot_product > 0
