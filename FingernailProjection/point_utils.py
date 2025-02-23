from shapely.geometry import Point
from matlike_utils import *


def add(a: Point, b: Point) -> Point:
    return Point(a.x + b.x, a.y + b.y)


def with_length(point: Point, new_length: float) -> Point:
    scale = new_length / point.distance(Point(0, 0))
    return Point(point.x * scale, point.y * scale)


def get_point_average(point1: Point, point2: Point) -> Point:
    """Returns the midpoint between two Point objects."""
    avg_x = (point1.x + point2.x) / 2
    avg_y = (point1.y + point2.y) / 2
    return Point(avg_x, avg_y)
