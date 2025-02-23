from typing import Callable, Optional
import numpy as np

from scipy.signal import savgol_filter
from shapely import MultiPoint
from shapely.geometry.base import BaseGeometry
from Point2 import Vector2
from drawing import LineString, Point
from helper import *
from shapely.geometry import LineString, Point
from drawing import *
from scipy.optimize import least_squares


def resample_linestring(line: LineString, spacing: float) -> LineString:
    """Resamples a LineString to ensure points are equidistant."""
    # Check if spacing is valid
    if spacing <= 0 or line.length == 0:
        return line

    # Generate equidistant distances along the LineString
    distances = np.arange(0, line.length, spacing)

    # Interpolate points along the LineString
    equidistant_points = [line.interpolate(distance) for distance in distances]

    # May close the new line too
    if line.is_closed:
        equidistant_points.append(equidistant_points[0])

    return LineString(equidistant_points)


def smooth_linestring(line: LineString, window_size=5) -> LineString:
    """
    Smooths a LineString by applying a moving average to its points,
    preserving the original number of points and ensuring continuity for rings.
    """
    # Ensure the line has enough points to apply smoothing
    if len(line.coords) < window_size or window_size <= 1:
        return line

    coords = np.array(line.coords)
    x = coords[:, 0]
    y = coords[:, 1]

    # Compute the padding size
    pad_size = window_size // 2

    # Handle rings (closed LineString)
    is_closed = line.is_closed
    if is_closed:
        # Extend x and y to wrap around for smoothing
        x = np.concatenate([x[-pad_size:], x, x[:pad_size]])
        y = np.concatenate([y[-pad_size:], y, y[:pad_size]])
    else:
        # Apply symmetric padding for non-rings
        x = np.pad(x, (pad_size, pad_size), mode="edge")
        y = np.pad(y, (pad_size, pad_size), mode="edge")

    # Apply moving average to x and y coordinates
    x_smooth = np.convolve(x, np.ones(window_size) / window_size, mode="valid")
    y_smooth = np.convolve(y, np.ones(window_size) / window_size, mode="valid")

    # For rings, drop the extra points added for wrapping
    if is_closed:
        x_smooth = x_smooth[pad_size:-pad_size]
        y_smooth = y_smooth[pad_size:-pad_size]

    # Create smoothed LineString
    smoothed_points = list(zip(x_smooth, y_smooth))
    if is_closed:
        smoothed_points.append(smoothed_points[0])  # Ensure it's closed

    return LineString(smoothed_points)


def simplify_linestring(linestring: LineString, simplification: int) -> LineString:
    """
    Reduces the number of points in a LineString based on the simplification level from 0 to 1 (highest).
    """
    if simplification != 0:
        # Maps simplification level to epsilon for LineString simplification
        epsilon = 10 ** (-6 + 3 * simplification)
        return linestring.simplify(epsilon, preserve_topology=False)
    else:
        return linestring


def get_upper_point_in_linestring(line: LineString) -> Point:
    """Returns the point in the LineString with the highest y-coordinate (height)."""
    highest_point = min(line.coords, key=lambda point: point[1])  # y-coordinate is at index 1
    return Point(highest_point)


# Took 60 ms for 4400 points with old approach filter_function(Point(point)) and 2.5 ms with new approach
def filter_linestring(
    line: LineString,
    filter_function: Callable[[Point], bool],
    close_result: Optional[bool] = None,
) -> LineString:
    """Filters points in one LineString based on a filter function that takes a Point and returns a boolean."""
    points = [point for point in line.coords if filter_function(point)]
    if len(points) > 0 and close_result is not None:
        # Close if stated
        if close_result and points[0] != points[-1]:
            points.append(points[0])
        # Open if stated
        if not close_result and points[0] == points[-1]:
            points.pop()

    return LineString(points) if len(points) >= 2 else LineString()


def filter_linestrings_far_from_point(
    lines: list[LineString], close_point: Vector2, max_distance: float
) -> list[LineString]:
    # Return only these lines where any point is close to the given point
    if max_distance == 0:
        return lines

    return [
        line
        for line in lines
        if any(
            (Vector2.from_tuple(point) - close_point).length < max_distance for point in line.coords
        )
    ]


def filter_points_close_to_point_at_index(
    line: LineString, index: int, max_distance: float
) -> LineString:
    """
    Filters points based on the distance from a point within the LineString defined by an index.
    Removes all points that are too far from the given point based on the defined arc distance along the LineString.
    """
    if max_distance == 0:
        return line

    distances = [line.project(Point(coord)) for coord in line.coords]
    reference_distance = distances[index]

    points = [
        point
        for point, distance in zip(line.coords, distances)
        if abs(distance - reference_distance) < max_distance
    ]

    if len(points) < 2:
        return line

    return LineString(points)


def get_farthest_point_in_linestring(line: LineString, reference_point: Vector2) -> Point:
    """Find the point in the LineString that is closest to the reference point"""
    if len(line.coords) == 0:
        return Point()

    closest_point = max(
        line.coords,
        key=lambda point: (Vector2.from_tuple(point) - reference_point).length,
    )
    return Point(closest_point)


def get_index_of_point_in_line_with_angle(linestring: LineString, angle: float) -> int:
    """Returns the index of the point in the LineString that is closest to the given angle."""
    window_length = 9
    orientations: list[float] = []
    centers: list[float] = []
    for start, end in enumerate_pairwise(linestring.coords):
        start = Vector2.from_tuple(start)
        end = Vector2.from_tuple(end)
        delta: Vector2 = end - start
        # Calculate the similarity of the line to the fingertip direction
        orientations.append(abs(delta.orientation_degrees - angle))
        centers.append((start + end) / 2)
    smoothed_orientations = (
        savgol_filter(orientations, window_length=window_length, polyorder=2)
        if window_length <= len(orientations)
        else orientations
    )
    # print_once(
    #     f"{join_string(orientations, converter=lambda x: f"{x:3.0f}")}->\n{join_string(smoothed_orientations, converter=lambda x: f"{x:3.0f}")}\n",
    #     "smooth",
    # )
    index_of_lowest_value = np.argmin(smoothed_orientations)
    return int(index_of_lowest_value)


def get_index_of_closest_point_in_line(line_string: LineString, close_point: Vector2) -> int:
    """Returns the index of the closest point in the LineString to the given point."""
    # Enumerate segments and calculate the distance from each segment to the given point
    min_distance = float("inf")
    closest_index = None

    for index, (start, end) in enumerate(enumerate_pairwise(line_string.coords)):
        segment = LineString([start, end])
        distance = segment.distance(close_point)

        # Update the closest segment index if the current distance is smaller
        if distance < min_distance:
            min_distance = distance
            closest_index = index

    return closest_index


def convert_contour_to_linestring(contour: np.ndarray, close: bool) -> LineString:
    if len(contour) < 2:
        return LineString()

    # Extract points from the contour
    points = [it[0] for it in contour]

    # Close the LineString if needed
    if close and not np.array_equal(points[0], points[-1]):
        points.append(points[0])

    return LineString(points)


def convert_contour_to_closed_linestring(contour: np.ndarray) -> LineString:
    if len(contour) >= 2:
        points = [tuple(it[0]) for it in contour]
        if points[0] != points[-1]:
            points.append(points[0])  # Close the loop
        return LineString(points)
    return LineString()


def find_half_circle(
    linestring, initial_guess_center: tuple[int, int], initial_guess_radius: int
) -> tuple[tuple[int, int], int]:
    """A orientation of 0 means that the half-circle faces upwards"""

    def residuals(params, points):
        x_c, y_c, r = params
        x, y = points[:, 0], points[:, 1]
        distances = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2) - r
        # Penalize points below the center (y > y_c)
        return np.where(y > y_c, distances**2, distances)

    points = np.array(linestring.coords)

    points = np.array(linestring.coords)
    result = least_squares(
        residuals,
        (initial_guess_center[0], initial_guess_center[1], initial_guess_radius),
        args=(points,),
    )
    x_center, y_center, radius = result.x
    return (x_center, y_center), radius


def get_first_intersection(line1: LineString, line2: LineString) -> Point | None:
    """Find first intersection of the two LineStrings"""
    intersection: BaseGeometry = line1.intersection(line2)

    # Process the intersection result
    if intersection.is_empty:
        return None  # No intersection
    elif isinstance(intersection, Point):
        return intersection  # Single point of intersection
    elif isinstance(intersection, LineString):
        return Point(intersection.coords[0])  # First coordinate of overlapping lines
    elif isinstance(intersection, MultiPoint):
        return Point(intersection.geoms[0])
    else:
        return None  # Catch-all for other cases (unlikely)
