import math
from tkinter import Image
import numpy as np
import cv2

from cv2.typing import MatLike
from enums import Anchor
from helper import *
from shapely.geometry import LineString, Point
from drawing import *
from typing import List, Tuple
from PIL.ImageFont import FreeTypeFont
from PIL import Image, ImageDraw
from tuple_helper import *
from PIL import ImageFont

# Colors
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
DARKRED = (0, 0, 64)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARKGRAY = (50, 50, 50)
LIGHTGRAY = (192, 192, 192)
DARKWHITE = (240, 240, 240)
LIGHTBLUE = (255, 128, 128)
LIGHTGREEN = (128, 255, 128)
LIGHTRED = (128, 128, 255)
PALEBLUE = (255, 192, 192)
PALEGREEN = (192, 255, 192)
PALERED = (192, 192, 255)
ALMOSTBLACK = (20, 20, 20)
BLACK = (0, 0, 0)

font_by_size = {}


def get_font(size: int, font_path: str) -> FreeTypeFont:
    if not (font := font_by_size.get(size)):
        font = ImageFont.truetype(
            font_path,
            size,
        )
        font_by_size[size] = font
    return font


def create_gray(value: int):
    """Returns (value, value, value)."""
    return (value, value, value)


def interpolate_color(
    color1: Tuple[int, int, int], color2: Tuple[int, int, int], scalar: float
) -> Tuple[int, int, int]:
    """Interpolate between two colors with a scalar from 0 to 1."""
    r = int(color1[0] + (color2[0] - color1[0]) * scalar)
    g = int(color1[1] + (color2[1] - color1[1]) * scalar)
    b = int(color1[2] + (color2[2] - color1[2]) * scalar)
    return (r, g, b)


def draw_linestring(
    image: MatLike, line: LineString, color=BLUE, thickness=1, offset: tuple[int, int] = (0, 0)
):
    if thickness == 0:
        return

    # Extract the coordinates from the LineString and iterate over consecutive points
    for start, end in enumerate_pairwise(line.coords):
        # Draw the line segment between the current and the next point
        cv2.line(
            image,
            add(offset, (int(start[0]), int(start[1]))),
            add(offset, (int(end[0]), int(end[1]))),
            color,
            thickness,
        )


def draw_line(image: MatLike, start: Tuple[int, int], end: Tuple[int, int], color=RED, thickness=1):
    cv2.line(image, start, end, color, thickness)


def point(image: MatLike, point: tuple[int, int] | Point, color=RED, radius=3):
    if isinstance(point, Point):
        point = (int(point.x), int(point.y))
    elif isinstance(point, tuple):
        point = (int(point[0]), int(point[1]))  # Make sure its int
    cv2.circle(image, point, radius, color, -1)  # -1 fill the circle


def draw_circle(image: MatLike, point: tuple[int, int] | Point, radius, color=RED, thickness=1):
    if isinstance(point, Point):
        point = (int(point.x), int(point.y))
    elif isinstance(point, tuple):
        point = (int(point[0]), int(point[1]))
    cv2.circle(image, point, int(radius), color, thickness)  # -1 fill the circle


def draw_half_circle(
    image,
    center: tuple[int, int],
    radius: float,
    start_degrees=0,
    end_degrees=180,
    color: tuple[int, int, int] = RED,
    thickness=1,
):
    cv2.ellipse(
        image, center, (radius, radius), 0, start_degrees, end_degrees, color, thickness=thickness
    )


def draw_contours(image: MatLike, contours: list[np.ndarray], color=RED, thickness=1):
    cv2.drawContours(image, contours, -1, color, thickness)


def draw_polygon(
    image: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int] = BLACK,
    thickness: int = 1,
    is_closed: bool = True,
):
    """
    Draws a polygon on the given image.
    """

    if thickness == 0:
        return

    # Convert points to a NumPy array and reshape for OpenCV
    points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    # Draw the polygon
    cv2.polylines(image, [points], isClosed=is_closed, color=color, thickness=thickness)


def rect(
    image: MatLike,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color=BLACK,
    thickness=1,
    margin=0,
):
    if thickness <= 0:
        return

    if margin != 0:
        start = (start[0] - margin, start[1] + margin)
        end = (end[0] + margin, end[1] - margin)

    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))
    cv2.rectangle(image, start, end, color, thickness)


def draw_rect_with_anchor(
    image: MatLike,
    anchor_point: Tuple[int, int],
    rect_size: Tuple[int, int],
    anchor: Anchor = Anchor.TOP_LEFT,
    color=BLACK,
    thickness=1,
    margin=0,
):
    """
    Draws a rectangle on the image using the given anchor point and anchor alignment.

    Args:
        image (MatLike): The image matrix.
        anchor_point (Tuple[int, int]): The reference point for the rectangle.
        rect_size (Tuple[int, int]): Width and height of the rectangle.
        anchor (Anchor): Anchor alignment for the rectangle (e.g., top-left, center).
        color: Rectangle color.
        thickness (int): Thickness of the rectangle lines.
        margin (int): Extra margin to expand the rectangle.

    Returns:
        None
    """
    x_ref, y_ref = anchor_point
    width, height = rect_size

    # Adjust the center based on the anchor
    match anchor.value:
        case Anchor.CENTER.value:
            center = (x_ref, y_ref)
        case Anchor.TOP.value:
            center = (x_ref, y_ref + height // 2)
        case Anchor.BOTTOM.value:
            center = (x_ref, y_ref - height // 2)
        case Anchor.LEFT.value:
            center = (x_ref + width // 2, y_ref)
        case Anchor.RIGHT.value:
            center = (x_ref - width // 2, y_ref)
        case Anchor.TOP_LEFT.value:
            center = (x_ref + width // 2, y_ref + height // 2)
        case Anchor.TOP_RIGHT.value:
            center = (x_ref - width // 2, y_ref + height // 2)
        case Anchor.BOTTOM_LEFT.value:
            center = (x_ref + width // 2, y_ref - height // 2)
        case Anchor.BOTTOM_RIGHT.value:
            center = (x_ref - width // 2, y_ref - height // 2)
        case _:
            raise ValueError("Invalid anchor value")

    # Calculate the top-left and bottom-right corners
    x_start = int(center[0] - width // 2)
    y_start = int(center[1] - height // 2)
    x_end = x_start + width
    y_end = y_start + height

    # Add margins if specified
    if margin != 0:
        x_start -= margin
        y_start -= margin
        x_end += margin
        y_end += margin

    # Ensure the coordinates are integers and within bounds
    x_start = max(0, int(x_start))
    y_start = max(0, int(y_start))
    x_end = min(image.shape[1], int(x_end))
    y_end = min(image.shape[0], int(y_end))

    # Draw the rectangle
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color, thickness)


def draw_text(
    image: MatLike,
    text: str,
    position: tuple[int, int],
    font: FreeTypeFont,
    anchor: Anchor = Anchor.TOP_LEFT,
    color: Tuple[int, int, int] = (0, 0, 0),
    background_color: Tuple[int, int, int, int] = (0, 0, 0, 0),  # Default: transparent
) -> None:
    """
    Draws text on the image with optional background color and rotation.
    """

    if font.size == 0:
        return

    # Descent is the distance of the lower text part of letters like y, g, p
    descent = font.getmetrics()[1]

    # Measure the text size
    # left, top, right, bottom = font.getbbox(text)
    left, top, right, bottom = font.getbbox(text, anchor="ls")  # 'ls' anchors to the baseline
    text_width = right - left
    text_height = bottom - top + descent
    # text_height = bottom - top

    # Adjust x and y based on the anchor
    x, y = position
    if anchor in [Anchor.CENTER, Anchor.TOP, Anchor.BOTTOM]:
        x -= text_width // 2
    if anchor in [Anchor.RIGHT, Anchor.TOP_RIGHT, Anchor.BOTTOM_RIGHT]:
        x -= text_width
    if anchor in [Anchor.CENTER, Anchor.LEFT, Anchor.RIGHT]:
        y -= text_height // 2
    if anchor in [Anchor.BOTTOM, Anchor.BOTTOM_LEFT, Anchor.BOTTOM_RIGHT]:
        y -= text_height

    # Ensure the overlay stays within the bounds of the image
    image_height, image_width = image.shape[:2]
    x = max(0, x)
    y = max(0, y)

    # Adjust text dimensions to fit within the image
    if x + text_width > image_width:
        text_width = image_width - x
    if y + text_height > image_height:
        text_height = image_height - y - descent

    # Create a new PIL image with background color and text drawn upon
    text_image_pil = Image.new("RGBA", (text_width, text_height + descent), background_color)
    ImageDraw.Draw(text_image_pil).text((0, 0), text, font=font, fill=color)

    # Convert to OpenCV format
    text_image_cv = cv2.cvtColor(np.array(text_image_pil), cv2.COLOR_RGBA2BGRA)

    # Update dimensions after rotation
    text_height, text_width = text_image_cv.shape[:2]

    # Extract BGR and alpha channels from text_image
    text_part = text_image_cv[:, :, :3]  # BGR part
    alpha_part = text_image_cv[:, :, 3] / 255.0  # Alpha part (normalized)

    # Ensure the overlay stays within bounds after rotation
    x_end = min(x + text_width, image_width)
    y_end = min(y + text_height, image_height)

    overlay = image[y:y_end, x:x_end]

    for channel in range(3):  # Blend BGR channels
        overlay[:, :, channel] = (
            overlay[:, :, channel] * (1 - alpha_part[: y_end - y, : x_end - x])
            + text_part[: y_end - y, : x_end - x, channel] * alpha_part[: y_end - y, : x_end - x]
        )


def create_pillow_empty_image(size: tuple[int, int], color: tuple[int, int, int], format="RGB"):
    return Image.new(format, size, color)


def get_text_bounding_box(text, font) -> tuple[int, int, int, int]:
    # Calculate the precise bounding box of the text
    draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))  # Create a small dummy image
    return draw.textbbox((0, 0), text, font=font)  # Exact bounding box of the text


def create_pillow_vertical_text_image(
    text: str,
    font: ImageFont.FreeTypeFont,
    color=(0, 0, 0),
    format="RGBA",
    background_color=(0, 0, 0, 0),
    spacing_factor=0.1,
) -> Image.Image:
    spacing = int(font.size * spacing_factor)

    # ascent, descent = font.getmetrics()
    # char_height = ascent + descent

    char_count = len(text)
    char_sizes = [font.getbbox(char) for char in text]
    char_height = max([bottom - top for _, top, _, bottom in char_sizes])
    char_width = max([right - left for left, _, right, _ in char_sizes])

    total_height = char_height * char_count + (char_count - 1) * spacing

    text_image_pillow = Image.new(format, (char_width, total_height), background_color)
    draw = ImageDraw.Draw(text_image_pillow)

    y = -10
    for char in text:
        draw.text((0, y), char, font=font, fill=color)
        y += char_height + spacing

    return text_image_pillow


def create_pillow_text_image(
    text: str,
    font: FreeTypeFont,
    color: Tuple[int, int, int] = (0, 0, 0),
    format="RGBA",
    background_color: Tuple[int, int, int, int] = (0, 0, 0, 0),
    use_ascend_and_descend: bool = True,
) -> np.ndarray:
    """Multiline is unsupported if use_ascend_and_descend is true."""
    left, top, right, bottom = (
        font.getbbox(text) if use_ascend_and_descend else get_text_bounding_box(text, font)
    )
    if use_ascend_and_descend:
        if "\n" in text:
            raise ValueError("Multiline is unsupported if use_ascend_and_descend is true.")
        pos = (0, 0)
        # We use ascent and descent if "ace", "bdf" or "gyp" shall have all same height
        # Ascent = part above baseline, Descent = part below baseline where
        # Baseline = lower height of letters without unterground part like a, b, c, d, e, f
        ascent, descent = font.getmetrics()
        text_height = ascent + descent
    else:
        pos = (-left, -top)
        text_height = bottom - top

    text_width = right - left
    text_image_pillow = Image.new(format, (text_width, text_height), background_color)
    draw = ImageDraw.Draw(text_image_pillow)
    draw.text(pos, text, font=font, fill=color)  # Offset using bbox
    return text_image_pillow


import cv2
import numpy as np
from typing import Tuple


def overlay_image(
    background: np.ndarray,
    overlay: np.ndarray,
    position: Tuple[int, int],
    use_center_not_topleft: bool = True,
) -> np.ndarray:
    """
    Overlay one image onto another at a specified position with bounds checking and clipping.

    :param background: The background image (as a NumPy array).
    :param overlay: The overlay image (as a NumPy array, supports alpha channel).
    :param position: The top-left position (x, y) for the overlay on the background.
    :return: The resulting image with the overlay applied.
    """
    x, y = position

    if use_center_not_topleft:  # Adjust position to be the top-left based on the center
        x -= overlay.shape[1] // 2
        y -= overlay.shape[0] // 2

    bg_height, bg_width = background.shape[:2]
    overlay_height, overlay_width = overlay.shape[:2]

    # Calculate the region of the overlay that will be used
    x_start_overlay = max(0, -x)  # Skip left part of overlay if x is negative
    y_start_overlay = max(0, -y)  # Skip top part of overlay if y is negative
    x_end_overlay = min(overlay_width, bg_width - x)  # Clip to the right edge of the background
    y_end_overlay = min(overlay_height, bg_height - y)  # Clip to the bottom edge of the background

    # Calculate the corresponding region in the background
    x_start_bg = max(0, x)
    y_start_bg = max(0, y)
    x_end_bg = x_start_bg + (x_end_overlay - x_start_overlay)
    y_end_bg = y_start_bg + (y_end_overlay - y_start_overlay)

    # Extract the relevant part of the overlay
    overlay_clipped = overlay[y_start_overlay:y_end_overlay, x_start_overlay:x_end_overlay]

    # Separate the alpha channel if present
    if overlay_clipped.shape[2] == 4:
        alpha = overlay_clipped[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
        overlay_rgb = overlay_clipped[:, :, :3]
    else:
        alpha = np.ones((y_end_overlay - y_start_overlay, x_end_overlay - x_start_overlay))
        overlay_rgb = overlay_clipped

    # Extract the region of interest (ROI) from the background
    roi = background[y_start_bg:y_end_bg, x_start_bg:x_end_bg]

    # Blend the overlay with the background using alpha blending
    for channel in range(3):  # Loop through BGR channels
        roi[:, :, channel] = roi[:, :, channel] * (1 - alpha) + overlay_rgb[:, :, channel] * alpha

    # Put the modified ROI back into the background
    background[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = roi

    return background


def create_roated_text(
    text: str,
    degrees: float,
    anchor: Anchor,
    font: FreeTypeFont,
    color: Tuple[int, int, int] = (0, 0, 0),
    background_color: Tuple[int, int, int, int] = (0, 0, 0, 0),  # Default: transparent
    use_ascend_and_descend=True,
) -> MatLike:
    """Creates a text image with the given rotation and anchor.
    The text start (and anchor) is in the center."""
    text_image = create_pillow_text_image(
        text,
        font,
        color,
        background_color=background_color,
        use_ascend_and_descend=use_ascend_and_descend,
    )
    width, height = text_image.size
    center = int(math.sqrt(width**2 + height**2))
    image_size = 2 * center
    image = Image.new("RGBA", (image_size, image_size), background_color)
    offset = Anchor.calculate_offset(anchor, width, height)
    image.paste(text_image, (center - offset[0], center - offset[1]))
    image = image.rotate(degrees, resample=Image.BICUBIC, expand=False)

    return image


def draw_roated_text(
    image_to_overlay: MatLike,
    text: str,
    position: tuple[int, int],
    degrees: float,
    anchor: Anchor,
    font: FreeTypeFont,
    color: Tuple[int, int, int] = (0, 0, 0),
    background_color: Tuple[int, int, int, int] = (0, 0, 0, 0),  # Default: transparent
    use_ascend_and_descend=True,
) -> None:
    text_image = create_roated_text(
        text, degrees, anchor, font, color, background_color, use_ascend_and_descend
    )

    overlay_image(image_to_overlay, np.array(text_image), position, use_center_not_topleft=True)
