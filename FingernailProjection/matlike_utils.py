import math
from PIL.ImageFont import FreeTypeFont
from typing import Tuple
import cv2
import numpy as np

from cv2.typing import MatLike
from enums import cv2
from helper import *
from drawing import *
from image_display import *
from linestring_utils import *
from enums import *


class ScalingMode(Enum):
    Nearest = cv2.INTER_NEAREST
    """Nearest-neighbor interpolation."""

    Linear = cv2.INTER_LINEAR
    """Bilinear interpolation, suitable for upscaling."""

    Cubic = cv2.INTER_CUBIC
    """Bicubic interpolation, smooth results, good for upscaling."""

    Lanczos4 = cv2.INTER_LANCZOS4
    """Lanczos interpolation, high-quality but slower, ideal for downscaling."""

    Area = cv2.INTER_AREA
    """Pixel area relation, often used for downscaling."""

    LinearExtract = cv2.INTER_LINEAR_EXACT
    """Exact bilinear interpolation, more accurate than INTER_LINEAR."""

    NearestExact = cv2.INTER_NEAREST_EXACT
    """Exact nearest-neighbor interpolation, slightly more accurate."""


def kernel_size_from_radius(radius: int):
    return 1 + 2 * radius


def kernel_from_radius(radius: int):
    return (kernel_size_from_radius(radius), kernel_size_from_radius(radius))


def grayscale_to_rgb(grayscale_matrix: MatLike):
    """Creates a copy to not alter the original matrix and returns this as rgb."""
    return cv2.cvtColor(grayscale_matrix.copy(), cv2.COLOR_GRAY2RGB)


def scale_image(
    image: MatLike, scaling_factor: float, scaling_mode: ScalingMode = ScalingMode.Area
) -> MatLike:
    return cv2.resize(
        image,
        (int(image.shape[1] * scaling_factor), int(image.shape[0] * scaling_factor)),
        interpolation=scaling_mode.value,
    )


def scale_image_to(
    image: MatLike, target_size: tuple[float, float], scaling_mode: ScalingMode = ScalingMode.Area
) -> MatLike:
    return cv2.resize(
        image,
        target_size,
        interpolation=scaling_mode.value,
    )


def apply_gaussian(image: MatLike, kernel_radius: int, sigma: int = 0):
    if kernel_radius == 0:
        return image

    return cv2.GaussianBlur(
        image,
        kernel_from_radius(kernel_radius),
        sigmaX=sigma,
        sigmaY=sigma,
    )


def apply_adaptive_threshold(image: MatLike, kernel_radius: int, threshold_offset: int):
    if kernel_radius == 0:
        return image

    return cv2.adaptiveThreshold(
        image,
        255,  # This value will be assigned to the pixel if the condition is met
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        kernel_size_from_radius(kernel_radius),
        threshold_offset,
    )


def apply_threshold(image: MatLike, value: int):
    _, image = cv2.threshold(
        image,
        value,  # This value is ignored as Otsu thresholding calculcates it from the histogram
        255,  # This value will be assigned to the pixel if the condition is met
        cv2.THRESH_BINARY,
    )
    return image


def apply_otsu_threshold(image: MatLike):
    image = ensure_grayscale(image)

    _, image = cv2.threshold(
        image,
        0,  # This value is ignored as Otsu thresholding calculcates it from the histogram
        255,  # This value will be assigned to the pixel if the condition is met
        cv2.THRESH_OTSU,
    )
    return image


def apply_morpholocial_operation(
    image: MatLike,
    morph_operation: MorphOperation,
    kernel_shape: KernelShape,
    kernel_radius: int,
):
    if kernel_radius == 0:
        return image

    return cv2.morphologyEx(
        image,
        morph_operation.value,
        cv2.getStructuringElement(
            kernel_shape.value,
            kernel_from_radius(kernel_radius),
        ),
    )


def find_contours(image: MatLike, mode: ContourMode = ContourMode.TREE) -> list[np.ndarray]:
    contours, _ = cv2.findContours(image, mode.value, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_longest_contour(image: MatLike) -> list[np.ndarray]:
    contours = find_contours(image, ContourMode.TREE)

    if not contours:
        return None

    return max(contours, key=lambda cnt: cv2.arcLength(cnt, closed=True))


def fill_contours(
    image: MatLike,
    contours: list[np.ndarray],
    min_area: int,
    max_area: int = float("inf"),
    fill_value: int = 0,
) -> list:
    """Fills each contour of white with black if the area is in the given range.
    Returns the unfilled contours that continue to exist."""
    unfilled_countours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            cv2.drawContours(image, [contour], -1, (fill_value), thickness=cv2.FILLED)
        else:
            unfilled_countours.append(contour)

    return unfilled_countours


def fill_all_contours(
    image: MatLike,
    min_area: int,
    max_area: int = float("inf"),
    fill_value: int = 0,
):
    fill_contours(
        image,
        find_contours(image),
        min_area,
        max_area,
        fill_value,
    )


def cut_out_rectangular_corners(
    image: MatLike, top_left: tuple[int, int], bottom_right: tuple[int, int]
) -> MatLike:
    x_start, y_start = max(0, top_left[0]), max(0, top_left[1])
    x_end, y_end = min(image.shape[1], bottom_right[0]), min(image.shape[0], bottom_right[1])
    return image[y_start:y_end, x_start:x_end]


def cut_out_rectangular(
    image: MatLike,
    anchor_point: Tuple[int, int],
    rect_size: Tuple[int, int],
    anchor: Anchor = Anchor.TOP_LEFT,
) -> tuple[MatLike, Tuple[int, int]]:
    """
    Cuts out a rotated rectangular sub-image from the image matrix.
    Returns the sub-image and the top-left corner of the sub-image.
    """
    anchor_x, anchor_y = anchor_point
    orig_width, orig_height = rect_size

    # Adjust the center based on the anchor
    match anchor.value:
        case Anchor.CENTER.value:
            center = (anchor_x, anchor_y)
        case Anchor.TOP.value:
            center = (anchor_x, anchor_y + orig_height // 2)
        case Anchor.BOTTOM.value:
            center = (anchor_x, anchor_y - orig_height // 2)
        case Anchor.LEFT.value:
            center = (anchor_x + orig_width // 2, anchor_y)
        case Anchor.RIGHT.value:
            center = (anchor_x - orig_width // 2, anchor_y)
        case Anchor.TOP_LEFT.value:
            center = (anchor_x + orig_width // 2, anchor_y + orig_height // 2)
        case Anchor.TOP_RIGHT.value:
            center = (anchor_x - orig_width // 2, anchor_y + orig_height // 2)
        case Anchor.BOTTOM_LEFT.value:
            center = (anchor_x + orig_width // 2, anchor_y - orig_height // 2)
        case Anchor.BOTTOM_RIGHT.value:
            center = (anchor_x - orig_width // 2, anchor_y - orig_height // 2)
        case _:
            raise ValueError("Invalid anchor value")

    # Calculate the top-left corner of the rectangle in the (rotated) image
    x_start = max(0, int(center[0] - orig_width // 2))
    y_start = max(0, int(center[1] - orig_height // 2))
    x_end = min(image.shape[1], x_start + orig_width)
    y_end = min(image.shape[0], y_start + orig_height)

    # Cut out the sub-image
    sub_image = image[y_start:y_end, x_start:x_end]

    # Return the sub-image and the new top-left corner of the sub-image
    return sub_image, (x_start, y_start)


def compute_orientation_with_houghline(
    image: MatLike,
    tr1,
    tr2,
    min_line_length_factor_of_width,
    threshold_hugh,
    min_line_length,
    max_line_gap,
    draw_lines_on_image=False,
):
    """Computs the orientation of the rgb image using HoughLinesP"""
    # Invert colors (background black, text white)
    bitwise_not_image = cv2.bitwise_not(image)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(bitwise_not_image, tr1, tr2, apertureSize=3)

    # Probabilistic Hough Transform to detect lines
    min_line_length = int(min_line_length_factor_of_width * bitwise_not_image.shape[1])
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=threshold_hugh,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,  # Maximum allowed gap between points on the same line
    )

    if lines is None or not lines.any():
        return 0

    # Display detected lines on the image for visualization
    angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angles.append(math.atan2(y2 - y1, x2 - x1))  # Calculate angle of each line

        # Draw orientation lines into image
        if draw_lines_on_image:
            draw_line(image, (x1, y1), (x2, y2), RED)

    # Calculate the mean angle (in radians)
    mean_angle = np.median(angles)

    # print(f"Rotation: {angle_degrees:.2f}°")
    return mean_angle


def rotate_image_degrees(image: MatLike, orientation_rad: float) -> None:
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(orientation_rad), 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (w, h), borderValue=WHITE)
    return image


def prepare_image_for_ocr(
    image: MatLike,
    text_threshold_mode: ThresholdMode,
    text_threshold_manual_value,
    text_threshold_adaptive_kernel_radius,
    text_threshold_adaptive_value,
    text_max_noise_area_to_fill,
) -> Tuple[MatLike, list[np.ndarray]]:
    """
    Scales the image, applies tresholding, adds a margin and fills black noise.
    Returns the processed image and the contours that were not filled.
    """

    # if config.text_scaling != 1.0:
    #     image = scale_image(image, config.text_scaling, ScalingMode.Cubic)

    match text_threshold_mode:
        case ThresholdMode.OTSU:
            image = apply_otsu_threshold(image)
        case ThresholdMode.MANUAL:
            image = apply_threshold(image, text_threshold_manual_value)
        case ThresholdMode.ADAPTIVE:
            image = apply_adaptive_threshold(
                image,
                text_threshold_adaptive_kernel_radius,
                text_threshold_adaptive_value,
            )

    # Create a margin around the image to ensure the text is not on the edge
    image = create_bordered_image(image)

    # Fill the small black areas of noise with white
    if text_max_noise_area_to_fill > 0:
        # Create a border around the image to ensure the contours are not on the edge
        unfilled_countours = fill_contours(
            image,
            find_contours(image),
            0,
            text_max_noise_area_to_fill,
            255,
        )
    else:
        unfilled_countours = []

    return image, unfilled_countours


def create_bordered_image(image, margin=2):
    return cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)


def create_empty_image(
    shape: Tuple[int, int, int],
    color: tuple[int, int, int] = WHITE,
):
    return np.full(shape, color, dtype=np.uint8)


def get_histogram_difference(image1: MatLike, image2: MatLike):
    """Gets the histogram difference for two grayscale images."""
    # Convert to grayscale
    # background1 = grayscale_to_rgb(background1)
    # background2 = grayscale_to_rgb(background2)
    # Calculate histograms
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

    # Compare histograms (e.g., Bhattacharyya distance)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


def is_rgb(image: MatLike) -> bool:
    return len(image.shape) == 3 and image.shape[2] == 3


def is_grayscale(image: MatLike) -> bool:
    return len(image.shape) == 2


def ensure_grayscale(image: MatLike) -> MatLike:
    """
    Converts the input image to grayscale if it is not already a single-channel image.
    """
    if is_rgb(image):  # Check for a 3-channel BGR image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif is_grayscale(image):  # Check for a single-channel image
        return image
    else:
        raise ValueError("Unsupported image format: The image must be either grayscale or BGR.")


def ensure_rgb(image: MatLike) -> MatLike:
    """
    Converts the input image to RGB if it is not already a 3-channel image.
    """
    if is_grayscale(image):  # Check for a single-channel image
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif is_rgb(image):  # Already a 3-channel image
        return image
    else:
        raise ValueError("Unsupported image format: The image must be either grayscale or BGR.")


def get_kernel(size, shape=cv2.MORPH_ELLIPSE):
    return cv2.getStructuringElement(shape, (size, size))


def get_stretched_kernel(size: tuple[int, int], shape=cv2.MORPH_ELLIPSE):
    size = (max(size[0], 2), max(size[1], 2))
    return cv2.getStructuringElement(shape, size)


def rotate_image(image: MatLike, image_rotation: Rotation):
    if image_rotation == Rotation.Rot0:
        return cv2.flip(image, 1)
    elif image_rotation == Rotation.Rot90:
        return cv2.flip(cv2.transpose(image), -1)
    elif image_rotation == Rotation.Rot180:
        return cv2.flip(image, 0)
    elif image_rotation == Rotation.Rot270:
        return cv2.transpose(image)


def increase_brightness(
    image: MatLike, multiplicative_increase: float, absolute_increase: int = 0
) -> MatLike:
    return cv2.convertScaleAbs(image, alpha=multiplicative_increase, beta=absolute_increase)
    # return np.clip(
    #     image.astype(np.float32) * multiplicative_increase, 0, 255
    # ).astype(np.uint8)


def downscale(image: MatLike, small_image_scale_factor: float) -> MatLike:
    """
    Performance by scaling factor:
    0.500: 1.3 ms
    0.490: 4.5 ms
    0.250: 2.0 ms
    0.240: 7.0 ms
    0.125: 6.0 ms
    """
    return (
        (
            cv2.resize(
                image,
                (
                    int(image.shape[1] * small_image_scale_factor),
                    int(image.shape[0] * small_image_scale_factor),
                ),
                interpolation=cv2.INTER_AREA,
            ),
        )[0]
        if small_image_scale_factor < 1
        else image
    )


def pick_colors_along_line(
    image: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int, int]]:
    """Picks all colors along a line between the start and end points."""
    # Get the coordinates of the line using Bresenham's algorithm
    line_points = cv2.line(np.zeros(image.shape[:2], dtype=np.uint8), start, end, 255, 1)
    y_indices, x_indices = np.where(line_points == 255)

    # Extract color values from the image at these coordinates
    colors = [tuple(image[y, x]) for y, x in zip(y_indices, x_indices)]

    return colors


def pick_colors_in_circle(
    image: np.ndarray, center: tuple[int, int], radius: int
) -> list[tuple[int, int, int]]:
    """Picks all colors in the circular area around the center point."""
    # Create a mask with a filled circle
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Extract pixel coordinates inside the circle
    y_indices, x_indices = np.where(mask == 255)

    # Extract color values from the image at these coordinates
    colors = [tuple(image[y, x]) for y, x in zip(y_indices, x_indices)]

    return colors


def pick_average_color_in_circle(
    image: np.ndarray, center: tuple[int, int], radius: int
) -> tuple[float, float, float]:
    """Picks the average color of the circular area around the center point."""
    # Create a mask with a filled circle
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Compute the mean color inside the circle using the mask
    mean_val = cv2.mean(image, mask=mask)

    # Return only the color channels (BGR for color images)
    return mean_val[:3]


def pick_median_color_in_circle(
    image: np.ndarray, center: tuple[int, int], radius: int
) -> tuple[float, float, float]:
    """Picks the median color of the circular area around the center point."""
    # Create a mask with a filled circle
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Extract the pixels inside the circle for each channel
    pixels = image[mask == 255]

    # Compute the median for each channel
    median_val = tuple(np.median(pixels[:, i]) for i in range(pixels.shape[1]))

    return median_val


def erode(image: MatLike, size: int | Tuple[int, int] = 3, iterations: int = 1) -> MatLike:
    """Make black parts bigger. Kernel size can be given in radius or Kernel"""

    if isinstance(size, int):
        if size == 0:
            return image
        kernel = get_kernel(size)
    elif isinstance(size, tuple):
        if size == (0, 0):
            return image
        kernel = get_stretched_kernel(size)
    else:
        raise Exception("Kernel has the wrong type")

    return cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel, iterations=iterations)


def dilate(image: MatLike, kernel_size: int = 3, iterations: int = 1) -> MatLike:
    """Make white parts bigger."""
    if kernel_size == 0:
        return image
    return cv2.morphologyEx(image, cv2.MORPH_DILATE, get_kernel(kernel_size), iterations=iterations)


def open_image(image: MatLike, kernel_size: int = 3, iterations: int = 1) -> MatLike:
    if kernel_size == 0:
        return image
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, get_kernel(kernel_size), iterations=iterations)


def close(image: MatLike, kernel_size: int = 3, iterations: int = 1) -> MatLike:
    if kernel_size == 0:
        return image
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, get_kernel(kernel_size), iterations=iterations)


def save(image, filename: str):
    # Create the directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save the image
    success = cv2.imwrite(filename, image)
    if not success:
        raise ValueError(f"Failed to save the image to {filename}")


def overlay_image_with_mask(image, mask, color=WHITE, opaqueness=1.0):
    """Overlays the image on the postions of the mask using the specified color."""
    # Create a color overlay for the mask
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask > 0] = color
    # Apply the overlay to the original image
    return cv2.addWeighted(image, 1, overlay, opaqueness, 0)


def floodfill(
    image,
    position: tuple[int, int],
    max_color_distance: int,
) -> None:
    flooded_image = image.copy()

    # Define the thresholds for color similarity (lo_diff and up_diff)
    lo_diff = (
        max_color_distance,
        max_color_distance,
        max_color_distance,
    )  # Lower bound for color difference
    up_diff = (
        max_color_distance,
        max_color_distance,
        max_color_distance,
    )  # Upper bound for color difference
    # Create the mask
    mask = np.zeros(
        (image.shape[0] + 2, image.shape[1] + 2), np.uint8
    )  # Note the +2 for the border
    # Perform flood fill
    cv2.floodFill(flooded_image, mask, position, (255, 255, 255), lo_diff, up_diff)

    # Extract the mask (remove the border added by floodFill)
    region_mask = mask[1:-1, 1:-1]  # Crop to match the original image size
    # Convert to a binary (white-black) mask
    binary_mask = (region_mask * 255).astype(np.uint8)
    return binary_mask


def create_color_similar_mask(
    image: np.ndarray, color: tuple[int, int, int], max_difference: tuple[int, int, int]
) -> np.ndarray:
    """Creates as binary mask of the image where the color is similar to the given color."""
    # Convert the image to float32 for precise calculations
    image = image.astype(np.float32)
    # Compute the lower and upper bounds for color similarity
    lower_bound = np.array([max(0, c - d) for c, d in zip(color, max_difference)], dtype=np.float32)
    upper_bound = np.array(
        [min(255, c + d) for c, d in zip(color, max_difference)], dtype=np.float32
    )
    return cv2.inRange(image, lower_bound, upper_bound)


def merge_images(imgs: List[np.ndarray], width_count: int, height_count: int) -> np.ndarray:
    # Check that the image list is not empty
    if not imgs:
        raise ValueError("The list of images is empty.")

    # Assume all images have the same dimensions
    height, width = imgs[0].shape[:2]
    channels = 1 if len(imgs[0].shape) == 2 else imgs[0].shape[2]

    # Ensure all images are 3-channel (convert grayscale to BGR if needed)
    processed_imgs = []
    for img in imgs:
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        processed_imgs.append(img)

    # Create a blank canvas with the size based on grid dimensions
    canvas = np.zeros((height * height_count, width * width_count, 3), dtype=np.uint8)

    # Place each image in the canvas
    for i, img in enumerate(processed_imgs):
        if i >= width_count * height_count:
            break  # Stop if images exceed the grid size
        y = (i // width_count) * height
        x = (i % width_count) * width
        canvas[y : y + height, x : x + width] = img

    return canvas


def merge_images_horizontally(imgs: List[np.ndarray]) -> np.ndarray:
    return merge_images(imgs, len(imgs), 1)


def merge_images_vertically(imgs: List[np.ndarray]) -> np.ndarray:
    return merge_images(imgs, 1, len(imgs))


class Conversion(Enum):
    RGB2BGR = cv2.COLOR_RGB2BGR
    BGR2RGB = cv2.COLOR_BGR2RGB
    BGR2HSV = cv2.COLOR_BGR2HSV
    RGB2HSV = cv2.COLOR_RGB2HSV
    BGR2GRAY = cv2.COLOR_BGR2GRAY


def convert(image: MatLike, conversion: Conversion):
    return cv2.cvtColor(image, conversion.value)


def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    if strength < 0:
        raise ValueError("Strength must be non-negative.")

    if strength == 0:
        return image

    # Base sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Adjust the kernel based on strength
    kernel = kernel + (strength - 1) * np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    # Apply sharpening filter
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened
