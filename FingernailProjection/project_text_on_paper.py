import drawing
import whitespace_finder
from config import make_tuple_color
from enums import cv2
from hand_contour_detector import *
from image_display import cv2, matlike_utils, np
from matlike_utils import *
from text_detection import get_font
import cv2
from typing import Callable, Optional, Tuple


class TextOnPaperProjection:
    def __init__(self, config: Config):
        self.config = config

        self.whitespace_topleft = None
        """ The position of sufficient whitespace for a rectangle to be drawn."""
        self.brightness_and_desaturation = None
        self.annotated_mask = None
        self.detection = None

        self.display = ImageDisplay(
            "Paper",
            Anchor.TOP,
            screen_index=lambda: config.debug_screen_index,
            scale_mode=ScaleMode.NATIVE,
        )

    def reset(self):
        self.whitespace_topleft = None

    def project_text(
        self,
        image: np.array,
        text: str,
        recorded_image_bgr: np.ndarray,
        detection: TextDetection,  # Text will be projected near this detection
    ) -> Optional[Callable[[np.ndarray], np.ndarray]]:

        # Reset when detection changes (in that case, we definetely need to recalc the position)
        if detection != self.detection:
            self.reset()
            self.detection = detection

        # Create text to project
        text_image = create_pillow_text_image(
            text,
            get_font(self.config.paper_text_font_size, self.config.font_path),
            make_tuple_color(self.config.paper_projection_color),
        )

        start_position_reduced = to_int_tuple(
            multiply(detection.boundingbox.center, self.config.paper_reduction_factor)
        )
        text_size_reduced = to_int_tuple(
            multiply(text_image.size, self.config.paper_reduction_factor)
        )

        drawcall = None

        if not self.whitespace_topleft:
            # Reduce image and apply gaussian blur
            small_image = matlike_utils.scale_image(
                recorded_image_bgr, self.config.paper_reduction_factor, ScalingMode.Area
            )
            small_image = matlike_utils.apply_gaussian(small_image, self.config.paper_gaussian)

            # Define threshold for white paper (low saturation, high brightness)
            hsv_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2HSV)
            saturation = hsv_image[:, :, 1]
            brightness = hsv_image[:, :, 2]
            desaturation = 255 - saturation
            self.brightness_and_desaturation = (
                self.config.paper_brightness_advantage * brightness
                + (1 - self.config.paper_brightness_advantage) * desaturation
            ).astype(np.uint8)

            mean_value = np.mean(self.brightness_and_desaturation)
            adaptive_threshold = mean_value * self.config.adaptive_factor  # Scale dynamically
            mask = cv2.bitwise_not(
                cv2.inRange(self.brightness_and_desaturation, 0, int(adaptive_threshold))
            )

            mask = matlike_utils.erode(mask, self.config.paper_eroding_kernel)

            binary_mask = (mask == 255).astype(np.uint8)

            self.whitespace_topleft = whitespace_finder.find_white_space_topleft_spiral(
                binary_mask,
                text_size_reduced,
                start_position_reduced,
                self.config.paper_spiral_step,
            )

            self.annotated_mask = matlike_utils.ensure_rgb(mask)
            if self.whitespace_topleft:
                drawing.point(self.annotated_mask, start_position_reduced, RED, 3)
                drawing.rect(
                    self.annotated_mask,
                    self.whitespace_topleft,
                    add(self.whitespace_topleft, text_size_reduced),
                    RED,
                    2,
                )

        # Find position where text can be placed
        if self.whitespace_topleft:
            overlay_image(
                image,
                np.array(text_image),
                to_int_tuple(divide(self.whitespace_topleft, self.config.paper_reduction_factor)),
                use_center_not_topleft=False,
            )

        if self.config.show_debug_windows:
            self.display.show([self.brightness_and_desaturation, self.annotated_mask])

        return drawcall
