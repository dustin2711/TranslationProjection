import drawing
from config import Config, make_tuple_color
from enums import Anchor, TextflowMode
from image_display import Anchor, join_string
from matlike_utils import *
from position_smoother import FloatSmoother, TupleSmoother
from text_detection import get_font


class TextOnSkinProjection:
    def __init__(self, config: Config):
        self.config = config
        self.position_smoother = TupleSmoother()
        self.orientation_smoother = TupleSmoother()

    def project_text(
        self,
        image: np.array,
        text,
        base_position: Tuple2,
        up_direction: Tuple2,
        offset: Tuple2,
        show_text_to_the_left: bool,
        textflow_mode: TextflowMode,
        font_size: int,
    ):
        # Smooth base position and orientation
        base_position = self.position_smoother.smooth_velocity_based(
            base_position, self.config.min_distance_change_for_smoothing_stop
        )
        up_direction = self.orientation_smoother.smooth_velocity_based(
            up_direction.normalized, self.config.min_up_vector_change_for_smoothing_stop
        )

        # Add vertical and horizontal offset
        horizontal_direction = up_direction.rotate(90)
        text_position = base_position.move_in_direction(up_direction, offset[0]).move_in_direction(
            horizontal_direction, offset[1]
        )

        if textflow_mode == TextflowMode.VERTICAL_KANJI:
            angle = -90
            anchor = Anchor.TOP
            use_ascend_and_descend = False
            text = join_string(text, "\n")
        else:
            base_angle = 90 if show_text_to_the_left else -90
            extra_angle = 0 if textflow_mode == TextflowMode.HORIZONTAL else -90
            angle = base_angle + extra_angle
            anchor = (
                (Anchor.RIGHT if show_text_to_the_left else Anchor.LEFT)
                if textflow_mode == TextflowMode.VERTICAL
                else Anchor.TOP
            )
            use_ascend_and_descend = True

        drawing.draw_roated_text(
            image,
            text,
            to_int_tuple(text_position),
            -up_direction.orientation_degrees + angle,
            anchor,
            get_font(font_size, self.config.font_path),
            make_tuple_color(self.config.skin_projection_color),
            use_ascend_and_descend=use_ascend_and_descend,
        )
