from typing import Optional
from param import *
from enums import *
from matlike_utils import ScalingMode
from parameter_config import *
from enums import *
from PIL import ImageColor
from varname import nameof


def make_tuple_color(color: Color):
    return ImageColor.getrgb(color)[::-1]


hand_contour_detection_mode_is_threshold = Condition(
    "hand_contour_detection_mode", HandCountourDetectionMode.THRESHOLD
)
hand_contour_detection_mode_is_subtraction = Condition(
    "hand_contour_detection_mode", HandCountourDetectionMode.SUBTRACTION
)
hand_contour_threshold_mode_is_adaptive = hand_contour_detection_mode_is_threshold & Condition(
    "hand_contour_threshold_mode", ThresholdMode.ADAPTIVE
)
hand_contour_threshold_mode_is_manual = hand_contour_detection_mode_is_threshold & Condition(
    "hand_contour_threshold_mode", ThresholdMode.MANUAL
)
difference_mode_is_static = hand_contour_detection_mode_is_subtraction & Condition(
    "difference_mode", DifferenceMode.NO_PREPROCESSING
)
difference_mode_is_canny = hand_contour_detection_mode_is_subtraction & Condition(
    "difference_mode", DifferenceMode.CANNY
)
use_manual_color_balance_is_true = Condition("use_manual_color_balance", True)
project_on_finger = Condition("projection_target", ProjectionTarget.FINGER)
project_on_paper = Condition("projection_target", ProjectionTarget.PAPER)
project_on_hand = Condition("projection_target", ProjectionTarget.HAND)


def Select(value: T, doc: Optional[str] = None):
    """Create an ObjectSelector with only one object."""
    return ObjectSelector(default=value, objects=list(type(value)), doc=doc)


class Config(ParameterConfig):

    category_general = True
    projection_target = Select(ProjectionTarget.FINGER)
    ocr_credits = Integer(3, bounds=(0, 500))
    translation_direction = Select(TranslationDirection.JAPANESE_TO_ENGLISH)
    show_debug_windows = Boolean(True)
    draw_annotations_on_recording_window = Boolean(True)
    draw_annotations_on_projection_window = Boolean(True)
    draw_words_only_when_paragraph_is_hovered = Boolean(True)

    category_recording = True
    image_retrieving_mode = Select(ImageRetrievingMode.CAMERA)
    exposure_time = Integer(10000, bounds=(0, 200000), step=100)
    image_rotation = ObjectSelector(default=Rotation.Rot0, objects=list(Rotation))
    brightness_scale_factor = Number(1, bounds=(1, 10))
    use_manual_color_balance = Boolean(False)
    red_balance = Number(1, bounds=(0.25, 8), step=0.01, doc=use_manual_color_balance_is_true.doc)
    blue_balance = Number(1, bounds=(0.25, 8), step=0.01, doc=use_manual_color_balance_is_true.doc)

    category_finger_detection = False
    skincolor_open_radius = Integer(3, bounds=(1, 50))
    skincolor_close_radius = Integer(3, bounds=(1, 50))
    linestring_resample_distance = Integer(3, bounds=(1, 20))
    linestring_smoothing = Integer(3, bounds=(0, 50))
    finger_width_sample_rate_px = Integer(2, bounds=(1, 20))

    category_stillstand_detection = False
    position_queue_length = Integer(3, bounds=(0, 15))
    max_cursor_mean_deviation_for_snapshot = Integer(3, bounds=(0, 20))
    distance_from_cursor_for_snapshot_reset = Integer(25, bounds=(0, 100))
    standstill_time_until_snapshot_ms = Integer(1000, bounds=(0, 2000), step=10)
    projection_delay_ms = Integer(50, bounds=(0, 1000))

    category_stillstand_detection_visualization = False
    cursor_circle_start_radius_mm = Integer(20, bounds=(0, 100))
    cursor_size = Integer(3, bounds=(0, 10))
    cursor_color = Color("#FFFFFF")
    cursor_circle_thickness = Integer(2, bounds=(0, 5))
    standstill_color = Color("#AA7777")

    category_cutout = False
    cutout_scale_factor = Number(1, bounds=(0.10, 1), step=0.05)
    cutout_size_as_multiple_of_tip_to_joint = Number(2, bounds=(1, 3))

    category_fingertip_geometry = True
    fingermark_down_move_mm = Integer(0, bounds=(-15, 15))
    """ Move fingermark a bit down to make sure its inside the finger as the 
    MediaPipe fingertip mark can be dangerously close to the contour. """
    finger_cursor_position_smoothing = Number(0.2, bounds=(0.0, 1.0))
    finger_angle_smoothing = Number(0.5, bounds=(0.0, 1.0))
    fingertip_contour_color = Color("#0000FF")
    fingertip_circle_color = Color("#FF0000")
    fingertip_visual_point_size = Integer(2, bounds=(0, 15))
    fingertip_visual_line_width = Integer(2, bounds=(0, 15))
    fingertip_visual_line_width_big = Integer(6, bounds=(0, 50))

    # fmt: off
    category_text_projection = True
    text_projection_enabled = Boolean(True)
    show_translation_no_reading = Boolean(True)
    # PAPER PROJECTION
    paper_always_reset = Boolean(True)
    paper_projection_color = Color("#FFFFFF", doc=project_on_paper.doc)
    paper_text_font_size = Integer(14, bounds=(6, 80), doc=project_on_paper.doc)
    paper_gaussian = Integer(3, bounds=(0, 15), doc=project_on_paper.doc)
    paper_reduction_factor = Number(0.25, bounds=(0.05, 1), step=0.05, doc=project_on_paper.doc)
    paper_brightness_advantage = Number(0.5, bounds=(0, 1), doc=project_on_paper.doc)
    paper_max_desaturation_brightness_sum = Integer(90, bounds=(0, 255), doc=project_on_paper.doc)
    paper_spiral_step = Integer(2, bounds=(0, 5), doc=project_on_paper.doc)
    paper_eroding_kernel = Integer(2, bounds=(1, 20))
    adaptive_factor = Number(1.0, bounds=(0.05, 2), step=0.05, doc=project_on_paper.doc)
    # FINGER AND HAND PROJECTION
    skin_projection_color = Color("#FFFFFF", doc=project_on_finger.doc)
    min_distance_change_for_smoothing_stop = Integer(30, bounds=(0, 200))
    min_up_vector_change_for_smoothing_stop = Number(0.1, bounds=(0, 0.3), step=0.01)
    # FINGER PROJECTION
    finger_text_font_size = Integer(14, bounds=(0, 80), doc=project_on_finger.doc)
    finger_vertical_projection_offset_mm = Integer(-12, bounds=(-50, 50), doc=project_on_finger.doc)
    finger_horizontal_projection_offset_mm = Integer(-0, bounds=(-10, 10), doc=project_on_finger.doc)
    finger_textflow_mode = Select(TextflowMode.VERTICAL_KANJI, doc=project_on_finger.doc)
 # HAND PROJECTION
    hand_textflow_mode = Select(TextflowMode.VERTICAL_KANJI, doc=project_on_hand.doc)
    hand_text_font_size = Integer(14, bounds=(6, 80), doc=project_on_hand.doc)
    hand_vertical_projection_offset_mm = Integer(0, bounds=(-50, 50), doc=project_on_hand.doc)
    hand_horizontal_projection_offset_mm = Integer(-0, bounds=(-10, 10), doc=project_on_hand.doc)
    # fmt: on

    category_mediapipe = True
    mediapipe_use_complex_model = Boolean(False)
    min_mediapipe_confidence = Number(0.5, bounds=(0.0, 1.0))
    mediapipe_smallering_scaling_mode = Select(ScalingMode.Area)
    mediaipipe_image_smallering_factor = Number(0.25, bounds=(0.05, 1.0), step=0.05)
    mediapipe_point_size = Integer(3, bounds=(0, 30))
    mediapipe_highlight_point_size = Integer(3, bounds=(0, 30))
    mediapipe_line_size = Integer(3, bounds=(0, 30))
    mediapipe_color = Color("#FFFFFF")
    mediapipe_pointing_color = Color("#FFFFFF")
    mediaipipe_grab_midjoint_to_basejoint_ratio = Number(0.5, bounds=(0.00, 1.0), step=0.05)

    category_bounding_boxes = None
    skincolor_reduction_factor = Number(0.25, bounds=(0.05, 1), step=0.01)
    skincolor_opening_kernel = Integer(2, bounds=(1, 20))
    skincolor_eroding_kernelsize_x = Integer(2, bounds=(0, 20))
    skincolor_eroding_kernelsize_y = Integer(2, bounds=(0, 20))
    show_bounding_box = Boolean(True)
    bounding_box_color_untranslated = Color("#FFFFFF")
    bounding_box_color = Color("#FFFFFF")
    bounding_box_thickness = Integer(1, bounds=(0, 5))
    bounding_box_thickness_on_hover = Integer(2, bounds=(0, 5))

    category_repeating_detected_text = None
    detected_text_enabled = Boolean(False)
    detected_text_bounding_box_anchor = Select(Anchor.TOP_LEFT)
    detected_text_font_size = Integer(14, bounds=(2, 32))
    detected_text_offset_x = Integer(0, bounds=(-60, 60))
    detected_text_offset_y = Integer(0, bounds=(-60, 60))
    show_confidence_score = Boolean(False)

    # category_contour = None
    # clip_contour_outside_of_text = Boolean(True)
    # contour_selection_mode = ObjectSelector(
    #     default=ContourSelectionMode.LONGEST, objects=list(ContourSelectionMode)
    # )
    # minimum_contour_length = Integer(400, bounds=(0, 5000))
    # curve_averaging = Integer(10, bounds=(0, 100))

    category_image_stabilization = False
    stabilization_enabled = Boolean(True)
    nfeatures = Integer(300, bounds=(1, 1000))
    fast_threshold = Integer(20, bounds=(1, 100))
    scale_factor = Number(1.5, bounds=(1.0, 2.0))
    ransac_value = Integer(10, bounds=(1, 50))
    blurr_size = Integer(10, bounds=(1, 50))
    use_affine = Boolean(True)

    category_ocr_limits = None
    min_detection_confidence = Number(0.9, bounds=(0.0, 1.0))
    min_detection_text_size = Integer(0, bounds=(0, 1000))
    """ The minimal size of the SMALLER side of a textbox in order to be accepted."""

    category_google_ocr = None
    include_paragraphs = Boolean(True)
    include_symbols = Boolean(True)

    category_camera_bounding = None
    camera_bounding_box_thickness = Integer(1, bounds=(0, 5))
    camera_bounding_box_color = Color("#FFFFFF")
    camera_scanline_thickness = Integer(1, bounds=(0, 100))
    camera_scanline_color = Color("#FFFFFF")

    category_logs = True
    show_camera_bounding = Boolean(False)
    show_mediapipe_points = Boolean(False)
    show_calibration_points = Boolean(False)
    log_main = Boolean(False)
    log_fingertip_data_retriever = Boolean(False)
    log_google_ocr = Boolean(False)
    log_stabilization = Boolean(False)
    log_contour_detection = Boolean(False)
    log_thresholding = Boolean(False)
    log_subtraction_contour_detection = Boolean(False)

    category_other = None
    finger_cursor_smoothing = Integer(30, bounds=(0, 500))
    finger_cursor_move_up_mm = Integer(5, bounds=(0, 20))
    """ Moves the cursor a bit more outside of the finger. """
    max_distance_from_bounding_box = Integer(1, bounds=(1, 100))
    pause_application = Boolean(False)
    freeze_camera = Boolean(False)
    """ If true, no new image will be captured. Instead, the last image will be used again. """
    debug_screen_index = Integer(0, bounds=(0, 2))
    projection_screen_index = Integer(0, bounds=(0, 2))

    category_paths = None
    font_path = String(
        r"C:\Users\sens\Desktop\FingernailProjection\FingernailProjection\ARTranslator-master\Koruri-Regular.ttf"
    )
    frozen_camera_image_path = String(
        r"C:\Users\sens\Desktop\FingernailProjection\FingernailProjection\frozen_camera_image.png"
    )

    category_text_proprocessing = False
    proprocessing_enabled_SCALE_ERROR = Boolean(False)
    cutout_size_mm = Integer(50, bounds=[10, 200])
    text_scale = Number(2, bounds=[1, 4])
    text_threshold_mode = ObjectSelector(default=ThresholdMode.MANUAL, objects=list(ThresholdMode))
    text_threshold_manual_value = Integer(128, bounds=(0, 255))
    text_threshold_adaptive_kernel_radius = Integer(128, bounds=(0, 255))
    text_threshold_adaptive_value = Integer(128, bounds=(0, 255))
    sharpening_strength = Number(2, bounds=[0, 8])

    ##############################################################

    # ONLY AVAILABLE IN DEBUG FILE
    category_continous_background_subtraction_debug = False
    subtraction_mode = Select(BackgroundSubtractorMode.KNN)
    history_count = Integer(300, bounds=(10, 1000))
    sensitivity_knn = Integer(
        1000,
        bounds=(100, 10000),
        doc=Condition(nameof(subtraction_mode), BackgroundSubtractorMode.KNN).doc,
    )
    sensitivity_mog2 = Integer(
        50,
        bounds=(0, 100),
        doc=Condition(nameof(subtraction_mode), BackgroundSubtractorMode.MOG2).doc,
    )
    detect_shadows = Boolean(False)

    category_floodfill = False
    floodfill_use_hsv = Boolean(True)
    floodfill_smoothing_radius = Integer(10, bounds=(1, 10))
    floodfill_threshold = Integer(10, bounds=(1, 10))
