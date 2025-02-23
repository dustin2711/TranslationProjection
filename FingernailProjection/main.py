import json
from Circle import Circle
from EightPointCalibration import EightPointCalibration
from aadrawing import draw_thin_polygon
from file_watcher import FileWatcher
from get_openai_translations import get_openai_translations
import helper
from project_text_on_finger import TextOnSkinProjection
from project_text_on_paper import TextOnPaperProjection
from skincolor_threshold_image import skincolor_threshold_image
from standstill_detector import StandstillDetector
from automatic_config_reloader import AutomaticConfigReloader
import drawing
from rotated_rect_crop import crop_rotated_rectangle
from skin_detect_config import SkinDetectConfig
from fingertip_data import FingertipData
from stopwatch import Stopwatch
from image_provider import FileImageProvider, ImageProvider
from pylon_image_capture import PylonImageCapture
from config import Config, make_tuple_color
import cv2
from image_display import ImageDisplay
from enums import *
from mediapipe_landsmarks_extractor import MediapipeLandmarksRetriever
from hand_contour_detector import *
from position_smoother import *
from ocr_provider import *
from text_detection import TextDetection
from matlike_utils import *
from monitor_watcher import MonitorWatcher
from cv2.typing import MatLike
from color_helper import *
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

ESCAPE_KEY = 27
SPACE_KEY = 32

stopwatch = Stopwatch(logging_enabled=False)
stopwatch_total = Stopwatch(logging_enabled=True)

config_loader = AutomaticConfigReloader(
    r"C:\Users\sens\Desktop\FingernailProjection\FingernailProjection\config.json", Config
)
config: Config = config_loader.config
skin_config, skin_config_reloader = AutomaticConfigReloader.get_config_and_reloader(
    r"C:\Users\sens\Desktop\FingernailProjection\FingernailProjection\skin_config.json",
    SkinDetectConfig,
)

monitor_watcher = MonitorWatcher()

cursor_smoother = TupleSmoother()

# Cursor stuff
cursor_mode = CursorMode.Mouse
""" Determines if the mouse or finger is used as a cursor."""
mouse_cursor_position = (0, 0)
""" Last position of the mouse on the base image in pixels."""
mouse_cursor_position_in_projection = (0, 0)
""" Last position of the mouse on the base image in pixels."""
finger_cursor_position = (0, 0)
""" Last position of the fingertip on the base image in pixels."""
fingerangle_smoother = FloatSmoother()

image_getter_by_mode: dict[ImageRetrievingMode, ImageProvider] = {
    ImageRetrievingMode.FROM_FILE: FileImageProvider(config, "Yotsubato_18.jpg"),
    ImageRetrievingMode.CAMERA: PylonImageCapture(config),
    # ImageRetrievingMode.CAMERA: XimeaImageCapture(config),
}

ocr_provider = FileOcrProvider(config, "yotsubato_18_ocr_response.pkl")
# stabilizer = ImageStabilizer(config)
landmarks_extractor = MediapipeLandmarksRetriever(config)
standstill_detector = StandstillDetector(config)
calibration = EightPointCalibration()

paper_projection = TextOnPaperProjection(config)
skin_projection = TextOnSkinProjection(config)

max_tip_to_joint_distance = 0
""" This variable saves the max ever occured distance between fingertip and first joint, in order to make cutout always the same big."""
pixel_per_mm = 10

ocr_data_path = (
    r"C:\Users\sens\Desktop\FingernailProjection\FingernailProjection\standstill_ocr_response.pkl"
)
# ocr_data: GoogleOcrData = picke_load(ocr_data_path)
ocr_data = None
image_scan_for_ocr_needed = True

previous_recorded_image_bgr: MatLike = None
empty_projection_image: np.array = None

translationdict_path = r"C:\Users\sens\Desktop\FingernailProjection\FingernailProjection\translationdicts_by_paragraph.pkl"
translationdicts_by_paragraphtext = helper.picke_load_or_create(translationdict_path, lambda: {})
translated_word_detections = []
untranslated_word_detections = []

executor = ThreadPoolExecutor(max_workers=1)


def update_word_detections():
    global translationdicts_by_paragraphtext, translated_word_detections, untranslated_word_detections

    if ocr_data is None:
        return

    translated_paragraphtexts = set(translationdicts_by_paragraphtext.keys())

    for paragraph in ocr_data.paragraph_detections:
        if paragraph.text in translated_paragraphtexts:
            translated_word_detections.extend(paragraph.children)
        else:
            untranslated_word_detections.extend(paragraph.children)


# This need to be called after translationdicts_by_paragraphtext has been loaded
update_word_detections()


def screen_mouse_event_happened(event, x, y, flags, param):
    global cursor_mode, mouse_cursor_position
    mouse_cursor_position = base_display.convert_screen_to_image_position((x, y))
    cursor_mode = CursorMode.Mouse

    if event == cv2.EVENT_LBUTTONDOWN:
        calibration.add_screen_point(mouse_cursor_position)


def projection_mouse_event_happened(event, x, y, flags, param):
    global mouse_cursor_position_in_projection
    mouse_cursor_position_in_projection = projection_display.convert_screen_to_image_position(
        (x, y)
    )

    if event == cv2.EVENT_LBUTTONDOWN:
        calibration.add_projector_point(mouse_cursor_position_in_projection)


base_display = ImageDisplay(
    "Recording",
    Anchor.TOP_LEFT,
    screen_index=lambda: config.debug_screen_index,  # This is the left screen with projector enabled
    scale_mode=ScaleMode.FILL_SCREEN_HEIGHT,
    # scale_mode=ScaleMode.NATIVE,
    mouse_event_callback=screen_mouse_event_happened,
)

finger_cutout_display = ImageDisplay(
    "Finger Cutout",
    Anchor.TOP_RIGHT,
    screen_index=lambda: config.debug_screen_index,
    scale_mode=ScaleMode.NATIVE,
)

fingertip_display = ImageDisplay(
    "Fingertip",
    Anchor.BOTTOM_RIGHT,
    screen_index=lambda: config.debug_screen_index,
    scale_mode=ScaleMode.DOUBLE,
)

skincolor_hand_display = ImageDisplay(
    "Skincolor Hand",
    Anchor.TOP,
    screen_index=lambda: config.debug_screen_index,  # This is the left screen with projector enabled
    scale_mode=ScaleMode.FILL_SCREEN_HEIGHT,
    # scale_mode=ScaleMode.NATIVE,
    mouse_event_callback=screen_mouse_event_happened,
)

projection_display = ImageDisplay(
    "Projection",
    Anchor.TOP_RIGHT,
    screen_index=lambda: config.projection_screen_index,
    scale_mode=ScaleMode.FILL,
    show_titlebar=False,
    mouse_event_callback=projection_mouse_event_happened,
)


def reset():
    global image_scan_for_ocr_needed, ocr_data, finger_cursor_position

    image_scan_for_ocr_needed = True
    ocr_data = None

    finger_cursor_position = None

    # Clear stillstand detection stuff
    standstill_detector.previous_positions.clear()

    print("Reset was executed.")


translations_dict_path = (
    r"C:\Users\sens\Desktop\FingernailProjection\FingernailProjection\translations.json"
)


def reload_translations() -> dict[str, str]:
    global translation_by_word
    with open(
        translations_dict_path,
        "r",
        encoding="utf-8",
    ) as f:
        translation_by_word = json.load(f)


translation_by_word: dict[str, str] = reload_translations()

translations_file_watcher = FileWatcher(translations_dict_path)


def get_translation(word_detection: TextDetection):
    """
    Returns the translation of the given word detection.
    Translates the paragraph of the given word detection if not done yet.
    """
    global translation_by_word
    if word := translation_by_word.get(word_detection.text, None):
        return word
    else:
        print(f"Error: {word_detection.text}")
        return "ERROR"

    # # Translate the whole paragraph
    # paragraph: TextDetection = word_detection.parent
    # key: str = paragraph.text
    # words: list[str]
    # global translation_by_word

    # if not (translation_by_word := translationdicts_by_paragraphtext.get(key, None)):
    #     words = [detection.text for detection in paragraph.children]
    #     translation_by_word = get_openai_translations(words, config.translation_direction)
    #     translationdicts_by_paragraphtext[key] = translation_by_word

    #     update_word_detections()

    #     helper.pickle_save(translationdict_path, translationdicts_by_paragraphtext)
    # else:
    #     pass
    # # Map the index of the word detection to the translation
    # # index = paragraph.children.index(word_detection)
    # # translation = translations[index]
    # if translation := translation_by_word.get(word_detection.text, None):
    #     return translation
    # else:
    #     try:
    #         print(
    #             f"{word_detection.text} could not be translated.\n{join_string(translation_by_word.values(), ", ", lambda pair: f'{pair[0]}: {pair[1]}')}"
    #         )
    #     except:
    #         print("Failed loading printing error.")
    #     return "ERROR"


def get_hovered_paragraph_and_word_detection(
    position: tuple[int, int],
    ocr_data: GoogleOcrData,
) -> Optional[tuple[TextDetection, TextDetection]]:
    """Gets the word detection on the specified position."""

    if position is None:
        return None, None

    close_paragraph_detections = [
        it
        for it in ocr_data.paragraph_detections
        if it.boundingbox.distance_to_point(position) < config.max_distance_from_bounding_box
    ]
    stopwatch.logandrestart("Got close paragaphs")

    # Find the paragraph detection closest du cursor
    hovered_paragraph_detection = min(
        close_paragraph_detections,
        key=lambda it: it.boundingbox.distance_to_point(position),
        default=None,
    )
    stopwatch.logandrestart("Got hovered_paragraph_detection")

    if hovered_paragraph_detection is None:
        return None, None

    # Filter word detection too far away from cursor
    close_detections = [
        it
        for it in hovered_paragraph_detection.children
        if it.boundingbox.distance_to_point(position) < config.max_distance_from_bounding_box
    ]

    # Find the word detection closest du cursor
    hovered_word_detection = min(
        close_detections,
        key=lambda it: it.boundingbox.distance_to_point(position),
        default=None,
    )
    stopwatch.logandrestart("Got hovered_word_detection")

    return (hovered_paragraph_detection, hovered_word_detection)


def get_skincolor_mask(reduced_image: np.array, original_image_size: tuple[int, int]):
    """Gets the mask where the hand is determined by the skincolor."""
    # print(f"config.log_main={config.log_main}")
    # stopwatch2 = Stopwatch(logging_enabled=config.log_main)
    # Add drawcall to remove bounding boxes at hand area
    reduced_skincolor_mask = skincolor_threshold_image(reduced_image, skin_config, True)
    # stopwatch2.logandrestart("Detected skin")
    reduced_skincolor_mask = matlike_utils.open_image(
        reduced_skincolor_mask, config.skincolor_opening_kernel
    )
    reduced_skincolor_mask = matlike_utils.erode(
        reduced_skincolor_mask,
        (config.skincolor_eroding_kernelsize_x, config.skincolor_eroding_kernelsize_y),
    )
    # stopwatch2.logandrestart("Dilates skin mask")
    mask = matlike_utils.scale_image_to(
        reduced_skincolor_mask,
        original_image_size,
        ScalingMode.Nearest,
    )
    # stopwatch2.logandrestart("Enlarged skin mask")
    return mask


# Improve cv2 performance
cv2.setUseOptimized(True)
cv2.setNumThreads(cv2.getNumberOfCPUs())

# Start main loop
while True:

    stopwatch_total.restart(True)
    stopwatch.restart(config.log_main)

    config_loader.reload_if_config_changed()
    skin_config_reloader.reload_if_config_changed()

    if translations_file_watcher.has_timestamp_changed():
        reload_translations()

    if config.pause_application:
        cv2.waitKey(1)
        continue

    image_provider = image_getter_by_mode[config.image_retrieving_mode]

    # Load previous_recorded_image_bgr from path if neccessary (after restart e.g.)
    if config.freeze_camera and previous_recorded_image_bgr is None:
        previous_recorded_image_bgr = cv2.imread(config.frozen_camera_image_path)

    recorded_image_bgr = (
        previous_recorded_image_bgr if config.freeze_camera else image_provider.get_image()
    )

    # This image contains all annotations for the projection and for debug
    annotation_image = matlike_utils.create_empty_image(recorded_image_bgr.shape, BLACK)

    stopwatch.logandrestart("Retrieved image")

    # if config.stabilization_enabled:
    #     image = stabilizer.stabilize(image)
    #     stopwatch.logandrestart("stabilize")

    tip_data = None
    fingertip_countour_circle: Circle = None
    """ Circle approximating the finger countour. """
    fingernail_circle: Circle = None
    """ Circle approximating the fingernail. """
    finger_orientation_degrees: float = 0

    finger_cursor_position = None

    # Calculate skincolor mask in parallel. But first reduce image so there is no collision with recorded_image_bgr usage
    reduced_image = matlike_utils.scale_image(
        recorded_image_bgr, config.skincolor_reduction_factor, ScalingMode.Area
    )
    stopwatch.logandrestart("Reduced size for skin detection")
    future = executor.submit(
        get_skincolor_mask,
        reduced_image,
        (recorded_image_bgr.shape[1], recorded_image_bgr.shape[0]),
    )

    pointing_finger = None

    if landmarks := landmarks_extractor.find_hand_landmarks(recorded_image_bgr, cv2.COLOR_BGR2RGB):
        stopwatch.logandrestart("Found mediapipe landmarks")

        if landmarks.is_grabbing(config.mediaipipe_grab_midjoint_to_basejoint_ratio):
            reset()
        else:
            # Use the finger as cursor since mediapipe detected a hand
            cursor_mode = CursorMode.Finger

            # Calculate pixel per mm
            pixel_distance_between_index_and_pinky_base = euclidean_distance(
                landmarks.index.basejoint, landmarks.pinky.basejoint
            )
            mm_distance_between_index_and_pinky_base = 65

            pixel_per_mm = (
                pixel_distance_between_index_and_pinky_base
                / mm_distance_between_index_and_pinky_base
            )

            pointing_finger = landmarks.get_farthest_finger(1.2)

            max_tip_to_joint_distance = max(
                euclidean_distance(pointing_finger.tip, pointing_finger.frontjoint),
                max_tip_to_joint_distance,
            )

            finger_orientation_degrees = fingerangle_smoother.smooth(
                get_orientation_degrees(subtract(pointing_finger.tip, pointing_finger.midjoint)),
                config.finger_angle_smoothing,
            )

            # Set cutout parameters
            cutout_width = int(
                config.cutout_size_as_multiple_of_tip_to_joint * max_tip_to_joint_distance
            )
            cutout_size = (cutout_width, cutout_width)
            cutout_center = pointing_finger.tip
            cutout_degrees = 90 + finger_orientation_degrees
            stopwatch.logandrestart("Do fingertip and cutout calculations")

            cutout_image = crop_rotated_rectangle(
                recorded_image_bgr, (cutout_center, cutout_size, cutout_degrees)
            )

            stopwatch.logandrestart("Crop rectangle")

            if cutout_image is not None:
                cutout_image = downscale(cutout_image, config.cutout_scale_factor)
                center_in_cutout = (
                    int(0.5 * config.cutout_scale_factor * cutout_width),
                    int(0.5 * config.cutout_scale_factor * cutout_width),
                )

                def transform_float_cutout_to_base(value: float):
                    return value / config.cutout_scale_factor

                def transform_position_cutout_to_base(position: tuple[float, float]):
                    target_from_center_in_cutout = subtract(position, center_in_cutout)
                    fingertipmark_to_apex = divide(
                        target_from_center_in_cutout, config.cutout_scale_factor
                    )
                    return add(
                        cutout_center,
                        rotate_vector(fingertipmark_to_apex, cutout_degrees),
                    )

                fingermark_in_cutout = Tuple2(
                    center_in_cutout[0],
                    center_in_cutout[1]
                    + config.fingermark_down_move_mm * pixel_per_mm * config.cutout_scale_factor,
                ).as_int

                thresholded_cutout_image = skincolor_threshold_image(cutout_image, skin_config)
                stopwatch.logandrestart("Skincolor-thresholding")

                tip_data: FingertipData
                if tip_data := FingertipData.create_from_mask(
                    thresholded_cutout_image,
                    config,
                    fingermark_in_cutout,
                ):
                    if config.show_debug_windows:
                        finger_cutout_display.show(
                            [cutout_image, tip_data.mask, tip_data.mask_opened_closed]
                        )
                        stopwatch.logandrestart("Got fingertip data")

                    # Get the cursor position in the cutout image by moving the circle center up
                    finger_cursor = Tuple2(
                        tip_data.finger_apex[0],
                        tip_data.finger_apex[1]
                        - config.finger_cursor_move_up_mm
                        * pixel_per_mm
                        * config.cutout_scale_factor,
                    ).as_int

                    ## Draw in cutout image
                    # Draw finger linestring and mark
                    if tip_data.linestring:
                        draw_linestring(
                            cutout_image,
                            tip_data.linestring,
                            make_tuple_color(config.fingertip_contour_color),
                            config.fingertip_visual_line_width,
                        )
                    drawing.point(
                        cutout_image,
                        fingermark_in_cutout,
                        make_tuple_color(config.mediapipe_color),
                        config.fingertip_visual_point_size,
                    )
                    # Draw approximating circle and cursor
                    helpcolor = make_tuple_color(config.fingertip_circle_color)
                    drawing.point(
                        cutout_image,
                        tip_data.finger_apex,
                        helpcolor,
                        config.fingertip_visual_point_size,
                    )
                    # Draw Vertical line
                    drawing.draw_line(
                        cutout_image,
                        finger_cursor,
                        Tuple2(tip_data.finger_apex[0], cutout_width),
                        helpcolor,
                        config.fingertip_visual_line_width,
                    )
                    # Draw hozizontal line
                    drawing.draw_line(
                        cutout_image,
                        tip_data.finger_apex,
                        tip_data.most_upper_position,
                        helpcolor,
                        config.fingertip_visual_line_width,
                    )
                    drawing.point(
                        cutout_image, finger_cursor, helpcolor, config.fingertip_visual_point_size
                    )

                    stopwatch.logandrestart("Drawn in cutout image")

                    # Calculate cursor position in base image
                    finger_cursor_position = transform_position_cutout_to_base(finger_cursor)

                    finger_cursor_position = cursor_smoother.smooth_velocity_based(
                        Tuple2(finger_cursor_position), config.finger_cursor_smoothing
                    )

                    # Projection may be stopped if the OCR image is taken
                    projection_is_allowed = True

                    if image_scan_for_ocr_needed:
                        # If standstill position is not set, check for standstill
                        progress_ms = standstill_detector.get_standstill_progress(
                            finger_cursor_position
                        )
                        stopwatch.logandrestart(
                            f"Standstill progress: {progress_ms} / {config.standstill_time_until_snapshot_ms}"
                        )

                        # Draw a shrinking circle when standstill position will be set
                        if 0 < progress_ms <= config.standstill_time_until_snapshot_ms:
                            time_until_projection_stop_ms = (
                                config.standstill_time_until_snapshot_ms
                                - progress_ms
                                - config.projection_delay_ms
                            )

                            projection_is_allowed = time_until_projection_stop_ms > 0

                            # The time until snapshot must exceed a delay for projection being enabled
                            if projection_is_allowed:
                                progress = (
                                    time_until_projection_stop_ms
                                    / config.standstill_time_until_snapshot_ms
                                )
                                # Show srinking circle
                                draw_circle(
                                    annotation_image,
                                    finger_cursor_position,
                                    config.cursor_circle_start_radius_mm * pixel_per_mm * progress,
                                    make_tuple_color(config.cursor_color),
                                    config.cursor_circle_thickness,
                                )

                                # # Show scanline
                                # image_height, image_width = recorded_image_bgr.shape[:2]
                                # y_position = progress * image_height
                                # cursor_drawcalls.append(
                                #     lambda image: draw_line(
                                #         image,
                                #         to_int_tuple((0, y_position)),
                                #         to_int_tuple((image_width, y_position)),
                                #         make_tuple_color(config.camera_scanline_color),
                                #         config.camera_scanline_thickness,
                                #     )
                                # )

                        elif progress_ms >= config.standstill_time_until_snapshot_ms:
                            print(f"Finger standstill long enough - SNAPSHOT!  {progress_ms}")

                            reload_translations()

                            # With credits, create dynamically. Without, load from file.
                            if config.ocr_credits > 0:
                                config.ocr_credits -= 1
                                config.save_to_file(config.path)
                                print(f"Used a credit. Left credits: {config.ocr_credits}")

                                ocr_data = GoogleOcrData.from_image(recorded_image_bgr, config)
                            else:
                                ocr_data = picke_load_or_create(
                                    r"C:\Users\sens\Desktop\FingernailProjection\FingernailProjection\standstill_ocr_response.pkl",
                                    lambda: GoogleOcrData.from_image(recorded_image_bgr, config),
                                )

                            if config.min_detection_text_size > 0:
                                # Remove small detections
                                ocr_data.word_detections = [
                                    detection
                                    for detection in ocr_data.word_detections
                                    if min(
                                        detection.boundingbox.width, detection.boundingbox.height
                                    )
                                    >= config.min_detection_text_size
                                ]

                            image_scan_for_ocr_needed = False

                    if projection_is_allowed:
                        # Draw cursor
                        drawing.point(
                            annotation_image,
                            finger_cursor_position,
                            make_tuple_color(config.cursor_color),
                            config.cursor_size,
                        )

                    # Convert fingertip circle to base image
                    fingertip_countour_circle = Circle(
                        transform_position_cutout_to_base(tip_data.finger_apex),
                        transform_float_cutout_to_base(tip_data.half_finger_width),
                    )

                    stopwatch.logandrestart("Drawn base image")

                if config.show_debug_windows:
                    fingertip_display.show(cutout_image)
                    stopwatch.logandrestart("Shown fingertip_display")
            else:
                print("Could not cutout.")

    if ocr_data:
        selected_paragraph_detection, selected_word_detection = (
            get_hovered_paragraph_and_word_detection(
                (
                    mouse_cursor_position
                    if cursor_mode == CursorMode.Mouse
                    else finger_cursor_position
                ),
                ocr_data,
            )
        )
    else:
        selected_paragraph_detection = None
        selected_word_detection = None

    stopwatch.logandrestart("Got hovered word detection:")

    if ocr_data and config.show_bounding_box:

        box_color = make_tuple_color(config.bounding_box_color)

        if config.draw_words_only_when_paragraph_is_hovered:

            # Draw paragraph detections
            for detection in ocr_data.paragraph_detections:

                if detection == selected_paragraph_detection:
                    continue

                draw_thin_polygon(
                    annotation_image,
                    detection.boundingbox.points,
                    box_color,
                )

            # Set word detections
            if selected_paragraph_detection:
                word_detections = [word for word in selected_paragraph_detection.children]
            else:
                word_detections = []
        else:
            # Set word detections
            word_detections = [
                word for para in ocr_data.paragraph_detections for word in para.children
            ]

        # Draw word detection
        for detection in word_detections:
            if detection == selected_word_detection:
                draw_polygon(
                    annotation_image,
                    detection.boundingbox.points,
                    box_color,
                    config.bounding_box_thickness_on_hover,
                )
            else:
                draw_thin_polygon(
                    annotation_image,
                    detection.boundingbox.points,
                    box_color,
                )

        # Get text detextion bounding box draw calls
        # TextDetection.draw_bounding_boxes(
        #     annotation_image,
        #     translated_word_detections,
        #     hovered_detection,
        #     config,
        #     make_tuple_color(config.bounding_box_color),
        # )
        # TextDetection.draw_bounding_boxes(
        #     annotation_image,
        #     untranslated_word_detections,
        #     hovered_detection,
        #     config,
        #     make_tuple_color(config.bounding_box_color_untranslated),
        # )
        stopwatch.logandrestart("Got text bounding box draw calls")

        # Later, when you need the result:
        skincolor_mask = future.result()  # Blocks until completed

        # Remove drawings where hand is by painting the hand area black
        annotation_image = cv2.copyTo(annotation_image, skincolor_mask)
        stopwatch.logandrestart("Draw hand black on annotation_image")

        if config.show_debug_windows:
            contours = matlike_utils.find_contours(skincolor_mask)
            skincolor_mask = matlike_utils.ensure_rgb(skincolor_mask)  # Convert to rgb for drawing
            matlike_utils.draw_contours(
                skincolor_mask,
                contours,
                make_tuple_color(config.fingertip_contour_color),
                config.fingertip_visual_line_width_big,
            )
            skincolor_hand_display.show(skincolor_mask)

    # Draw area in which the hand can be recognized
    if config.show_camera_bounding:
        drawing.rect(
            annotation_image,
            (0, 0),
            (recorded_image_bgr.shape[1], recorded_image_bgr.shape[0]),
            make_tuple_color(config.camera_bounding_box_color),
            config.camera_bounding_box_thickness,
        )

    # May draw all landmarks
    if landmarks and config.show_mediapipe_points:
        mediapipe_color = make_tuple_color(config.mediapipe_color)
        for finger in landmarks.fingers:
            for mark in finger.all:
                drawing.point(
                    annotation_image,
                    mark,
                    mediapipe_color,
                    config.mediapipe_point_size,
                )
            drawing.draw_line(
                annotation_image,
                landmarks.wrist,
                finger.basejoint,
                mediapipe_color,
                config.mediapipe_line_size,
            )
            drawing.point(
                annotation_image,
                landmarks.wrist,
                mediapipe_color,
                config.mediapipe_point_size,
            )
            for a, b in enumerate_pairwise(finger.all):
                drawing.draw_line(
                    annotation_image,
                    a,
                    b,
                    mediapipe_color,
                    config.mediapipe_line_size,
                )

        if pointing_finger:
            for mark in pointing_finger.all:
                drawing.point(
                    annotation_image,
                    mark,
                    mediapipe_color,
                    config.mediapipe_highlight_point_size,
                )

    # Create the projection drawcall for the hovered word
    if tip_data and selected_word_detection and config.text_projection_enabled:
        text = get_translation(selected_word_detection)

        # Project the text on the finger
        if config.projection_target == ProjectionTarget.FINGER:
            skin_projection.project_text(
                annotation_image,
                text,
                Tuple2(transform_position_cutout_to_base(tip_data.fingertip_center)),
                Tuple2(pointing_finger.tip) - Tuple2(pointing_finger.basejoint),
                Tuple2(
                    config.finger_vertical_projection_offset_mm,
                    config.finger_horizontal_projection_offset_mm
                    * (1 if landmarks.is_right_hand else -1),
                )
                * pixel_per_mm,
                landmarks.is_right_hand,
                config.finger_textflow_mode,
                config.finger_text_font_size,
            )
        elif config.projection_target == ProjectionTarget.HAND:
            skin_projection.project_text(
                annotation_image,
                text,
                (landmarks.middle.basejoint + landmarks.wrist) * 0.5,
                Tuple2(0, 1),
                Tuple2(
                    config.hand_vertical_projection_offset_mm,
                    config.hand_horizontal_projection_offset_mm,
                )
                * pixel_per_mm,
                True,
                config.hand_textflow_mode,
                config.hand_text_font_size,
            )
        elif config.projection_target == ProjectionTarget.PAPER:
            if config.paper_always_reset:
                paper_projection.reset()

            paper_projection.project_text(
                annotation_image, text, recorded_image_bgr, selected_word_detection
            )

        stopwatch.logandrestart("Added highlight draw call")
    else:
        paper_projection.reset()

    # May draw calibration points
    if config.show_calibration_points:
        for point in calibration.calibration_points_screen:
            drawing.point(annotation_image, point, YELLOW, 8)

        def make_drawcall(a, b):
            drawing.draw_line(annotation_image, to_int_tuple(a), to_int_tuple(b), YELLOW, 2)

        for a, b in helper.enumerate_pairwise(calibration.calibration_points_screen):
            make_drawcall(a, b)

        drawing.point(annotation_image, mouse_cursor_position, RED, 5)
        stopwatch.logandrestart("Show calibration points")

    if config.show_debug_windows:
        if config.draw_annotations_on_recording_window:
            # Apply the annotation_image on the recorded image
            mask = (annotation_image != [0, 0, 0]).any(axis=-1)
            h, w = annotation_image.shape[:2]
            recorded_image_bgr[:h, :w][mask] = annotation_image[mask]
            stopwatch.logandrestart("Draw annotations on recorded image")

        # Show the recorded image including draw calls
        base_display.show(recorded_image_bgr)
        stopwatch.logandrestart("Show annotated recorded image")

    # Show projection
    if monitor_watcher.current_monitor_count == 3:
        # Get projection screen size
        width, height = projection_display.get_screen_size()

        # Create empty 4k image
        if empty_projection_image is None:
            empty_projection_image = create_empty_image((width, height, 3), BLACK)
            output_projection_image = np.empty(
                (height, width, 3), dtype=empty_projection_image.dtype
            )
        projection_image = empty_projection_image.copy()

        stopwatch.logandrestart("Projection image: Created")

        if config.draw_annotations_on_projection_window:
            # Draw annotation image
            h, w = annotation_image.shape[:2]
            projection_image[:h, :w] = annotation_image
            stopwatch.logandrestart("Drawn annotations on projection image")

        # Apply transform
        cv2.warpPerspective(
            projection_image,
            calibration.calibration_transform,
            (width, height),
            dst=output_projection_image,
        )
        stopwatch.logandrestart("Projection image: Warped")

        projection_display.show(output_projection_image)
        stopwatch.logandrestart("Projection image: Displayed")

    key = cv2.waitKey(1)
    # if key == ord("r"):
    #     # Press r to reset the reference image
    #     stabilizer.reference_image = None
    if key == ord("q"):
        # Press q to quit
        break
    stopwatch.logandrestart("Wait key")

    if config.freeze_camera:
        # Write frozen image to file at first time - for reloading after restart
        if previous_recorded_image_bgr is None:
            cv2.imwrite(config.frozen_camera_image_path, recorded_image_bgr)
        previous_recorded_image_bgr = recorded_image_bgr
    else:
        previous_recorded_image_bgr = None

    stopwatch.logandrestart("Finished loop")

    stopwatch_total.log("ONE LOOP")

for provider in image_getter_by_mode.values():
    del provider
