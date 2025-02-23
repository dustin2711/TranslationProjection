from dataclasses import dataclass
from hand_contour_detector import *
from matlike_utils import *


@dataclass
class OcrPreprocessData:
    source_image: np.array
    sharpened_image: np.array
    grayscale_image: np.array
    scaled_image: np.array
    thresholded_image: np.array


def preprocess_image_for_ocr(image: np.array, config):

    data = OcrPreprocessData()

    # Sharpen
    data.sharpened_image = matlike_utils.sharpen_image(image, config.sharpening_strength)

    # Make grayscale
    data.grayscale_image = matlike_utils.ensure_grayscale(data.sharpened_image)

    # Scale
    data.scaled_image = matlike_utils.scale_image(
        data.grayscale_image, config.text_scale, ScalingMode.Cubic
    )

    # Threshold image
    match config.text_threshold_mode:
        case ThresholdMode.OTSU:
            data.thresholded_image = apply_otsu_threshold(data.scaled_image)
        case ThresholdMode.MANUAL:
            data.thresholded_image = apply_threshold(
                data.scaled_image, config.text_threshold_manual_value
            )
        case ThresholdMode.ADAPTIVE:
            data.thresholded_image = apply_adaptive_threshold(
                data.scaled_image,
                config.text_threshold_adaptive_kernel_radius,
                config.text_threshold_adaptive_value,
            )

    data.source_image = matlike_utils.scale_image(image, config.text_scale, ScalingMode.Cubic)
    data.sharpened_image = (
        matlike_utils.scale_image(data.sharpened_image, config.text_scale, ScalingMode.Cubic),
    )

    return data
