from typing import Optional
import cv2
import mediapipe as mp
from LandmarkedHand import LandmarkedHand
from config import Config
import matlike_utils


class MediapipeLandmarksRetriever:
    """A wrapper for convenient usage of mediapipe."""

    def __init__(self, config: Config):
        self.config = config
        self.update_complexcity(1 if self.config.mediapipe_use_complex_model else 0)

    def update_complexcity(self, complexity: int):
        self.complexity = complexity
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=self.complexity,
        )

    def find_hand_landmarks(
        self, image: cv2.typing.MatLike, color_conversion: Optional[int]
    ) -> LandmarkedHand:
        """Returns the position of finger and hand landmarks in absolute coordinate.
        Requires RGB image after the color conversion.
        Needs 13 ms for simple model, 23 for complex.
        """

        reduced_image = matlike_utils.scale_image(
            image,
            self.config.mediaipipe_image_smallering_factor,
            self.config.mediapipe_smallering_scaling_mode,
        )

        if color_conversion:
            image = cv2.cvtColor(reduced_image, color_conversion)

        self.hands.min_detection_confidence = self.config.min_mediapipe_confidence
        self.hands.min_tracking_confidence = self.config.min_mediapipe_confidence

        # May update complecxity
        new_complexity = 1 if self.config.mediapipe_use_complex_model else 0
        if new_complexity != self.complexity:
            self.hands.close()
            self.update_complexcity(new_complexity)

        # Process the image with Mediapipe
        results = self.hands.process(image)

        if not results.multi_hand_landmarks:
            return None  # No hands detected

        best_hand_index, best_score = max(
            enumerate(results.multi_handedness), key=lambda item: item[1].classification[0].score
        )

        # print(f"Hand index: {best_hand_index}, Score: {best_score.classification[0].score:.1f}")

        # if len(results.multi_handedness) > 1:
        #     print(
        #         f"Multiple hands detected. Confidences: {(100 * results.multi_handedness[0].classification[0].score):.1f} % vs. {(100 * results.multi_handedness[1].classification[0].score):.1f} %"
        #     )

        return LandmarkedHand(
            results.multi_hand_landmarks[best_hand_index],
            results.multi_handedness[best_hand_index],
            reduced_image.shape[1] / self.config.mediaipipe_image_smallering_factor,
            reduced_image.shape[0] / self.config.mediaipipe_image_smallering_factor,
        )
