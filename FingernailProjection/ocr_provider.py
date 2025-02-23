from google_ocr_data import GoogleOcrData
from text_detection import TextDetection
from config import Config
from cv2.typing import MatLike
from helper import picke_load_or_create


class FileOcrProvider:
    """OCR result is obtained from a file."""

    def __init__(self, config: Config, pickle_path: str):
        self.config = config
        self.pickle_path = pickle_path
        self.data: GoogleOcrData = None

    def get_text_detections(self, image: MatLike) -> list[TextDetection]:
        # Recreate OCR response if pickle file was deleted
        self.data = picke_load_or_create(
            self.pickle_path,
            lambda: GoogleOcrData.from_image(image, self.config),
        )

        return self.data.word_detections
