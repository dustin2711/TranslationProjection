from dataclasses import dataclass
from typing import List, Tuple
import cv2
from google.cloud import vision
from axis_bounding_box import AxisBoundingBox, QuadBoundingBox
from config import Config
from stopwatch import Stopwatch
from text_detection import TextDetection
from cv2.typing import MatLike
from dataclasses import dataclass, field
from google.cloud.vision_v1.types.text_annotation import Word
from enums import TranslationDirection


@dataclass
class GoogleOcrData:
    """Contains the preprocessed OCR data from Google Cloud Vision API."""

    def __repr__(self):
        return f"<GoogleOcrData: {len(self.data)} items>"

    paragraph_detections: list[TextDetection] = field(default_factory=list)
    word_detections: list[TextDetection] = field(default_factory=list)
    symbol_detections: list[TextDetection] = field(default_factory=list)

    def from_image(input_image: MatLike, config: Config) -> "GoogleOcrData":
        """
        This method will take some time as it may involve deseriliazing the Google response.
        Takes 17 ms for 1 Manga page.
        """
        stopwatch = Stopwatch(logging_enabled=config.log_google_ocr)
        stopwatch.restart()

        # Encode the OpenCV image to bytes in PNG format
        _, encoded_image = cv2.imencode(".png", input_image)
        content = encoded_image.tobytes()

        stopwatch.log("Encoded image")

        # Send the image to Google Cloud Vision API
        image = vision.Image(content=content)
        client = vision.ImageAnnotatorClient()
        print("Sending image to Google Cloud Vision API...")
        google_response = client.document_text_detection(
            image=image,
            image_context={
                "language_hints": [
                    (
                        "ja"
                        if (
                            config.translation_direction == TranslationDirection.JAPANESE_TO_ENGLISH
                        )
                        else "en"
                    )
                ]
            },
        )
        stopwatch.log("Got Google response")

        data = GoogleOcrData()
        for page in google_response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    GoogleOcrData.add_detections_from_paragraph(data, config, paragraph)

        stopwatch.log("Converted to TextDetections")

        return data

    def bounding_box_from_vertices(vertices):
        return QuadBoundingBox([(vertex.x, vertex.y) for vertex in vertices])

    def add_detections_from_paragraph(data: "GoogleOcrData", config: Config, paragraph):
        if config.include_paragraphs:
            paragraph_detection = TextDetection(
                None,
                GoogleOcrData.bounding_box_from_vertices(paragraph.bounding_box.vertices),
                paragraph.confidence,
            )
            data.paragraph_detections.append(paragraph_detection)
        else:
            paragraph_detection = None

        word: Word
        for word in paragraph.words:
            if word.confidence < config.min_detection_confidence:
                continue

            word_detection = TextDetection(
                None,
                GoogleOcrData.bounding_box_from_vertices(word.bounding_box.vertices),
                word.confidence,
                paragraph_detection,
            )
            data.word_detections.append(word_detection)

            if paragraph_detection:
                paragraph_detection.children.append(word_detection)

            if config.include_symbols:
                for symbol in word.symbols:
                    if symbol.confidence < config.min_detection_confidence:
                        continue

                    box = GoogleOcrData.bounding_box_from_vertices(symbol.bounding_box.vertices)

                    if box.area < config.min_detection_text_size:
                        continue

                    symbol_detection = TextDetection(
                        symbol.text,
                        box,
                        symbol.confidence,
                        word_detection,
                    )
                    data.symbol_detections.append(symbol_detection)
                    word_detection.children.append(symbol_detection)

                word_detection.text = "".join(
                    [symbol_detection.text for symbol_detection in word_detection.children]
                )
            else:
                word_detection.text = "".join([symbol.text for symbol in word.symbols])

        if paragraph_detection:
            paragraph_detection.text = " ".join(
                [detection.text for detection in paragraph_detection.children]
            )
