import cv2
from config import Config
from abc import ABC, abstractmethod
from cv2.typing import MatLike


class ImageProvider(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def get_image(self) -> MatLike:
        pass


class FileImageProvider(ImageProvider):
    """Image is obtained from a file."""

    def __init__(self, config: Config, image_path: str):
        super().__init__(config)
        self.image = cv2.imread(image_path)

    def get_image(self):
        return self.image.copy()
