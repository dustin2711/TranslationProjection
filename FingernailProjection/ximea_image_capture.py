from enums import Rotation
from config import Config
from image_provider import ImageProvider
from matlike_utils import increase_brightness, rotate_image
from ximea import xiapi
from cv2.typing import MatLike
import cv2
import numpy as np


class XimeaImageCapture(ImageProvider):

    def __init__(self, config: Config):
        self.config = config
        self.ximea_image = xiapi.Image()
        self.camera = xiapi.Camera()
        self.camera.open_device()
        self.camera.start_acquisition()

        print(f"Started camera.")

    def get_image(self) -> MatLike:
        """Returns the current image."""
        # Update exposure if needed
        if self.camera.get_exposure() != self.config.exposure_time:
            self.camera.set_exposure(self.config.exposure_time)

        self.camera.get_image(self.ximea_image)

        image = np.frombuffer(self.ximea_image.get_image_data_raw(), dtype=np.uint8).reshape(
            self.ximea_image.height, self.ximea_image.width, 1
        )

        image = rotate_image(image, self.config.rotation)
        image = increase_brightness(image, self.config.brightness_scale_factor)

        return image

    def __del__(self):
        self.camera.stop_acquisition()
        self.camera.close_device()
        print(f"Closed camera.")
