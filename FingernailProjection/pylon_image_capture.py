# from logging import config
# from enums import Rotation
# from image_provider import ImageProvider
# from cv2.typing import MatLike
# from pypylon import pylon
# from matlike_utils import rotate_image
# import numpy as np
# from config import Config


# class PylonImageCapture(ImageProvider):
#     def __init__(self, config: Config = None):
#         # Initialize and configure the camera
#         self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
#         self.camera.Open()
#         self.camera.PixelFormat.SetValue("BGR8")

#         # Set automatic brightness
#         self.camera.ExposureAuto.SetValue("Continuous")

#         # Frame rate control (disable forcing frame rate for maximum flexibility)
#         self.camera.AcquisitionFrameRateEnable.SetValue(False)

#         self.camera.OutputQueueSize.SetValue(1)  # Keep only the latest frame

#         # Start grabbing images
#         self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

#     def get_image(self) -> MatLike:
#         """Grabs the latest frame from the camera."""
#         try:
#             result = self.camera.RetrieveResult(
#                 5000, pylon.TimeoutHandling_Return
#             )  # 5000ms timeout
#             if result.GrabSucceeded():
#                 image = result.Array
#                 # Apply transformations
#                 image = rotate_image(image, Rotation.Rot90)  # Default to no rotation
#             else:
#                 print(f"Error grabbing image: {result.ErrorCode} {result.ErrorDescription}")
#                 image = None
#             result.Release()
#         except Exception as e:
#             print(f"Pylon error: {e}")
#             image = None

#         return image

#     def __del__(self):
#         if self.camera.IsGrabbing():
#             self.camera.StopGrabbing()
#         self.camera.Close()


############################

from config import Config
from image_provider import ImageProvider
from cv2.typing import MatLike
from pypylon import pylon
from matlike_utils import *


class PylonImageCapture(ImageProvider):
    def __init__(self, config: Config):
        self.config = config

        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        self.camera.PixelFormat.SetValue("BGR8")

        # Enable frame rate control
        self.camera.AcquisitionFrameRateEnable.SetValue(True)
        self.camera.OutputQueueSize.SetValue(1)  # Only keep the most recent frame

        # Start grabbing images
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self.camera.AcquisitionFrameRateEnable.SetValue(False)

    def _crop(self, factor: float):
        self.camera.Height = int(factor * 1296)
        self.camera.Width = int(factor * 2592)

    def get_image(self) -> MatLike:
        """Returns a BGR-image."""

        if self.camera.ExposureTime.GetValue() != self.config.exposure_time:

            if self.config.exposure_time == 0:
                # Set automatic mode
                if self.camera.ExposureAuto.GetValue() != "Continuous":
                    self.camera.ExposureAuto.SetValue("Continuous")
            else:
                self.camera.ExposureTime.SetValue(self.config.exposure_time)
                # Set manual mode
                if self.camera.ExposureAuto.GetValue() != "Off":
                    self.camera.ExposureAuto.SetValue("Off")

        if self.config.use_manual_color_balance:
            self.camera.BalanceWhiteAuto.SetValue("Off")  # Disable auto white balancing
            # Adjust the balance ratios for each channel
            self.camera.BalanceRatioSelector.SetValue("Red")
            self.camera.BalanceRatio.SetValue(self.config.red_balance)
            self.camera.BalanceRatioSelector.SetValue("Blue")
            self.camera.BalanceRatio.SetValue(self.config.blue_balance)
            self.camera.BalanceRatioSelector.SetValue("Green")
            self.camera.BalanceRatio.SetValue(1)
        else:
            self.camera.BalanceWhiteAuto.SetValue("Continuous")  # Disable auto white balancing

        try:
            # stopwatch.start()
            result = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        except pylon.GenericException as e:
            print(f"Pylon error: {e}")
            return None

        if result.GrabSucceeded():
            image = result.Array
            image = rotate_image(image, self.config.image_rotation)
            image = increase_brightness(image, self.config.brightness_scale_factor)
            # stopwatch.log("Grabbing time")
        else:
            print(f"Error: {result.ErrorCode} {result.ErrorDescription}")
            image = None

        result.Release()
        return image

    def __del__(self):
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        self.camera.Close()
