import ctypes
from dataclasses import dataclass
from enum import Enum
import cv2
from cv2.typing import MatLike
from cv_loop import CvLoop
from enums import ScaleMode
from matlike_utils import Anchor
from screeninfo import get_monitors
from helper import *
from typing import Tuple
import matlike_utils
from tkinter.filedialog import asksaveasfilename

monitors = get_monitors()
monitor_count = len(monitors)
moved_window_names: list[str] = []


ASK_INDIVIDUAL_FILENAME = True
EXTENSION = ".jpg"
SAVE_DIRECTORY = (
    r"C:\Users\sens\Desktop\FingernailProjection\FingernailProjection\prototyping\hand_dataset"
)


@dataclass
class ImageDisplay:
    windowname: str
    anchor: Anchor = Anchor.TOP_LEFT
    screen_index: int | Callable[[], int] = 0
    scale_mode: ScaleMode | Callable[[], ScaleMode] = ScaleMode.NATIVE
    stick_at_position: bool = False
    offset: Tuple[int, int] = (0, 0)
    show_titlebar: bool = True
    mouse_event_callback: callable = None
    window_size = (1, 1)
    image = None

    def convert_screen_to_image_position(self, mouse_position: tuple[int, int]):
        height, width = self.image.shape[:2]
        return (
            mouse_position[0] * width / self.window_size[0],
            mouse_position[1] * height / self.window_size[1],
        )

    def screen_mouse_event_happened(self, event, x, y, flags, param):
        if self.mouse_event_callback:
            self.mouse_event_callback(event, x, y, flags, param)

        if event == cv2.EVENT_RBUTTONDOWN:
            # If 's' is pressed, open a save file dialog
            if ASK_INDIVIDUAL_FILENAME:
                # Ask for a save path
                path = asksaveasfilename(
                    title="Save Image As",
                    filetypes=[("Image Files", "*.jpg *.png"), ("All Files", "*.*")],
                    defaultextension=EXTENSION,
                    initialdir=SAVE_DIRECTORY,
                )
            else:
                path = CvLoop.get_next_filename(SAVE_DIRECTORY)

            # Save the image if a path is provided
            if path:
                cv2.imwrite(path, self.image)
                print(f"Image saved to {path}")

    def show(self, image: MatLike | list[MatLike]):
        """Returns the scale factor used for the image."""
        if image is None:
            return

        if isinstance(image, list):
            # Sort out None images
            image = [it for it in image if it is not None]

            if len(image) == 0:
                return

            # Merge images
            image = matlike_utils.merge_images_horizontally(image)

        height, width = image.shape[:2]

        self.image = image

        index = self.screen_index if isinstance(self.screen_index, int) else self.screen_index()
        if index < 0 or index >= monitor_count:
            return None

        monitor = monitors[index]
        scale_mode = (
            self.scale_mode if isinstance(self.scale_mode, ScaleMode) else self.scale_mode()
        )

        match scale_mode:
            case ScaleMode.NATIVE:
                self.window_size = (width, height)
            case ScaleMode.DOUBLE:
                self.window_size = (2 * width, 2 * height)
            case ScaleMode.QUADRUPLE:
                self.window_size = (4 * width, 4 * height)
            case ScaleMode.FILL:
                self.show_fullscreen(monitor)
                return
                # scaleHeight = monitor.height / height
                # scaleWidth = monitor.width / width
                # self.window_size = (int(scaleWidth * width), int(scaleHeight * height))
            # Scale to the target screen
            case ScaleMode.FILL_SCREEN_HEIGHT:
                scale = monitor.height / height
                self.window_size = (int(scale * width), int(scale * height))
            case ScaleMode.FILL_SCREEN_WIDTH:
                scale = monitor.width / width
                new_height = int(scale * height)
                # Ensure the new height doesn't exceed the screen's height
                if new_height > monitor.height:
                    scale = monitor.height / height
                self.window_size = (int(scale * width), int(scale * height))
            case ScaleMode.FILL_HALF_SCREEN_HEIGHT:
                scale = 0.5 * monitor.height / height
                self.window_size = (int(scale * width), int(scale * height))

        cv2.imshow(
            self.windowname,
            cv2.resize(
                image,
                self.window_size,
                interpolation=cv2.INTER_AREA if self.window_size[0] < width else cv2.INTER_LINEAR,
            ),
        )
        # cv2.resizeWindow(self.windowname, *self.window_size)

        # May hide the title bar
        if not self.show_titlebar:
            self.hide_titlebar()

        if self.stick_at_position or self.windowname not in moved_window_names:
            x, y = monitor.x + self.offset[0], monitor.y + self.offset[1]
            cv2.moveWindow(
                self.windowname,
                x if ("LEFT" in str(self.anchor)) else (x + monitor.width - self.window_size[0]),
                y if ("TOP" in str(self.anchor)) else (y + monitor.height - self.window_size[1]),
            )
            moved_window_names.append(self.windowname)

        cv2.setMouseCallback(self.windowname, self.screen_mouse_event_happened)

    def get_screen_size(self) -> Tuple[int, int]:
        """Returns the size of the screen specified by screen_index."""
        index = self.screen_index if isinstance(self.screen_index, int) else self.screen_index()
        if index < 0 or index >= monitor_count:
            return 0, 0
        screen = monitors[index]
        return screen.width, screen.height

    def hide_titlebar(self):
        if hwnd := ctypes.windll.user32.FindWindowW(None, self.windowname):
            # Get the HWND for the OpenCV window (Windows-specific)
            GWL_STYLE = -16
            WS_BORDER = 0x00800000
            WS_DLGFRAME = 0x00400000
            WS_CAPTION = WS_BORDER | WS_DLGFRAME
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style & ~WS_CAPTION)
            ctypes.windll.user32.SetWindowPos(
                hwnd, None, 0, 0, 0, 0, 0x27
            )  # 0x27 = SWP_SHOWWINDOW | SWP_NOSIZE | SWP_NOMOVE

    def show_fullscreen(self, monitor):
        x_offset, y_offset, screen_width, screen_height = (
            monitor.x,
            monitor.y,
            monitor.width,
            monitor.height,
        )

        self.window_size = (monitor.width, monitor.height)

        # Create a borderless window
        cv2.namedWindow(self.windowname, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.windowname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Move window to the desired screen
        cv2.moveWindow(self.windowname, x_offset, y_offset)
        cv2.resizeWindow(self.windowname, screen_width, screen_height)

        # Resize the image to match the screen size
        if self.image.shape[1] != screen_width or self.image.shape[0] != screen_height:
            self.image = cv2.resize(
                self.image, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR
            )

        cv2.imshow(self.windowname, self.image)

        cv2.setMouseCallback(self.windowname, self.screen_mouse_event_happened)
