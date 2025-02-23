from tkinter.filedialog import asksaveasfilename
import cv2
import os

ASK_INDIVIDUAL_FILENAME = False
EXTENSION = ".jpg"
SAVE_DIRECTORY = r"C:\Users\sens\Desktop\FingernailProjection\Abstract"


class CvLoop:
    """Helps in looping using open cv. Supports exiting with q and screenshotting with s."""

    def __init__(self, step_function: callable, wait_key_milliseconds=1):
        """The step function returns the image to screenshot."""
        self.step_function = step_function
        self.wait_key_milliseconds = wait_key_milliseconds

    @staticmethod
    def get_next_filename(directory, base_name="Image", extension=EXTENSION):
        i = 1
        while os.path.exists(os.path.join(directory, f"{base_name}{i}{extension}")):
            i += 1
        return os.path.join(directory, f"{base_name}{i}{extension}")

    def start(self):
        while True:
            image = self.step_function()

            # Wait for key press
            key = cv2.waitKey(1)
            if key == ord("s"):
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
                    cv2.imwrite(path, image)
                    print(f"Image saved to {path}")

            # If 'q' is pressed, exit the loop
            elif key == ord("q"):
                print("Exiting...")
                break
