import matplotlib.pyplot as plt
import cv2
import numpy as np


def plot_color(color: tuple, title: str = "Color"):
    # Normalize color to [0, 1] if needed (assumes input in [0, 255])
    if max(color) > 1.0:
        color = tuple(c / 255.0 for c in color)

    # Create a figure with a single square
    plt.figure(figsize=(2, 2))
    plt.imshow([[color]])  # Single square with the specified color
    plt.axis("off")  # Turn off axes
    plt.title(title, fontsize=14)
    plt.show()


def plot_histogram(histogram_1d, title):
    # Visualize as an image (2D heatmap)
    histogram_2d = histogram_1d.reshape(180, 256)
    plt.imshow(histogram_2d, extent=[0, 255, 0, 179], origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="Frequency Difference")
    plt.xlabel("Saturation")
    plt.ylabel("Hue")
    plt.title(title)
    plt.show()


def plot_rgb_image(rgb_image: np.ndarray, title: str = "Image", scale: float = 1) -> None:
    height, width, _ = rgb_image.shape

    # Calculate figure size in inches
    figsize_width = (width * scale) / 100
    figsize_height = (height * scale) / 100

    plt.figure(figsize=(figsize_width, figsize_height))
    plt.imshow(rgb_image)
    plt.axis("off")  # Hide axes
    plt.title(title)
    plt.show()


def plot_hsv_image(hsv_image: np.ndarray, title: str = "Image", scale: float = 1) -> None:
    plot_rgb_image(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB), title, scale)


def plot_bgr_image(hsv_image: np.ndarray, title: str = "Image", scale: float = 1) -> None:
    plot_rgb_image(cv2.cvtColor(hsv_image, cv2.COLOR_BGR2RGB), title, scale)


def plot_rgb_images_horizontally(
    images: list[np.ndarray], titles: list[str] = [], scale: float = 1
) -> None:
    """
    Plots a list of images with titles horizontally.
    Args:
        images: List of tuples containing the image (numpy array) and its title (string).
        scale: Scaling factor for image display size.
    """
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * scale * 5, 5))

    # Ensure axes is always iterable
    if num_images == 1:
        axes = [axes]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.axis("off")  # Hide axes
        ax.set_title(title)

    plt.tight_layout()
    plt.show()
