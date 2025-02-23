import matplotlib.pyplot as plt
import colorsys


def get_color_palette20(color_count: int) -> list[tuple[int, int, int]]:
    """Uses a categorical colormap like 'tab20' for well-separated distinct colors."""
    cmap = plt.get_cmap("tab20b")  # 'tab20' has up to 20 well-separated colors
    if color_count <= cmap.N:
        colors = [cmap(i)[:3] for i in range(color_count)]
    else:
        # Repeat palette if more colors are requested than available
        colors = [cmap(i % cmap.N)[:3] for i in range(color_count)]
    # Convert to RGB in the range of 0-255
    return [tuple(int(c * 255) for c in color) for color in colors]


palette = get_color_palette20(20)


def get_color_from_palette20(index: int):
    """Gets a color from a palette with 20 colors, repeating if necessary."""
    return palette[index % 20]


def change_color(
    rgb_color: tuple[int, int, int],
    hue: float = None,
    saturation: float = None,
    value: float = None,
) -> tuple[int, int, int]:
    h, s, v = colorsys.rgb_to_hsv(*rgb_color)
    h = hue if hue else h
    s = saturation if saturation else s
    v = value if value else v
    return colorsys.hsv_to_rgb(h, s, v)
