import statistics
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "leafscans-orchard-compressed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_NAME = "letterbox_comparison"
RASTER_DPI = 300
IMAGE_EXTENSION = ".tif"

TARGET_SIZE = 128
DISPLAY_LONG_SIDE = 600

CELL_SIZE = 0.7
ORIGINAL_ROW_HEIGHT = 1.0
CELL_GAP = 0.05
ROW_LABEL_WIDTH = 1.0
TITLE_HEIGHT = 0.38
LABEL_GAP = 0.08
ROW_LABEL_WRAP_WIDTH = 15
COLUMN_TITLE_WRAP_WIDTH = 10

INK = "#222222"
FRAME_COLOR = "#8C8C8C"
FRAME_WIDTH = 0.6

ROW_LABELS = ("Original", "Redimensionado directo", "Letterbox (Relleno)")
ROW_INTERPOLATIONS = ("antialiased", "nearest", "nearest")

SPANISH_CLASS_NAMES = {
    "Apple": "Manzano",
    "Apricot": "Albaricoquero",
    "European Plum": "Ciruelo europeo",
    "Peach": "Melocotonero",
    "Pear": "Peral",
    "Sour Cherry": "Guindo",
    "Sweet Cherry": "Cerezo dulce",
}

PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "CMU Serif", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 9,
    "text.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def list_class_folders(data_dir):
    return sorted(folder for folder in data_dir.iterdir() if folder.is_dir())


def to_spanish_name(class_folder):
    return SPANISH_CLASS_NAMES[class_folder.name.split(" (")[0]]


def measure_aspect_ratio(image_path):
    with Image.open(image_path) as image:
        width, height = image.size
    return max(width, height) / min(width, height)


def pick_representative_image(class_folder):
    image_paths = sorted(path for path in class_folder.iterdir() if path.suffix.lower() == IMAGE_EXTENSION)
    if not image_paths:
        raise FileNotFoundError(f"No {IMAGE_EXTENSION} images found in {class_folder}")

    aspect_ratios = {image_path: measure_aspect_ratio(image_path) for image_path in image_paths}
    median_ratio = statistics.median(aspect_ratios.values())
    return min(image_paths, key=lambda image_path: abs(aspect_ratios[image_path] - median_ratio))


def load_rgb_image(image_path):
    with Image.open(image_path) as image:
        return image.convert("RGB")


def resize_directly(image, size):
    return image.resize((size, size), Image.Resampling.BILINEAR)


def estimate_background_color(image):
    width, height = image.size
    corners = [
        image.getpixel((0, 0)),
        image.getpixel((width - 1, 0)),
        image.getpixel((0, height - 1)),
        image.getpixel((width - 1, height - 1)),
    ]
    return tuple(int(value) for value in np.median(corners, axis=0))


def letterbox_resize(image, size):
    width, height = image.size
    scale = size / max(width, height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized_image = image.resize(resized_size, Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (size, size), estimate_background_color(resized_image))
    offset = ((size - resized_size[0]) // 2, (size - resized_size[1]) // 2)
    canvas.paste(resized_image, offset)
    return canvas


def shrink_for_display(image):
    preview = image.copy()
    preview.thumbnail((DISPLAY_LONG_SIDE, DISPLAY_LONG_SIDE), Image.Resampling.LANCZOS)
    return preview


def build_panels(data_dir):
    panels = []
    for class_folder in list_class_folders(data_dir):
        image_path = pick_representative_image(class_folder)
        image = load_rgb_image(image_path)
        processed_images = (
            shrink_for_display(image),
            resize_directly(image, TARGET_SIZE),
            letterbox_resize(image, TARGET_SIZE),
        )
        panels.append((to_spanish_name(class_folder), image_path, processed_images))
    return panels


def wrap_label(text, width):
    return textwrap.fill(text, width, break_long_words=False, break_on_hyphens=False)


def compute_layout(num_columns):
    row_heights = (ORIGINAL_ROW_HEIGHT, CELL_SIZE, CELL_SIZE)
    figure_width = ROW_LABEL_WIDTH + num_columns * CELL_SIZE + CELL_GAP * (num_columns - 1)
    figure_height = TITLE_HEIGHT + sum(row_heights) + CELL_GAP * (len(row_heights) - 1)

    row_bottoms = []
    next_top = figure_height - TITLE_HEIGHT
    for row_height in row_heights:
        row_bottoms.append(next_top - row_height)
        next_top -= row_height + CELL_GAP

    column_lefts = [ROW_LABEL_WIDTH + column * (CELL_SIZE + CELL_GAP) for column in range(num_columns)]
    return figure_width, figure_height, row_heights, row_bottoms, column_lefts


def draw_image(axis, image, interpolation):
    axis.imshow(np.asarray(image), interpolation=interpolation)
    axis.add_patch(Rectangle((-0.5, -0.5), image.width, image.height, fill=False,
                             edgecolor=FRAME_COLOR, linewidth=FRAME_WIDTH, clip_on=False))
    axis.axis("off")


def build_figure(panels):
    figure_width, figure_height, row_heights, row_bottoms, column_lefts = compute_layout(len(panels))
    figure = Figure(figsize=(figure_width, figure_height))

    title_bottom = (figure_height - TITLE_HEIGHT + LABEL_GAP) / figure_height
    for column_left, (class_label, _, images) in zip(column_lefts, panels):
        for row_bottom, row_height, image, interpolation in zip(row_bottoms, row_heights, images,
                                                                ROW_INTERPOLATIONS):
            axis = figure.add_axes((column_left / figure_width, row_bottom / figure_height,
                                    CELL_SIZE / figure_width, row_height / figure_height))
            draw_image(axis, image, interpolation)

        figure.text((column_left + CELL_SIZE / 2) / figure_width, title_bottom,
                    wrap_label(class_label, COLUMN_TITLE_WRAP_WIDTH), ha="center", va="bottom",
                    multialignment="center")

    for row_bottom, row_height, row_label in zip(row_bottoms, row_heights, ROW_LABELS):
        figure.text((ROW_LABEL_WIDTH - LABEL_GAP) / figure_width, (row_bottom + row_height / 2) / figure_height,
                    wrap_label(row_label, ROW_LABEL_WRAP_WIDTH), ha="right", va="center", multialignment="right")

    return figure


def save_figure(figure, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{FIGURE_NAME}.pdf"
    png_path = output_dir / f"{FIGURE_NAME}.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=RASTER_DPI)
    return pdf_path, png_path


def print_selected_images(panels):
    for class_label, image_path, _ in panels:
        print(f"{class_label:<17}{image_path.name}")


def print_natural_size(png_path):
    with Image.open(png_path) as image:
        width, height = image.size
    print(f"Natural size: {width / RASTER_DPI:.2f} x {height / RASTER_DPI:.2f} in "
          f"(include it at width={width / RASTER_DPI:.2f}in to keep the text at its design size)")


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_DIR}. Download it before plotting the comparison.")

    panels = build_panels(DATA_DIR)
    print_selected_images(panels)

    with matplotlib.rc_context(PLOT_STYLE):
        figure = build_figure(panels)
        pdf_path, png_path = save_figure(figure, OUTPUT_DIR)

    print(f"Figure saved to {pdf_path}")
    print(f"Figure saved to {png_path}")
    print_natural_size(png_path)


if __name__ == "__main__":
    main()