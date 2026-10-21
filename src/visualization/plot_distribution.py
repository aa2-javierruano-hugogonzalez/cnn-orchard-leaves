from collections import Counter
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from src.data.make_dataset import RANDOM_SEED, split_dataset_by_class

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "leafscans-orchard-compressed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_NAME = "class_distribution"
RASTER_DPI = 300

FIGURE_SIZE = (5.95, 3.4)
BAR_WIDTH = 0.55
X_MARGIN = 0.7
SEGMENT_GAP = 1.5
INNER_LABEL_PADDING = 2
CHIP_SPACING = 0.15
GAP_ABOVE_BAR = 4
CHIP_ROW_HEIGHT = 13
Y_HEADROOM = 1.15
Y_TICK_STEP = 100
SEGMENT_LABEL_SIZE = 8
TOTAL_LABEL_SIZE = 8.5

SURFACE = "#FFFFFF"
INK = "#1A1A1A"
MUTED_INK = "#666666"
GRID_GRAY = "#E5E5E5"

SPLIT_STYLES = (
    ("Entrenamiento", "#5C8BB5"),
    ("Validación", "#D88B9E"),
    ("Prueba", "#F6B352"),
)

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
    "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",     
    "font.size": 9,              
    "axes.labelsize": 9,
    "xtick.labelsize": 7,       
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED_INK,
    "axes.linewidth": 0.8,        
    "xtick.color": INK,
    "ytick.color": MUTED_INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def count_images_per_split(data_dir):
    train_samples, validation_samples, test_samples, class_names = split_dataset_by_class(data_dir, seed=RANDOM_SEED)

    split_counts = []
    for samples in (train_samples, validation_samples, test_samples):
        images_per_label = Counter(label for _, label in samples)
        split_counts.append([images_per_label[label] for label in range(len(class_names))])

    return class_names, split_counts


def to_spanish_name(class_name):
    return SPANISH_CLASS_NAMES[class_name.split(" (")[0]]


def relative_luminance(color):
    linear_channels = [channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
                       for channel in to_rgb(color)]
    return 0.2126 * linear_channels[0] + 0.7152 * linear_channels[1] + 0.0722 * linear_channels[2]


def contrast_ratio(first_color, second_color):
    lighter, darker = sorted((relative_luminance(first_color), relative_luminance(second_color)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def pick_text_color(fill_color):
    return max((SURFACE, INK), key=lambda text_color: contrast_ratio(fill_color, text_color))


def build_legend_labels(split_counts):
    split_totals = [sum(counts) for counts in split_counts]
    grand_total = sum(split_totals)
    return [f"{split_name} ({round(100 * split_total / grand_total)}\u00a0%)"
            for (split_name, _), split_total in zip(SPLIT_STYLES, split_totals)]


def draw_stacked_bars(axis, split_counts):
    positions = list(range(len(split_counts[0])))
    bottoms = [0] * len(positions)

    split_bars = []
    for (_, color), counts in zip(SPLIT_STYLES, split_counts):
        bars = axis.bar(positions, counts, width=BAR_WIDTH, bottom=bottoms, color=color,
                        edgecolor=SURFACE, linewidth=SEGMENT_GAP, zorder=3)
        split_bars.append(bars)
        bottoms = [bottom + count for bottom, count in zip(bottoms, counts)]

    return split_bars, bottoms


def style_axis(axis, class_labels, totals):
    axis.set_xticks(range(len(class_labels)))
    axis.set_xticklabels([class_label.replace(" ", "\n") for class_label in class_labels])
    axis.set_xlim(-X_MARGIN, len(class_labels) - 1 + X_MARGIN)
    axis.set_ylim(0, max(totals) * Y_HEADROOM)
    axis.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEP))

    axis.set_xlabel("Especie", labelpad=6)
    axis.set_ylabel("Número de imágenes", labelpad=6)

    highest_tick = max(tick for tick in axis.get_yticks() if tick <= max(totals))
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["left"].set_bounds(0, highest_tick)
    axis.spines["bottom"].set_zorder(4)
    axis.tick_params(axis="x", length=0, pad=5)
    axis.tick_params(axis="y", length=3, width=0.7)

    axis.yaxis.grid(True, linestyle=":", linewidth=1.0, color=GRID_GRAY)
    axis.xaxis.grid(False)
    axis.set_axisbelow(True)


def add_legend(axis, legend_labels):
    handles = [Patch(facecolor=color, edgecolor="none", label=legend_label)
               for (_, color), legend_label in zip(SPLIT_STYLES, legend_labels)]
    axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.00), ncol=len(handles),
                frameon=False, handlelength=1.2, handleheight=0.9, columnspacing=1.8, borderaxespad=0.3)


def fits_inside(label, bar, renderer):
    label_box = label.get_window_extent(renderer)
    bar_box = bar.get_window_extent(renderer)
    padding = renderer.points_to_pixels(INNER_LABEL_PADDING)
    return label_box.width + 2 * padding <= bar_box.width and label_box.height + 2 * padding <= bar_box.height


def label_segments(axis, split_bars, split_counts, renderer):
    segments_without_room = [[] for _ in split_counts[0]]

    for (_, color), bars, counts in zip(SPLIT_STYLES, split_bars, split_counts):
        for position, (bar, count) in enumerate(zip(bars, counts)):
            label = axis.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(count),
                              ha="center", va="center", fontsize=SEGMENT_LABEL_SIZE,
                              color=pick_text_color(color), zorder=4)
            if not fits_inside(label, bar, renderer):
                label.remove()
                segments_without_room[position].append((count, color))

    return segments_without_room


def label_small_segments(axis, segments_without_room, totals):
    for position, segments in enumerate(segments_without_room):
        offsets = [CHIP_SPACING * (2 * index - len(segments) + 1) for index in range(len(segments))]
        for (count, color), offset in zip(segments, offsets):
            axis.annotate(str(count), xy=(position + offset, totals[position]), xytext=(0, GAP_ABOVE_BAR),
                          textcoords="offset points", ha="center", va="bottom", fontsize=SEGMENT_LABEL_SIZE,
                          color=pick_text_color(color), zorder=4,
                          bbox={"boxstyle": "round,pad=0.25", "facecolor": color, "edgecolor": "none"})


def label_totals(axis, segments_without_room, totals):
    for position, total in enumerate(totals):
        lift = GAP_ABOVE_BAR + (CHIP_ROW_HEIGHT if segments_without_room[position] else 0)
        axis.annotate(str(total), xy=(position, total), xytext=(0, lift), textcoords="offset points",
                      ha="center", va="bottom", fontsize=TOTAL_LABEL_SIZE, fontweight="bold", color=INK, zorder=4,
                      bbox={"boxstyle": "square,pad=0.1", "facecolor": SURFACE, "edgecolor": "none"})


def build_figure(class_labels, split_counts):
    figure = Figure(figsize=FIGURE_SIZE)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot()

    split_bars, totals = draw_stacked_bars(axis, split_counts)
    style_axis(axis, class_labels, totals)
    add_legend(axis, build_legend_labels(split_counts))

    segments_without_room = label_segments(axis, split_bars, split_counts, canvas.get_renderer())
    label_small_segments(axis, segments_without_room, totals)
    label_totals(axis, segments_without_room, totals)

    return figure


def save_figure(figure, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{FIGURE_NAME}.pdf"
    png_path = output_dir / f"{FIGURE_NAME}.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=RASTER_DPI)
    return pdf_path, png_path


def print_split_table(class_labels, split_counts):
    split_names = [split_name for split_name, _ in SPLIT_STYLES]
    print(f"{'Especie':<17}" + "".join(f"{split_name:>15}" for split_name in split_names) + f"{'Total':>8}")

    for position, class_label in enumerate(class_labels):
        counts = [counts_per_split[position] for counts_per_split in split_counts]
        print(f"{class_label:<17}" + "".join(f"{count:>15}" for count in counts) + f"{sum(counts):>8}")

    split_totals = [sum(counts) for counts in split_counts]
    print(f"{'Total':<17}" + "".join(f"{total:>15}" for total in split_totals) + f"{sum(split_totals):>8}")


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_DIR}. Download it before plotting the distribution.")

    class_names, split_counts = count_images_per_split(DATA_DIR)
    class_labels = [to_spanish_name(class_name) for class_name in class_names]
    print_split_table(class_labels, split_counts)

    with matplotlib.rc_context(PLOT_STYLE):
        figure = build_figure(class_labels, split_counts)
        output_paths = save_figure(figure, OUTPUT_DIR)

    for output_path in output_paths:
        print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()