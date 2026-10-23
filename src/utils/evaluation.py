import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

matplotlib.use("Agg")

DARK_BLUE = "#1B2A6B"
DEEP_PURPLE = "#5B2A86"
BRIGHT_YELLOW = "#F5C518"
SUN_RED = "#E63946"
NEUTRAL_INK = "#222222"
GRID_GRAY = "#D9D9D9"

IEEE_SINGLE_COLUMN_WIDTH = 3.5
IEEE_DOUBLE_COLUMN_WIDTH = 7.16
RASTER_DPI = 600

IEEE_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": NEUTRAL_INK,
    "axes.labelcolor": NEUTRAL_INK,
    "xtick.color": NEUTRAL_INK,
    "ytick.color": NEUTRAL_INK,
    "text.color": NEUTRAL_INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID_GRAY,
    "grid.linewidth": 0.4,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

CONFUSION_COLORMAP = LinearSegmentedColormap.from_list(
    "orchard_sequential", ["#FFFFFF", BRIGHT_YELLOW, SUN_RED, DEEP_PURPLE, DARK_BLUE]
)


def evaluate_model(model, test_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    true_labels = []
    predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predicted_labels = outputs.argmax(dim=1)

            true_labels.extend(labels.tolist())
            predictions.extend(predicted_labels.cpu().tolist())

    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)
    per_class_precision = precision_score(true_labels, predictions, average=None, zero_division=0)
    per_class_recall = recall_score(true_labels, predictions, average=None, zero_division=0)
    test_confusion_matrix = confusion_matrix(true_labels, predictions)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": test_confusion_matrix,
        "true_labels": true_labels,
        "predictions": predictions,
    }


def shorten_class_name(class_name):
    return class_name.split(" (")[0]


def print_evaluation_report(metrics, class_names):
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"{'Class':<16}{'Precision':>10}{'Recall':>10}")
    for class_index, class_name in enumerate(class_names):
        precision = metrics["per_class_precision"][class_index]
        recall = metrics["per_class_recall"][class_index]
        print(f"{shorten_class_name(class_name):<16}{precision:>10.4f}{recall:>10.4f}")


def save_predictions(metrics, class_names, test_samples, output_path):
    rows = zip(test_samples, metrics["true_labels"], metrics["predictions"])

    with open(output_path, "w", newline="", encoding="utf-8") as predictions_file:
        writer = csv.writer(predictions_file)
        writer.writerow(["image", "true_label", "predicted_label", "true_class", "predicted_class", "correct"])

        for (sample_path, sample_label), true_label, predicted_label in rows:
            if sample_label != true_label:
                raise ValueError(f"Prediction order does not match the test samples at {sample_path}")

            writer.writerow([
                Path(sample_path).stem,
                true_label,
                predicted_label,
                shorten_class_name(class_names[true_label]),
                shorten_class_name(class_names[predicted_label]),
                int(true_label == predicted_label),
            ])


def save_figure(figure, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".pdf"))
    figure.savefig(output_path.with_suffix(".png"), dpi=RASTER_DPI)
    plt.close(figure)


def plot_training_curves(history, output_path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    best_epoch = history.get("best_epoch")

    with plt.rc_context(IEEE_STYLE):
        figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COLUMN_WIDTH, 2.4))

        loss_axis.plot(epochs, history["train_loss"], color=DARK_BLUE, linewidth=1.4,
                       linestyle="-", marker="o", markersize=3, label="Training")
        loss_axis.plot(epochs, history["val_loss"], color=SUN_RED, linewidth=1.4,
                       linestyle="--", marker="s", markersize=3, label="Validation")
        loss_axis.set_title("(a) Cross-entropy loss")
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("Loss")

        accuracy_axis.plot(epochs, np.array(history["train_accuracy"]) * 100, color=DARK_BLUE, linewidth=1.4,
                           linestyle="-", marker="o", markersize=3, label="Training")
        accuracy_axis.plot(epochs, np.array(history["val_accuracy"]) * 100, color=SUN_RED, linewidth=1.4,
                           linestyle="--", marker="s", markersize=3, label="Validation")
        accuracy_axis.set_title("(b) Classification accuracy")
        accuracy_axis.set_xlabel("Epoch")
        accuracy_axis.set_ylabel("Accuracy (%)")

        for axis in (loss_axis, accuracy_axis):
            if best_epoch:
                axis.axvspan(best_epoch, epochs[-1] + 0.5, color=BRIGHT_YELLOW, alpha=0.25, linewidth=0,
                             label="Patience window")
                axis.axvline(best_epoch, color=DEEP_PURPLE, linewidth=1.2, linestyle=":",
                             label=f"Best epoch ({best_epoch})")
            axis.set_xlim(0.5, epochs[-1] + 0.5)
            axis.xaxis.get_major_locator().set_params(integer=True)

        handles, labels = loss_axis.get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.06))
        figure.tight_layout(rect=(0, 0.06, 1, 1))

        save_figure(figure, output_path)


def plot_confusion_matrix(test_confusion_matrix, class_names, output_path):
    short_class_names = [shorten_class_name(class_name) for class_name in class_names]
    row_totals = test_confusion_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(test_confusion_matrix, row_totals,
                                  out=np.zeros_like(test_confusion_matrix, dtype=float), where=row_totals > 0)

    with plt.rc_context(IEEE_STYLE):
        figure, axis = plt.subplots(figsize=(IEEE_SINGLE_COLUMN_WIDTH, 3.3))
        image = axis.imshow(normalized_matrix, cmap=CONFUSION_COLORMAP, vmin=0.0, vmax=1.0)

        for row_index in range(normalized_matrix.shape[0]):
            for column_index in range(normalized_matrix.shape[1]):
                cell_fraction = normalized_matrix[row_index, column_index]
                cell_count = test_confusion_matrix[row_index, column_index]
                text_color = "#FFFFFF" if cell_fraction > 0.45 else NEUTRAL_INK
                axis.text(column_index, row_index, f"{cell_count}\n{cell_fraction * 100:.0f}%",
                          ha="center", va="center", fontsize=5.5, color=text_color)

        axis.set_xticks(range(len(short_class_names)))
        axis.set_yticks(range(len(short_class_names)))
        axis.set_xticklabels(short_class_names, rotation=45, ha="right")
        axis.set_yticklabels(short_class_names)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.grid(False)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)

        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        colorbar.set_label("Row-normalized rate")
        colorbar.outline.set_linewidth(0.4)

        figure.tight_layout()
        save_figure(figure, output_path)
