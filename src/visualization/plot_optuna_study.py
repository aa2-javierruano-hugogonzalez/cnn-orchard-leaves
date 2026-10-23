from itertools import accumulate
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import optuna
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_NAME = "exp06_optuna_tuning"
STORAGE_PATH = PROJECT_ROOT / "results" / STUDY_NAME / "tuning.db"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
HISTORY_FILE_NAME = "optuna_history.pdf"
IMPORTANCE_FILE_NAME = "optuna_fanova.pdf"
IMPORTANCE_SEED = 42

FIGURE_SIZE = (3.1, 2.4)
TRIAL_MARKER_SIZE = 18
BEST_MARKER_SIZE = 70
UNFINISHED_MARKER_HEIGHT = 0.04
BAR_HEIGHT = 0.6
LEGEND_ANCHOR = (1.0, 0.86)
IMPORTANCE_HEADROOM = 1.2

DARK_BLUE = "#1B2A6B"
SUN_RED = "#E63946"
NEUTRAL_INK = "#222222"
MUTED_GRAY = "#8C8C8C"
GRID_GRAY = "#D9D9D9"

PARAMETER_LABELS = {
    "learning_rate": "Tasa de aprendizaje",
    "num_conv_blocks": "Bloques convolucionales",
    "first_filters": "Filtros iniciales",
    "dropout_rate": "Dropout",
    "kernel_size": "Tamaño del kernel",
    "use_batchnorm": "Normalización por lotes",
    "fc_width": "Neuronas de la capa densa",
    "batch_size": "Tamaño de lote",
}

PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "text.color": NEUTRAL_INK,
    "axes.labelcolor": NEUTRAL_INK,
    "axes.edgecolor": NEUTRAL_INK,
    "xtick.color": NEUTRAL_INK,
    "ytick.color": NEUTRAL_INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "grid.color": GRID_GRAY,
    "grid.linewidth": 0.4,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def format_decimal(value, digits):
    return f"{value:.{digits}f}".replace(".", ",")


def decimal_axis_formatter(digits):
    return FuncFormatter(lambda value, _: format_decimal(value, digits))


def load_study(storage_path):
    if not storage_path.exists():
        raise FileNotFoundError(f"No tuning database found at {storage_path}. Run {STUDY_NAME} first.")
    return optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{storage_path.as_posix()}")


def split_trials(study):
    completed_trials = sorted((trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE),
                              key=lambda trial: trial.number)
    unfinished_trials = [trial for trial in study.trials if trial.state != optuna.trial.TrialState.COMPLETE]
    return completed_trials, unfinished_trials


def plot_optimization_history(study, output_path):
    completed_trials, unfinished_trials = split_trials(study)
    trial_numbers = [trial.number for trial in completed_trials]
    validation_losses = [trial.value for trial in completed_trials]
    best_so_far = list(accumulate(validation_losses, min))
    best_trial = study.best_trial
    last_trial_number = max(trial.number for trial in study.trials)

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.plot(trial_numbers + [last_trial_number], best_so_far + [best_so_far[-1]], drawstyle="steps-post",
              color=SUN_RED, linewidth=1.4, zorder=2)
    axis.scatter(trial_numbers, validation_losses, s=TRIAL_MARKER_SIZE, color=DARK_BLUE, zorder=3)
    axis.scatter([best_trial.number], [best_trial.value], s=BEST_MARKER_SIZE, facecolors="none",
                 edgecolors=SUN_RED, linewidths=1.2, zorder=4)
    axis.scatter([trial.number for trial in unfinished_trials], [UNFINISHED_MARKER_HEIGHT] * len(unfinished_trials),
                 transform=axis.get_xaxis_transform(), marker="x", s=14, color=MUTED_GRAY, linewidths=0.9, zorder=3)

    axis.set_xlim(-0.8, last_trial_number + 0.8)
    axis.set_ylim(0, max(validation_losses) * 1.08)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_formatter(decimal_axis_formatter(1))
    axis.yaxis.grid(True)
    axis.set_axisbelow(True)
    axis.set_xlabel("Número de prueba")
    axis.set_ylabel("Pérdida en validación")

    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=4, color=DARK_BLUE, label="Prueba completa"),
        Line2D([], [], color=SUN_RED, linewidth=1.4, label="Mejor valor acumulado"),
        Line2D([], [], marker="o", linestyle="none", markersize=7, markerfacecolor="none",
               markeredgecolor=SUN_RED, label=f"Mejor prueba (n.º {best_trial.number})"),
        Line2D([], [], marker="x", linestyle="none", markersize=4, color=MUTED_GRAY, label="Podada o interrumpida"),
    ]
    axis.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=LEGEND_ANCHOR, handletextpad=0.4,
                borderaxespad=0.2)

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def compute_importances(study):
    evaluator = optuna.importance.FanovaImportanceEvaluator(seed=IMPORTANCE_SEED)
    importances = optuna.importance.get_param_importances(study, evaluator=evaluator)
    positive_importances = [(name, value) for name, value in importances.items() if value > 0]
    return sorted(positive_importances, key=lambda item: item[1], reverse=True)


def format_importance(value):
    return format_decimal(value, 2) if value >= 0.005 else "<0,01"


def plot_importances(importances, output_path):
    ordered_importances = list(reversed(importances))
    labels = [PARAMETER_LABELS.get(name, name) for name, _ in ordered_importances]
    values = [value for _, value in ordered_importances]
    positions = range(len(values))

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.barh(positions, values, height=BAR_HEIGHT, color=DARK_BLUE, zorder=3)
    for position, value in zip(positions, values):
        axis.annotate(format_importance(value), xy=(value, position), xytext=(3, 0), textcoords="offset points",
                      ha="left", va="center", fontsize=7, color=NEUTRAL_INK)

    axis.set_yticks(list(positions))
    axis.set_yticklabels(labels)
    axis.tick_params(axis="y", length=0)
    axis.set_xlim(0, max(values) * IMPORTANCE_HEADROOM)
    axis.xaxis.set_major_formatter(decimal_axis_formatter(1))
    axis.xaxis.grid(True)
    axis.set_axisbelow(True)
    axis.set_xlabel("Importancia")

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def print_summary(study, importances, output_paths):
    completed_trials, unfinished_trials = split_trials(study)
    print(f"Study {STUDY_NAME}: {len(completed_trials)} completed and {len(unfinished_trials)} pruned or failed trials")
    print(f"Best trial {study.best_trial.number} with validation loss {study.best_value:.4f}")
    print(f"fANOVA importance (seed {IMPORTANCE_SEED}):")
    for name, value in importances:
        print(f"  {PARAMETER_LABELS.get(name, name):<28}{value:.3f}")
    for output_path in output_paths:
        print(f"Figure saved to {output_path}")


def main():
    study = load_study(STORAGE_PATH)
    importances = compute_importances(study)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history_path = OUTPUT_DIR / HISTORY_FILE_NAME
    importance_path = OUTPUT_DIR / IMPORTANCE_FILE_NAME

    with matplotlib.rc_context(PLOT_STYLE):
        plot_optimization_history(study, history_path)
        plot_importances(importances, importance_path)

    print_summary(study, importances, (history_path, importance_path))


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
