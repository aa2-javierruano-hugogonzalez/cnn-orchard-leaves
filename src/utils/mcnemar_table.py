import csv
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
SIGNIFICANCE_LEVEL = 0.05


def find_experiments_with_predictions(results_root):
    experiment_dirs = []

    for experiment_dir in sorted(results_root.iterdir()):
        if not experiment_dir.is_dir():
            continue

        if (experiment_dir / "predictions.csv").exists():
            experiment_dirs.append(experiment_dir)
        else:
            print(f"Skipping {experiment_dir.name}: no predictions.csv", file=sys.stderr)

    return experiment_dirs


def load_correctness(experiment_dir):
    correctness = {}

    with open(experiment_dir / "predictions.csv", newline="", encoding="utf-8") as predictions_file:
        for row in csv.DictReader(predictions_file):
            correctness[(row["image"], row["true_label"])] = row["correct"] == "1"

    return correctness


def exact_mcnemar_p_value(only_first_correct, only_second_correct):
    num_discordant = only_first_correct + only_second_correct
    if num_discordant == 0:
        return 1.0

    smaller_count = min(only_first_correct, only_second_correct)
    tail_count = sum(math.comb(num_discordant, successes) for successes in range(smaller_count + 1))
    return min(1.0, 2 * tail_count / 2 ** num_discordant)


def holm_adjust(p_values):
    num_tests = len(p_values)
    ascending_order = sorted(range(num_tests), key=lambda index: p_values[index])
    adjusted_p_values = [0.0] * num_tests
    running_maximum = 0.0

    for rank, index in enumerate(ascending_order):
        scaled_p_value = min(1.0, (num_tests - rank) * p_values[index])
        running_maximum = max(running_maximum, scaled_p_value)
        adjusted_p_values[index] = running_maximum

    return adjusted_p_values


def compare_all_pairs(experiment_names, correctness_by_experiment):
    comparisons = []

    for first_index, first_name in enumerate(experiment_names):
        for second_name in experiment_names[first_index + 1:]:
            first_correctness = correctness_by_experiment[first_name]
            second_correctness = correctness_by_experiment[second_name]

            only_first_correct = sum(first_correctness[key] and not second_correctness[key] for key in first_correctness)
            only_second_correct = sum(second_correctness[key] and not first_correctness[key] for key in first_correctness)

            comparisons.append({
                "first_name": first_name,
                "second_name": second_name,
                "only_first_correct": only_first_correct,
                "only_second_correct": only_second_correct,
                "p_value": exact_mcnemar_p_value(only_first_correct, only_second_correct),
            })

    holm_p_values = holm_adjust([comparison["p_value"] for comparison in comparisons])
    for comparison, holm_p_value in zip(comparisons, holm_p_values):
        comparison["holm_p_value"] = holm_p_value

    return comparisons


def short_label(experiment_name):
    match = re.match(r"exp(\d+)", experiment_name)
    if match:
        return str(int(match.group(1)))
    return experiment_name.replace("_", r"\_")


def format_p_value(p_value):
    if p_value < 0.001:
        return "$<$0.001"
    return f"{p_value:.3f}"


def build_latex_table(comparisons, accuracies, num_images):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Pairwise exact McNemar tests on the test split ($n = {num_images}$). "
        r"Each pair compares Experiment A with Experiment B; accuracies are in \%. "
        r"A only and B only count the test images classified correctly by only one of the two models. "
        rf"Holm-adjusted $p$-values below {SIGNIFICANCE_LEVEL} are shown in bold.}}",
        r"\label{tab:mcnemar}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Pair & Acc.\ A & Acc.\ B & A only & B only & $p$ & Holm $p$ \\",
        r"\hline",
    ]

    for comparison in comparisons:
        first_name = comparison["first_name"]
        second_name = comparison["second_name"]

        holm_text = format_p_value(comparison["holm_p_value"])
        if comparison["holm_p_value"] < SIGNIFICANCE_LEVEL:
            holm_text = rf"\textbf{{{holm_text}}}"

        cells = [
            rf"{short_label(first_name)} vs.\ {short_label(second_name)}",
            f"{accuracies[first_name]:.1f}",
            f"{accuracies[second_name]:.1f}",
            str(comparison["only_first_correct"]),
            str(comparison["only_second_correct"]),
            format_p_value(comparison["p_value"]),
            holm_text,
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    if not RESULTS_ROOT.exists():
        print(f"No results directory found at {RESULTS_ROOT}", file=sys.stderr)
        return

    experiment_dirs = find_experiments_with_predictions(RESULTS_ROOT)
    if len(experiment_dirs) < 2:
        print("At least two experiments with a predictions.csv are needed", file=sys.stderr)
        return

    experiment_names = [experiment_dir.name for experiment_dir in experiment_dirs]
    correctness_by_experiment = {experiment_dir.name: load_correctness(experiment_dir) for experiment_dir in experiment_dirs}

    reference_images = set(correctness_by_experiment[experiment_names[0]])
    for experiment_name in experiment_names:
        if set(correctness_by_experiment[experiment_name]) != reference_images:
            raise ValueError(f"{experiment_name} was evaluated on different test images than {experiment_names[0]}")

    accuracies = {
        experiment_name: 100 * sum(correctness.values()) / len(correctness)
        for experiment_name, correctness in correctness_by_experiment.items()
    }

    comparisons = compare_all_pairs(experiment_names, correctness_by_experiment)
    print(build_latex_table(comparisons, accuracies, len(reference_images)))


if __name__ == "__main__":
    main()
