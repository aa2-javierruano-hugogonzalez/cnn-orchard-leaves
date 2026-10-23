import json
from pathlib import Path

import torch
import torch.nn as nn

from src.data.make_dataset import get_dataloaders
from src.models.cnn import BaselineCNN
from src.utils.evaluation import evaluate_model, save_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "leafscans-orchard-compressed"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_ROOT = PROJECT_ROOT / "results"


def find_experiments_without_predictions(results_root):
    experiment_dirs = []

    for experiment_dir in sorted(results_root.iterdir()):
        if not experiment_dir.is_dir():
            continue

        has_summary = (experiment_dir / "summary.json").exists()
        has_model = (experiment_dir / "best_model.pt").exists()
        has_predictions = (experiment_dir / "predictions.csv").exists()

        if has_summary and has_model and not has_predictions:
            experiment_dirs.append(experiment_dir)

    return experiment_dirs


def load_summary(experiment_dir):
    with open(experiment_dir / "summary.json", encoding="utf-8") as summary_file:
        return json.load(summary_file)


def build_model_from_hyperparameters(hyperparameters, num_classes):
    return BaselineCNN(
        num_classes=num_classes,
        conv_filters=tuple(hyperparameters["conv_filters"]),
        kernel_size=hyperparameters["kernel_size"],
        fc_width=hyperparameters["fc_width"],
        dropout_rate=hyperparameters["dropout_rate"],
        activation=getattr(nn, hyperparameters["activation"]),
        use_batchnorm=hyperparameters["use_batchnorm"],
        use_global_pooling=hyperparameters["use_global_pooling"],
        input_size=hyperparameters["image_size"],
    )


def rebuild_predictions(experiment_dir, device):
    summary = load_summary(experiment_dir)
    hyperparameters = summary["hyperparameters"]

    _, _, test_loader, class_names = get_dataloaders(
        DATA_DIR, PROCESSED_DIR, batch_size=hyperparameters["batch_size"], num_workers=0,
        seed=hyperparameters["random_seed"], image_size=hyperparameters["image_size"]
    )

    model = build_model_from_hyperparameters(hyperparameters, len(class_names)).to(device)
    state_dict = torch.load(experiment_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    metrics = evaluate_model(model, test_loader, device=device)
    save_predictions(metrics, class_names, test_loader.dataset.samples, experiment_dir / "predictions.csv")

    matches_summary = metrics["confusion_matrix"].tolist() == summary["confusion_matrix"]
    status = "matches summary.json" if matches_summary else "WARNING: does not match summary.json"
    print(f"{experiment_dir.name}: test accuracy {metrics['accuracy']:.4f} "
          f"(summary {summary['test_accuracy']:.4f}), {status}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not RESULTS_ROOT.exists():
        print(f"No results directory found at {RESULTS_ROOT}")
        return

    experiment_dirs = find_experiments_without_predictions(RESULTS_ROOT)
    if not experiment_dirs:
        print("Every experiment already has a predictions.csv")
        return

    for experiment_dir in experiment_dirs:
        try:
            rebuild_predictions(experiment_dir, device)
        except KeyError as missing_setting:
            print(f"Skipping {experiment_dir.name}: summary.json has no {missing_setting} setting")
        except RuntimeError:
            print(f"Skipping {experiment_dir.name}: best_model.pt does not match the architecture in summary.json")


if __name__ == "__main__":
    main()
