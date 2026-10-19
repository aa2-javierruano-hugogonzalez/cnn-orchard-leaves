import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.make_dataset import get_dataloaders
from src.models.cnn import BaselineCNN
from src.models.train import train_model
from src.utils.evaluation import (evaluate_model, plot_confusion_matrix, plot_training_curves,
                                  print_evaluation_report, save_predictions)

EXPERIMENT_NAME = "exp08_multi_seed_evaluation"
SEEDS = (42, 43, 44)
SPLIT_SEED = 42
SELECTION_METRIC = "val_macro_f1"
IMAGE_SIZE = 128
CONV_FILTERS = (32, 64, 128, 256)
KERNEL_SIZE = 5
USE_BATCHNORM = True
USE_GLOBAL_POOLING = False
FC_WIDTH = 256
ACTIVATION = nn.GELU
BATCH_SIZE = 32
LEARNING_RATE = 5.5147185878698906e-05
DROPOUT_RATE = 0.38767857660247906
WEIGHT_DECAY = 0.0
NUM_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 7
NUM_WORKERS = 2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "leafscans-orchard-compressed"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / EXPERIMENT_NAME


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes):
    return BaselineCNN(
        num_classes=num_classes,
        conv_filters=CONV_FILTERS,
        kernel_size=KERNEL_SIZE,
        fc_width=FC_WIDTH,
        dropout_rate=DROPOUT_RATE,
        activation=ACTIVATION,
        use_batchnorm=USE_BATCHNORM,
        use_global_pooling=USE_GLOBAL_POOLING,
        input_size=IMAGE_SIZE,
    )


def train_with_seed(seed, device):
    set_global_seed(seed)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        DATA_DIR, PROCESSED_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        seed=SPLIT_SEED, image_size=IMAGE_SIZE, shuffle_seed=seed
    )

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE, device=device, scheduler=None
    )

    val_metrics = evaluate_model(model, val_loader, device=device)
    test_metrics = evaluate_model(model, test_loader, device=device)

    return {
        "seed": seed,
        "history": history,
        "class_names": class_names,
        "test_samples": test_loader.dataset.samples,
        "test_metrics": test_metrics,
        "model_state": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
    }


def print_seed_result(run):
    print(f"Seed {run['seed']}: test accuracy {run['test_accuracy']:.4f} | test macro-F1 {run['test_macro_f1']:.4f} | "
          f"val macro-F1 {run['val_macro_f1']:.4f} | best epoch {run['history']['best_epoch']} "
          f"of {run['history']['stopped_epoch']}")


def mean_and_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1))


def print_robustness_summary(runs):
    print(f"Results over seeds {', '.join(str(run['seed']) for run in runs)}")
    for metric_name in ("test_accuracy", "test_macro_f1", "val_macro_f1"):
        mean, std = mean_and_std([run[metric_name] for run in runs])
        print(f"  {metric_name:<14} mean {mean:.4f} | std {std:.4f}")


def describe_run(run):
    return {
        "seed": run["seed"],
        "best_epoch": run["history"]["best_epoch"],
        "stopped_epoch": run["history"]["stopped_epoch"],
        "val_accuracy": run["val_accuracy"],
        "val_macro_f1": run["val_macro_f1"],
        "test_accuracy": run["test_accuracy"],
        "test_macro_f1": run["test_macro_f1"],
    }


def save_experiment_summary(selected_run, runs, output_path):
    test_metrics = selected_run["test_metrics"]
    class_names = selected_run["class_names"]
    history = selected_run["history"]
    test_accuracy_mean, test_accuracy_std = mean_and_std([run["test_accuracy"] for run in runs])
    test_macro_f1_mean, test_macro_f1_std = mean_and_std([run["test_macro_f1"] for run in runs])

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "selection_metric": SELECTION_METRIC,
        "selected_seed": selected_run["seed"],
        "hyperparameters": {
            "random_seed": SPLIT_SEED,
            "training_seed": selected_run["seed"],
            "seeds": list(SEEDS),
            "image_size": IMAGE_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "optimizer": "Adam",
            "scheduler": None,
            "loss_function": "CrossEntropyLoss",
            "conv_filters": list(CONV_FILTERS),
            "kernel_size": KERNEL_SIZE,
            "fc_width": FC_WIDTH,
            "dropout_rate": DROPOUT_RATE,
            "activation": ACTIVATION.__name__,
            "use_batchnorm": USE_BATCHNORM,
            "use_global_pooling": USE_GLOBAL_POOLING,
        },
        "best_epoch": history["best_epoch"],
        "stopped_epoch": history["stopped_epoch"],
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_accuracy_mean": test_accuracy_mean,
        "test_accuracy_std": test_accuracy_std,
        "test_macro_f1_mean": test_macro_f1_mean,
        "test_macro_f1_std": test_macro_f1_std,
        "seed_runs": [describe_run(run) for run in runs],
        "per_class_precision": dict(zip(class_names, test_metrics["per_class_precision"].tolist())),
        "per_class_recall": dict(zip(class_names, test_metrics["per_class_recall"].tolist())),
        "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        "history": {
            "train_loss": history["train_loss"],
            "train_accuracy": history["train_accuracy"],
            "val_loss": history["val_loss"],
            "val_accuracy": history["val_accuracy"],
            "learning_rate": history["learning_rate"],
        },
    }

    with open(output_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)


def save_selected_run(selected_run, runs, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    class_names = selected_run["class_names"]
    test_metrics = selected_run["test_metrics"]

    torch.save(selected_run["model_state"], results_dir / "best_model.pt")
    plot_training_curves(selected_run["history"], results_dir / "training_curves")
    plot_confusion_matrix(test_metrics["confusion_matrix"], class_names, results_dir / "confusion_matrix")
    save_experiment_summary(selected_run, runs, results_dir / "summary.json")
    save_predictions(test_metrics, class_names, selected_run["test_samples"], results_dir / "predictions.csv")


def main():
    if RESULTS_DIR.exists():
        raise FileExistsError(f"{RESULTS_DIR} already exists. Choose a new EXPERIMENT_NAME or move the old results.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Device: {device}")

    runs = []
    for seed in SEEDS:
        print(f"Training with seed {seed}")
        run = train_with_seed(seed, device)
        print_seed_result(run)
        runs.append(run)

    print()
    for run in runs:
        print_seed_result(run)
    print()
    print_robustness_summary(runs)

    selected_run = max(runs, key=lambda run: run[SELECTION_METRIC])
    print()
    print(f"Selected seed {selected_run['seed']} by {SELECTION_METRIC}")
    print_evaluation_report(selected_run["test_metrics"], selected_run["class_names"])

    save_selected_run(selected_run, runs, RESULTS_DIR)
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
