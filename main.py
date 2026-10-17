import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.make_dataset import IMAGE_SIZE, get_dataloaders
from src.models.cnn import BaselineCNN
from src.models.train import train_model
from src.utils.evaluation import evaluate_model, plot_confusion_matrix, plot_training_curves, print_evaluation_report

EXPERIMENT_NAME = "baseline"
RANDOM_SEED = 42
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
SCHEDULER_FACTOR = 0.3
SCHEDULER_PATIENCE = 2
EARLY_STOPPING_PATIENCE = 7
BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_WORKERS = 2
CONV_FILTERS = (16, 32, 64)
KERNEL_SIZE = 5
FC_WIDTH = 128
DROPOUT_RATE = 0.25
ACTIVATION = nn.ReLU

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "leafscans-orchard-compressed"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_ROOT = PROJECT_ROOT / "results"


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_run_name(experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{experiment_name}_{timestamp}"


def save_experiment_summary(run_name, history, metrics, class_names, output_path):
    summary = {
        "experiment_name": run_name,
        "hyperparameters": {
            "random_seed": RANDOM_SEED,
            "image_size": IMAGE_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": SCHEDULER_FACTOR,
            "scheduler_patience": SCHEDULER_PATIENCE,
            "loss_function": "CrossEntropyLoss",
            "conv_filters": list(CONV_FILTERS),
            "kernel_size": KERNEL_SIZE,
            "fc_width": FC_WIDTH,
            "dropout_rate": DROPOUT_RATE,
            "activation": ACTIVATION.__name__,
        },
        "best_epoch": history["best_epoch"],
        "stopped_epoch": history["stopped_epoch"],
        "test_accuracy": float(metrics["accuracy"]),
        "test_macro_f1": float(metrics["macro_f1"]),
        "per_class_precision": dict(zip(class_names, metrics["per_class_precision"].tolist())),
        "per_class_recall": dict(zip(class_names, metrics["per_class_recall"].tolist())),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
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


def main():
    set_global_seed(RANDOM_SEED)
    run_name = create_run_name(EXPERIMENT_NAME)
    results_dir = RESULTS_ROOT / run_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Experiment: {run_name}")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        DATA_DIR, PROCESSED_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=RANDOM_SEED
    )
    print(f"Train: {len(train_loader.dataset)} | Validation: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = BaselineCNN(
        num_classes=len(class_names), conv_filters=CONV_FILTERS, kernel_size=KERNEL_SIZE,
        fc_width=FC_WIDTH, dropout_rate=DROPOUT_RATE, activation=ACTIVATION
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE, device=device, scheduler=scheduler
    )

    metrics = evaluate_model(model, test_loader, device=device)
    print_evaluation_report(metrics, class_names)

    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), results_dir / "best_model.pt")
    plot_training_curves(history, results_dir / "training_curves")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, results_dir / "confusion_matrix")
    save_experiment_summary(run_name, history, metrics, class_names, results_dir / "summary.json")

    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
