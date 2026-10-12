import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.make_dataset import get_dataloaders
from src.models.cnn import BaselineCNN
from src.models.train import train_model
from src.utils.evaluation import evaluate_model, plot_confusion_matrix, plot_training_curves, print_evaluation_report

EXPERIMENT_NAME = "baseline"
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 50
PATIENCE = 7
NUM_WORKERS = 2

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "leafscans-orchard-compressed"
RESULTS_DIR = PROJECT_ROOT / "results" / EXPERIMENT_NAME


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_experiment_summary(history, metrics, class_names, output_path):
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "hyperparameters": {
            "random_seed": RANDOM_SEED,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
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
        },
    }

    with open(output_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)


def main():
    set_global_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=RANDOM_SEED
    )
    print(f"Train: {len(train_loader.dataset)} | Validation: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = BaselineCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=NUM_EPOCHS, patience=PATIENCE, device=device
    )

    metrics = evaluate_model(model, test_loader, device=device)
    print_evaluation_report(metrics, class_names)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RESULTS_DIR / "best_model.pt")
    plot_training_curves(history, RESULTS_DIR / "training_curves")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, RESULTS_DIR / "confusion_matrix")
    save_experiment_summary(history, metrics, class_names, RESULTS_DIR / "summary.json")

    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
