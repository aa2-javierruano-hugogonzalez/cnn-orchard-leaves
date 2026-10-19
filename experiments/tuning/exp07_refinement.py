import csv
import json
import random
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn

from src.data.make_dataset import get_dataloaders
from src.models.cnn import BaselineCNN
from src.models.train import train_model
from src.utils.evaluation import evaluate_model

EXPERIMENT_NAME = "exp07_refinement"
RANDOM_SEED = 42
IMAGE_SIZE = 128
CONV_FILTERS = (32, 64, 128, 256)
KERNEL_SIZE = 5
USE_BATCHNORM = True
FC_WIDTH = 256
ACTIVATION = nn.GELU
BATCH_SIZE = 32
NUM_TRIALS = 8
NUM_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 7
NUM_WORKERS = 2
SAMPLER_STARTUP_TRIALS = 4
LEARNING_RATE_RANGE = (2e-5, 3e-4)
DROPOUT_RANGE = (0.15, 0.40)
TOP_TRIALS_TO_PRINT = 3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "leafscans-orchard-compressed"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / EXPERIMENT_NAME


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_classes(data_dir):
    return len([folder for folder in data_dir.iterdir() if folder.is_dir()])


def build_model(dropout_rate, num_classes):
    return BaselineCNN(
        num_classes=num_classes,
        conv_filters=CONV_FILTERS,
        kernel_size=KERNEL_SIZE,
        fc_width=FC_WIDTH,
        dropout_rate=dropout_rate,
        activation=ACTIVATION,
        use_batchnorm=USE_BATCHNORM,
        use_global_pooling=False,
        input_size=IMAGE_SIZE,
    )


def objective(trial):
    set_global_seed(RANDOM_SEED)
    learning_rate = trial.suggest_float("learning_rate", *LEARNING_RATE_RANGE, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", *DROPOUT_RANGE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, class_names = get_dataloaders(
        DATA_DIR, PROCESSED_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        seed=RANDOM_SEED, image_size=IMAGE_SIZE
    )

    model = build_model(dropout_rate, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def report_validation_loss(val_loss, epoch):
        trial.report(val_loss, epoch)

    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE, device=device, scheduler=None,
        epoch_callback=report_validation_loss
    )

    validation_metrics = evaluate_model(model, val_loader, device=device)
    trial.set_user_attr("best_epoch", history["best_epoch"])
    trial.set_user_attr("stopped_epoch", history["stopped_epoch"])
    trial.set_user_attr("val_accuracy", float(validation_metrics["accuracy"]))
    trial.set_user_attr("val_macro_f1", float(validation_metrics["macro_f1"]))

    return min(history["val_loss"])


def build_sampler(num_finished_trials):
    return optuna.samplers.TPESampler(n_startup_trials=SAMPLER_STARTUP_TRIALS, seed=RANDOM_SEED + num_finished_trials)


def create_study(storage_path):
    return optuna.create_study(
        study_name=EXPERIMENT_NAME,
        storage=f"sqlite:///{storage_path.as_posix()}",
        direction="minimize",
        sampler=build_sampler(0),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )


def get_completed_trials(study):
    return [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]


def save_trials_table(study, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as trials_file:
        writer = csv.writer(trials_file)
        writer.writerow([
            "trial", "state", "best_val_loss", "val_macro_f1", "val_accuracy",
            "best_epoch", "stopped_epoch", "learning_rate", "dropout_rate",
        ])

        for trial in study.trials:
            writer.writerow([
                trial.number,
                trial.state.name,
                trial.value,
                trial.user_attrs.get("val_macro_f1"),
                trial.user_attrs.get("val_accuracy"),
                trial.user_attrs.get("best_epoch"),
                trial.user_attrs.get("stopped_epoch"),
                trial.params.get("learning_rate"),
                trial.params.get("dropout_rate"),
            ])


def save_best_hyperparameters(study, output_path):
    best_trial = study.best_trial
    best_hyperparameters = {
        "trial": best_trial.number,
        "best_val_loss": best_trial.value,
        "image_size": IMAGE_SIZE,
        "conv_filters": list(CONV_FILTERS),
        "kernel_size": KERNEL_SIZE,
        "use_batchnorm": USE_BATCHNORM,
        "fc_width": FC_WIDTH,
        "activation": ACTIVATION.__name__,
        "batch_size": BATCH_SIZE,
        **best_trial.params,
        **best_trial.user_attrs,
    }

    with open(output_path, "w", encoding="utf-8") as best_file:
        json.dump(best_hyperparameters, best_file, indent=2)


def print_top_trials(study):
    ranked_trials = sorted(get_completed_trials(study), key=lambda trial: trial.value)

    print(f"Top {min(TOP_TRIALS_TO_PRINT, len(ranked_trials))} trials by validation loss:")
    for trial in ranked_trials[:TOP_TRIALS_TO_PRINT]:
        print(f"Trial {trial.number:2d} | val_loss {trial.value:.4f} | "
              f"val_macro_f1 {trial.user_attrs['val_macro_f1']:.4f} | "
              f"val_accuracy {trial.user_attrs['val_accuracy']:.4f} | "
              f"best epoch {trial.user_attrs['best_epoch']} of {trial.user_attrs['stopped_epoch']} | "
              f"lr {trial.params['learning_rate']:.2e} | dropout {trial.params['dropout_rate']:.3f}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    study = create_study(RESULTS_DIR / "tuning.db")
    num_finished_trials = len(get_completed_trials(study))
    study.sampler = build_sampler(num_finished_trials)

    num_remaining_trials = NUM_TRIALS - num_finished_trials
    print(f"Study {EXPERIMENT_NAME}: {num_finished_trials} finished trials, {max(num_remaining_trials, 0)} to run")

    if num_remaining_trials > 0:
        study.optimize(objective, n_trials=num_remaining_trials, gc_after_trial=True,
                       catch=(torch.cuda.OutOfMemoryError,))

    save_trials_table(study, RESULTS_DIR / "trials.csv")

    if not get_completed_trials(study):
        print("No trial has completed yet")
        return

    save_best_hyperparameters(study, RESULTS_DIR / "best_hyperparameters.json")
    print_top_trials(study)


if __name__ == "__main__":
    main()
