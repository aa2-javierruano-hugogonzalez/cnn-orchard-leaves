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

EXPERIMENT_NAME = "exp06_optuna_tuning"
RANDOM_SEED = 42
IMAGE_SIZE = 128
NUM_TRIALS = 40
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 7
NUM_WORKERS = 2
ACTIVATION = nn.GELU
MAX_PARAMETERS = 10_000_000
MAX_FILTERS = 256
SAMPLER_STARTUP_TRIALS = 10
PRUNER_STARTUP_TRIALS = 5
PRUNER_WARMUP_EPOCHS = 5
TOP_TRIALS_TO_PRINT = 5

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


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_conv_filters(first_filters, num_conv_blocks):
    return tuple(min(first_filters * 2 ** block_index, MAX_FILTERS) for block_index in range(num_conv_blocks))


def suggest_hyperparameters(trial):
    num_conv_blocks = trial.suggest_int("num_conv_blocks", 3, 5)
    first_filters = trial.suggest_categorical("first_filters", [16, 32, 48])

    return {
        "conv_filters": build_conv_filters(first_filters, num_conv_blocks),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
        "use_batchnorm": trial.suggest_categorical("use_batchnorm", [False, True]),
        "fc_width": trial.suggest_categorical("fc_width", [128, 256, 512]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.6),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
    }


def build_model(hyperparameters, num_classes):
    return BaselineCNN(
        num_classes=num_classes,
        conv_filters=hyperparameters["conv_filters"],
        kernel_size=hyperparameters["kernel_size"],
        fc_width=hyperparameters["fc_width"],
        dropout_rate=hyperparameters["dropout_rate"],
        activation=ACTIVATION,
        use_batchnorm=hyperparameters["use_batchnorm"],
        use_global_pooling=False,
        input_size=IMAGE_SIZE,
    )


def objective(trial):
    set_global_seed(RANDOM_SEED)
    hyperparameters = suggest_hyperparameters(trial)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(hyperparameters, count_classes(DATA_DIR))
    num_parameters = count_parameters(model)
    trial.set_user_attr("conv_filters", list(hyperparameters["conv_filters"]))
    trial.set_user_attr("num_parameters", num_parameters)

    if num_parameters > MAX_PARAMETERS:
        raise optuna.exceptions.TrialPruned(f"{num_parameters:,} parameters exceed the {MAX_PARAMETERS:,} limit")

    train_loader, val_loader, _, _ = get_dataloaders(
        DATA_DIR, PROCESSED_DIR, batch_size=hyperparameters["batch_size"], num_workers=NUM_WORKERS,
        seed=RANDOM_SEED, image_size=IMAGE_SIZE
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

    def report_validation_loss(val_loss, epoch):
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned(f"Pruned at epoch {epoch} with validation loss {val_loss:.4f}")

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
        pruner=optuna.pruners.MedianPruner(n_startup_trials=PRUNER_STARTUP_TRIALS, n_warmup_steps=PRUNER_WARMUP_EPOCHS),
        load_if_exists=True,
    )


def count_finished_trials(study):
    finished_states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return sum(trial.state in finished_states for trial in study.trials)


def get_completed_trials(study):
    return [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]


def save_trials_table(study, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as trials_file:
        writer = csv.writer(trials_file)
        writer.writerow([
            "trial", "state", "best_val_loss", "val_macro_f1", "val_accuracy", "num_parameters", "best_epoch",
            "conv_filters", "kernel_size", "use_batchnorm", "fc_width", "dropout_rate", "learning_rate", "batch_size",
        ])

        for trial in study.trials:
            conv_filters = trial.user_attrs.get("conv_filters", [])
            writer.writerow([
                trial.number,
                trial.state.name,
                trial.value,
                trial.user_attrs.get("val_macro_f1"),
                trial.user_attrs.get("val_accuracy"),
                trial.user_attrs.get("num_parameters"),
                trial.user_attrs.get("best_epoch"),
                "-".join(str(filters) for filters in conv_filters),
                trial.params.get("kernel_size"),
                trial.params.get("use_batchnorm"),
                trial.params.get("fc_width"),
                trial.params.get("dropout_rate"),
                trial.params.get("learning_rate"),
                trial.params.get("batch_size"),
            ])


def save_best_hyperparameters(study, output_path):
    best_trial = study.best_trial
    best_hyperparameters = {
        "trial": best_trial.number,
        "best_val_loss": best_trial.value,
        **best_trial.params,
        **best_trial.user_attrs,
    }

    with open(output_path, "w", encoding="utf-8") as best_file:
        json.dump(best_hyperparameters, best_file, indent=2)


def print_top_trials(study):
    ranked_trials = sorted(get_completed_trials(study), key=lambda trial: trial.value)

    print(f"Top {min(TOP_TRIALS_TO_PRINT, len(ranked_trials))} trials by validation loss:")
    for trial in ranked_trials[:TOP_TRIALS_TO_PRINT]:
        filters = "-".join(str(filters) for filters in trial.user_attrs["conv_filters"])
        print(f"Trial {trial.number:3d} | val_loss {trial.value:.4f} | "
              f"val_macro_f1 {trial.user_attrs['val_macro_f1']:.4f} | filters {filters} | "
              f"kernel {trial.params['kernel_size']} | batchnorm {trial.params['use_batchnorm']} | "
              f"fc {trial.params['fc_width']} | dropout {trial.params['dropout_rate']:.2f} | "
              f"lr {trial.params['learning_rate']:.1e} | batch {trial.params['batch_size']}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    study = create_study(RESULTS_DIR / "tuning.db")
    num_finished_trials = count_finished_trials(study)
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
