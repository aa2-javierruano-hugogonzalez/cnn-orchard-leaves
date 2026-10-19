from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_NAME = "exp06_optuna_tuning"
STORAGE_PATH = PROJECT_ROOT / "results" / STUDY_NAME / "tuning.db"
TOP_TRIALS_TO_SHOW = 3
IMPORTANCE_SEED = 42
IMPORTANCE_BAR_WIDTH = 40


def load_study(storage_path):
    return optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{storage_path.as_posix()}")


def get_trials_by_state(study, state):
    return [trial for trial in study.trials if trial.state == state]


def print_progress(study):
    completed_trials = get_trials_by_state(study, optuna.trial.TrialState.COMPLETE)
    pruned_trials = get_trials_by_state(study, optuna.trial.TrialState.PRUNED)
    failed_trials = get_trials_by_state(study, optuna.trial.TrialState.FAIL)
    running_trials = get_trials_by_state(study, optuna.trial.TrialState.RUNNING)

    pruned_by_median = [trial for trial in pruned_trials if trial.intermediate_values]
    rejected_by_size = [trial for trial in pruned_trials if not trial.intermediate_values]

    print("Progress")
    print(f"  Completed trials: {len(completed_trials)}")
    print(f"  Pruned trials:    {len(pruned_trials)} "
          f"({len(pruned_by_median)} by the median pruner, {len(rejected_by_size)} by the parameter limit)")
    print(f"  Failed trials:    {len(failed_trials)}")
    print(f"  Running trials:   {len(running_trials)}")
    print(f"  Recorded trials:  {len(study.trials)}")


def print_top_trials(study):
    completed_trials = get_trials_by_state(study, optuna.trial.TrialState.COMPLETE)
    ranked_trials = sorted(completed_trials, key=lambda trial: trial.value)

    print(f"Top {min(TOP_TRIALS_TO_SHOW, len(ranked_trials))} trials by validation loss")
    for trial in ranked_trials[:TOP_TRIALS_TO_SHOW]:
        print(f"  Trial {trial.number}: val_loss {trial.value:.4f} | "
              f"val_macro_f1 {trial.user_attrs.get('val_macro_f1', float('nan')):.4f} | "
              f"val_accuracy {trial.user_attrs.get('val_accuracy', float('nan')):.4f} | "
              f"best epoch {trial.user_attrs.get('best_epoch')} of {trial.user_attrs.get('stopped_epoch')} | "
              f"{trial.user_attrs.get('num_parameters', 0):,} parameters")
        print(f"    conv_filters: {trial.user_attrs.get('conv_filters')}")
        print(f"    hyperparameters: {trial.params}")

    best_by_macro_f1 = max(completed_trials, key=lambda trial: trial.user_attrs.get("val_macro_f1", 0.0), default=None)
    if best_by_macro_f1 is not None:
        print(f"  Highest validation macro-F1: trial {best_by_macro_f1.number} "
              f"({best_by_macro_f1.user_attrs.get('val_macro_f1', float('nan')):.4f})")


def print_param_importances(study):
    completed_trials = get_trials_by_state(study, optuna.trial.TrialState.COMPLETE)
    if len(completed_trials) < 2:
        print("Hyperparameter importance needs at least two completed trials")
        return

    evaluator = optuna.importance.FanovaImportanceEvaluator(seed=IMPORTANCE_SEED)
    importances = optuna.importance.get_param_importances(study, evaluator=evaluator)

    print(f"Hyperparameter importance (fANOVA on {len(completed_trials)} completed trials)")
    for name, importance in importances.items():
        bar = "=" * round(importance * IMPORTANCE_BAR_WIDTH)
        print(f"  {name:<16} {importance:.3f}  {bar}")


def main():
    if not STORAGE_PATH.exists():
        print(f"No tuning database found at {STORAGE_PATH}")
        return

    study = load_study(STORAGE_PATH)
    print_progress(study)
    print()
    print_top_trials(study)
    print()
    print_param_importances(study)


if __name__ == "__main__":
    main()
