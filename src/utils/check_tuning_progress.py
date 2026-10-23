import argparse
from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_STUDY_NAME = "exp07_refinement"
TOP_TRIALS_TO_SHOW = 3
IMPORTANCE_SEED = 42
IMPORTANCE_BAR_WIDTH = 40


def parse_arguments():
    parser = argparse.ArgumentParser(description="Show the progress of an Optuna tuning study.")
    parser.add_argument(
        "--study", "--experiment", dest="study_name", default=DEFAULT_STUDY_NAME,
        help=f"study name, matching its folder in results/ (default: {DEFAULT_STUDY_NAME})"
    )
    return parser.parse_args()


def find_available_studies(results_root):
    if not results_root.exists():
        return []
    return sorted(folder.name for folder in results_root.iterdir() if (folder / "tuning.db").exists())


def load_study(study_name, storage_path):
    return optuna.load_study(study_name=study_name, storage=f"sqlite:///{storage_path.as_posix()}")


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
        details = [
            f"val_loss {trial.value:.4f}",
            f"val_macro_f1 {trial.user_attrs.get('val_macro_f1', float('nan')):.4f}",
            f"val_accuracy {trial.user_attrs.get('val_accuracy', float('nan')):.4f}",
            f"best epoch {trial.user_attrs.get('best_epoch')} of {trial.user_attrs.get('stopped_epoch')}",
        ]
        if "num_parameters" in trial.user_attrs:
            details.append(f"{trial.user_attrs['num_parameters']:,} parameters")

        print(f"  Trial {trial.number}: " + " | ".join(details))
        if "conv_filters" in trial.user_attrs:
            print(f"    conv_filters: {trial.user_attrs['conv_filters']}")
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
    arguments = parse_arguments()
    storage_path = RESULTS_ROOT / arguments.study_name / "tuning.db"

    if not storage_path.exists():
        print(f"No tuning database found at {storage_path}")
        available_studies = find_available_studies(RESULTS_ROOT)
        if available_studies:
            print(f"Available studies: {', '.join(available_studies)}")
        return

    study = load_study(arguments.study_name, storage_path)
    print(f"Study: {arguments.study_name}")
    print()
    print_progress(study)
    print()
    print_top_trials(study)
    print()
    print_param_importances(study)


if __name__ == "__main__":
    main()
