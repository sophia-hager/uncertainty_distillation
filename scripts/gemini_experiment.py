"""CLI for the Gemini (API-model) uncertainty distillation pipeline.

Tuning and batch inference are asynchronous, so an experiment is a resumable
record stored under ``runs/gemini/experiments.json``:

    # start a new experiment on a small slice of the data
    python scripts/gemini_experiment.py start --config configs/gemini_mmlu.yaml

    # advance it whenever its cloud jobs have finished
    python scripts/gemini_experiment.py continue 0 --full

    # see what exists and where each experiment got to
    python scripts/gemini_experiment.py list

Any configuration field can be overridden in dot notation, for example
``--calibration.method=isotonic_regression`` or ``--n_generations=32``.
"""

import argparse
import logging

from omegaconf import OmegaConf

import _bootstrap  # noqa: F401

from uncertainty_distillation.gemini import store
from uncertainty_distillation.gemini.config import PipelineConfig
from uncertainty_distillation.gemini.pipeline import Pipeline


def load_config(args) -> OmegaConf:
    """Merge the structured defaults, any YAML files, and CLI overrides."""
    config = OmegaConf.structured(PipelineConfig)

    for config_path in getattr(args, "config", None) or []:
        config = OmegaConf.merge(config, OmegaConf.load(config_path))

    overrides = [override.lstrip("-") for override in getattr(args, "overrides", [])]
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))

    if args.eval_on_test:
        config.test_split = True
    return config


def command_start(args):
    config = load_config(args)
    print(OmegaConf.to_yaml(config))
    experiment = Pipeline(config).run(full=args.full)
    print(f"Experiment {experiment.id} is now in stage {experiment.status.name}.")


def command_continue(args):
    experiments = store.get_experiments()
    if not experiments:
        raise SystemExit("No experiments found. Run 'start' first.")

    selected = (
        experiments
        if args.experiment == "ALL"
        else [experiments[int(args.experiment)]]
    )
    for experiment in selected:
        config = OmegaConf.merge(
            OmegaConf.structured(PipelineConfig), OmegaConf.create(experiment.config)
        )
        if args.eval_on_test:
            config.test_split = True
        pipeline = Pipeline(config)
        experiment = pipeline.run_experiment(experiment, full=args.full, redo=args.redo)
        print(f"Experiment {experiment.id} is now in stage {experiment.status.name}.")


def command_list(args):
    del args
    experiments = store.get_experiments()
    if not experiments:
        print(f"No experiments in {store.store_path()}.")
        return

    print(f"{'#':<3} {'id':<38} {'stage':<20} {'model':<28} metrics")
    for index, experiment in enumerate(experiments):
        config = experiment.config
        metrics = experiment.test_metrics or experiment.validation_metrics
        summary = (
            f"acc={metrics.get('accuracy', float('nan')):.3f} "
            f"auroc={metrics.get('auroc', float('nan')):.3f}"
            if metrics
            else "-"
        )
        print(
            f"{index:<3} {experiment.id:<38} {experiment.status.name:<20} "
            f"{str(config.get('base_model', '?')):<28} {summary}"
        )


def get_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Create and run a new experiment.")
    start.add_argument("--config", nargs="*", help="YAML config files, merged in order.")
    start.add_argument("--full", action="store_true",
                       help="Run on the full dataset (otherwise only dev_set_size examples).")
    start.add_argument("--eval_on_test", action="store_true",
                       help="Evaluate on the test half of the evaluation split.")
    start.add_argument("overrides", nargs="*", default=[],
                       help="Config overrides, e.g. n_generations=32")
    start.set_defaults(func=command_start)

    resume = subparsers.add_parser("continue", help="Advance an existing experiment.")
    resume.add_argument("experiment", help="Experiment index from 'list', or ALL.")
    resume.add_argument("--full", action="store_true")
    resume.add_argument("--redo", action="store_true", help="Re-run the current stage.")
    resume.add_argument("--eval_on_test", action="store_true")
    resume.set_defaults(func=command_continue)

    listing = subparsers.add_parser("list", help="List experiments and their metrics.")
    listing.set_defaults(func=command_list)

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = get_args()
    args.func(args)


if __name__ == "__main__":
    main()
