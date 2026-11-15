"""A small JSON store of experiments, kept inside the repository.

Cloud jobs outlive a single process, so the state of each experiment is written
to disk after every stage and read back when the experiment is resumed.
"""

import json
import os
from dataclasses import asdict

from omegaconf import DictConfig, OmegaConf

from uncertainty_distillation.paths import RUNS_DIR, ensure_parent

from .experiment import Experiment, ExperimentStatus

#: Repository-relative default; override with the UD_EXPERIMENT_STORE env var.
DEFAULT_STORE_PATH = RUNS_DIR / "gemini" / "experiments.json"


def store_path() -> str:
    return os.environ.get("UD_EXPERIMENT_STORE", str(DEFAULT_STORE_PATH))


def _load_raw() -> list[dict]:
    path = store_path()
    if not os.path.exists(path):
        return []
    with open(path) as store_file:
        try:
            return json.load(store_file)
        except json.JSONDecodeError:
            return []


def _to_experiment(record: dict) -> Experiment:
    record = dict(record)
    if isinstance(record.get("config"), dict):
        record["config"] = OmegaConf.create(record["config"])
    return Experiment(**record)


def _save(experiments: list[Experiment]) -> None:
    records = []
    for experiment in experiments:
        record = dict(experiment.__dict__)
        record["status"] = ExperimentStatus(record["status"]).value
        if isinstance(record["config"], DictConfig):
            record["config"] = OmegaConf.to_container(record["config"], resolve=True)
        records.append(record)

    with open(ensure_parent(store_path()), "w") as store_file:
        json.dump(records, store_file, indent=2)


def create_experiment(config) -> Experiment:
    experiment = Experiment(config=config)
    experiments = [_to_experiment(record) for record in _load_raw()]
    experiments.append(experiment)
    _save(experiments)
    return experiment


def get_experiments() -> list[Experiment]:
    return [_to_experiment(r) for r in _load_raw() if not r.get("is_deleted", False)]


def get_experiment(experiment_id: str) -> Experiment | None:
    for experiment in get_experiments():
        if experiment.id == experiment_id:
            return experiment
    return None


def update_experiment(experiment: Experiment) -> None:
    experiments = [_to_experiment(record) for record in _load_raw()]
    for index, existing in enumerate(experiments):
        if existing.id == experiment.id:
            experiments[index] = experiment
            break
    else:
        experiments.append(experiment)
    _save(experiments)


__all__ = [
    "asdict",
    "create_experiment",
    "get_experiment",
    "get_experiments",
    "store_path",
    "update_experiment",
]
