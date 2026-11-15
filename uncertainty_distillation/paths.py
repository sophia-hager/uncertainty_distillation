"""Repository-relative default paths.

Nothing in this project writes outside the repository by default; every path
below is derived from the location of this file so that the code works from any
checkout and from any working directory.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Prompt/label JSONL files built by ``scripts/write_dataset.py``.
DATA_DIR = REPO_ROOT / "data"

#: Teacher samples written by ``scripts/sample_teacher*.py``.
SAMPLES_DIR = REPO_ROOT / "samples"

#: Fine-tuning datasets written by ``scripts/build_finetuning_data*.py``.
DATASETS_DIR = REPO_ROOT / "datasets"

#: Model checkpoints, predictions and metrics written by ``scripts/train.py``.
RUNS_DIR = REPO_ROOT / "runs"


def ensure_parent(path) -> Path:
    """Create the parent directory of ``path`` and return ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
