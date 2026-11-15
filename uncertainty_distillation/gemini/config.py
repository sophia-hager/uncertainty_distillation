"""Configuration for the Gemini pipeline.

The config is an OmegaConf structured config, so every field below can be
overridden from a YAML file or from the command line in dot notation
(``--calibration.method=isotonic_regression``).  Values that identify *your*
cloud resources are read from the environment, so nothing in this repository
points at a particular project or bucket.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConfidenceLevel(str, Enum):
    FIVE_BINS = "FIVE_BINS"
    SEVEN_BINS = "SEVEN_BINS"


class ExperimentType(str, Enum):
    #: Distill confidence into the model: sample, augment, tune, evaluate.
    REGULAR = "REGULAR"
    #: Skip straight to evaluation of the untuned base model, for comparison.
    BASELINE = "BASELINE"


#: How the augmented training set is built from the teacher's samples.
AUGMENTATION_SCHEMES = (
    # Keep only the correct answer, labelled with its calibrated confidence.
    "CORRECT_ONLY",
    # Correct answer plus `augmentation_scheme_n` randomly chosen wrong ones.
    "CORRECT_PLUS_N",
    # Correct answer plus one wrong answer per occupied confidence bin.
    "STRATIFIED",
    # One target listing every bin and the answer that falls in it (empty bins
    # are explicit nulls).
    "BINNED_ANSWERS",
    # As above, but bins with no answer are omitted from the target.
    "SPARSE_BINNED_ANSWERS",
)


@dataclass
class DatasetConfig:
    """A Hugging Face dataset, or a local JSONL file ending in ``.jsonl``."""

    name: str = "answerdotai/MMLU-SemiPro"
    # When split="test" the dataset provider further divides that split
    # deterministically into validation/test halves; set `test_split` to use
    # the second half instead of the first.
    split: str = "train"
    candidates_field: Optional[str] = "options"
    answer_field_contains: str = "full_answer"


@dataclass
class GenerationConfig:
    thinking_budget: int = 0
    temperature: float = 1.0
    candidate_count: int = 8


@dataclass
class GoogleConfig:
    project_id: Optional[str] = "${oc.env:GOOGLE_CLOUD_PROJECT,null}"
    location: Optional[str] = "${oc.env:GOOGLE_CLOUD_LOCATION,us-east1}"


@dataclass
class CalibrationConfig:
    #: One of platt_scaling, temperature_scaling, isotonic_regression, or null
    #: to use the raw sample frequency as the confidence.
    method: Optional[str] = None
    #: Fraction of prompts held out of augmentation to fit the calibrator.
    validation_split: float = 0.1


@dataclass
class PipelineConfig:
    experiment_type: ExperimentType = ExperimentType.REGULAR
    base_model: str = "gemini-2.5-flash-lite"

    #: Samples drawn per training question to estimate the teacher's confidence.
    n_generations: int = 128
    confidence_level: ConfidenceLevel = ConfidenceLevel.FIVE_BINS
    include_reasoning: bool = False
    augmentation_scheme: str = "CORRECT_PLUS_N"
    augmentation_scheme_n: int = 1
    prompt_name: str = "PROMPT_DEFAULT"

    training_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    validation_dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(split="test")
    )
    test_dataset: Optional[DatasetConfig] = None
    #: Use the second half of the evaluation data (the test half) instead of
    #: the first (the validation half).
    test_split: bool = False
    #: Number of examples used when not running on the full dataset.
    dev_set_size: int = 10

    # Google Cloud. The bucket is yours to create: bucket names are globally
    # unique, so there is no default to ship. Set UD_GCS_BUCKET or configure it.
    gcs_bucket: Optional[str] = "${oc.env:UD_GCS_BUCKET,null}"
    #: Project that owns the bucket; inferred from your credentials if unset.
    gcs_project: Optional[str] = "${oc.env:UD_GCS_PROJECT,null}"
    google: GoogleConfig = field(default_factory=GoogleConfig)

    # Vertex AI supervised tuning.
    tuned_model_display_name: str = "uncertainty-distilled-model"
    epochs: int = 10
    learning_rate_multiplier: float = 1.0
    adapter_size: int = 4
    #: Fraction of the augmented dataset held out as the tuning validation set.
    tuning_validation_split: float = 0.1

    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    #: Online inference is simpler and fine for small runs; batch inference is
    #: much cheaper and is what full-scale runs use.
    online_inference: bool = True
