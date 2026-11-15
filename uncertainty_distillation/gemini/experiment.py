"""The resumable record of one Gemini experiment, and job status types."""

import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum


class ExperimentStatus(IntEnum):
    """Stages an experiment moves through, in order."""

    GENERATING_SAMPLES = 1
    CREATE_TUNING_DATA = 2
    TUNING = 3
    INFERENCE = 4
    COMPLETE = 5


class JobStatus(str, Enum):
    """Normalized status of an asynchronous cloud job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobResult:
    """A job, or a set of outputs standing in for one.

    Online inference produces no job, but writes the same outputs to the same
    place, so it is represented as an already-succeeded job.
    """

    name: str
    status: JobStatus
    dest_uris: list[str] = field(default_factory=list)
    raw: object = None


@dataclass
class Experiment:
    """Everything needed to resume an experiment and to report on it."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: ExperimentStatus = ExperimentStatus.GENERATING_SAMPLES
    tuning_job_id: str | None = None
    sample_generation_job_ids: list = field(default_factory=list)
    validation_job_ids: list[str] = field(default_factory=list)
    training_samples_gcs_uri: str | None = None
    augmented_training_dataset_gcs_uri: str | None = None
    tuning_training_dataset_gcs_uri: str | None = None
    tuning_validation_dataset_gcs_uri: str | None = None
    tuned_model_output_gcs_uri: str | None = None
    predictions_dataset_gcs_uri: str | None = None
    validation_metrics_gcs_uri: str | None = None
    test_metrics_gcs_uri: str | None = None
    sample_generation_metrics: dict = field(default_factory=dict)
    validation_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    missing_ground_truth_stats: dict = field(default_factory=dict)
    augmentation_stats: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    full: bool = False
    is_deleted: bool = False

    def __post_init__(self):
        if isinstance(self.status, int):
            self.status = ExperimentStatus(self.status)


def job_status_from_state(state_name: str) -> JobStatus:
    """Translate a Vertex AI job state into a provider-agnostic status."""
    return {
        "JOB_STATE_SUCCEEDED": JobStatus.SUCCEEDED,
        "JOB_STATE_FAILED": JobStatus.FAILED,
        "JOB_STATE_CANCELLED": JobStatus.CANCELLED,
        "JOB_STATE_RUNNING": JobStatus.RUNNING,
        "JOB_STATE_QUEUED": JobStatus.RUNNING,
    }.get(state_name, JobStatus.PENDING)
