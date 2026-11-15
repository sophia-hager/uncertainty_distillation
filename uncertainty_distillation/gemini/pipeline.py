"""The Gemini uncertainty-distillation pipeline.

One ``Pipeline`` runs one experiment as far as it can and then stops: tuning
and batch inference are asynchronous cloud jobs, so an experiment that is
waiting on one is saved and resumed by running it again.
"""

import collections
import dataclasses
import logging
import random

from omegaconf import OmegaConf
from tqdm import tqdm

from . import store
from .augmentation import BINNED_SCHEMES, AugmentationManager
from .calibration import train_and_validate_calibrator
from .config import ExperimentType, PipelineConfig
from .datasets import HuggingFaceDatasetProvider
from .evaluator import Evaluator
from .experiment import (
    Experiment,
    ExperimentStatus,
    JobResult,
    JobStatus,
    job_status_from_state,
)
from .normalization import StringNormalizer
from .prompts import get_prompt_with_instructions
from .provider import GoogleGenAIProvider
from .schemas import get_mmlu_response_model
from .storage import get_client, read_jsonl, write_jsonl

#: Candidates a single Gemini request may return.
MAX_CANDIDATE_COUNT = 8

#: Rows Vertex AI accepts in one batch prediction job.
BATCH_SHARD_SIZE = 200_000

@dataclasses.dataclass
class PreparedDatasets:
    training_gcs_uri: str
    training_examples: int
    validation_gcs_uri: str
    validation_examples: int


class Pipeline:
    """Runs uncertainty distillation against the Gemini API."""

    def __init__(self, config):
        self.config = OmegaConf.merge(
            OmegaConf.structured(PipelineConfig), OmegaConf.create(config)
        )
        if not self.config.gcs_bucket:
            raise ValueError(
                "No GCS bucket configured. Set the UD_GCS_BUCKET environment variable, or "
                "pass gcs_bucket=<your bucket> in a config file or on the command line."
            )
        if self.config.n_generations % self.config.generation_config.candidate_count != 0:
            raise ValueError(
                "n_generations must be a multiple of candidate_count: "
                f"n_generations={self.config.n_generations}, "
                f"candidate_count={self.config.generation_config.candidate_count}"
            )

        self.provider = GoogleGenAIProvider(
            project_id=self.config.google.project_id,
            location=self.config.google.location,
        )
        self.gcs_client = get_client(self.config.get("gcs_project"))
        self.normalizer = StringNormalizer()
        self.augmentation_manager = AugmentationManager(self.config, self.normalizer)

    # --- Entry points ----------------------------------------------------

    def run(self, full: bool = False) -> Experiment:
        """Start a new experiment and advance it as far as possible."""
        experiment = store.create_experiment(self.config)
        logging.info("Created experiment %s", experiment.id)
        return self.run_experiment(experiment, full=full)

    def run_experiment(
        self, experiment: Experiment, full: bool = False, redo: bool = False
    ) -> Experiment:
        """Advance an experiment until it completes or waits on a cloud job."""
        if not full:
            # Batch inference is only worth its latency on the full dataset.
            self.config.online_inference = True

        while experiment.status != ExperimentStatus.COMPLETE:
            if (
                self.config.experiment_type == ExperimentType.BASELINE
                and experiment.status < ExperimentStatus.INFERENCE
            ):
                logging.info("Baseline experiment: skipping to inference.")
                experiment.status = ExperimentStatus.INFERENCE
                store.update_experiment(experiment)

            previous_status = experiment.status

            if experiment.status == ExperimentStatus.GENERATING_SAMPLES:
                self._advance_sample_generation(experiment, full=full, redo=redo)
            elif experiment.status == ExperimentStatus.CREATE_TUNING_DATA:
                self._create_augmented_dataset_from_samples(
                    experiment, self._sample_generation_jobs(experiment), full=full
                )
                experiment.status = ExperimentStatus.TUNING
                store.update_experiment(experiment)
            elif experiment.status == ExperimentStatus.TUNING:
                self._advance_tuning(experiment, redo=redo)
            elif experiment.status == ExperimentStatus.INFERENCE:
                self._advance_inference(experiment, full=full, redo=redo)

            redo = False
            if experiment.status == previous_status:
                logging.info(
                    "Experiment %s is waiting on a cloud job in stage %s; run it again later.",
                    experiment.id,
                    experiment.status.name,
                )
                break
        return experiment

    # --- Stage transitions -------------------------------------------------

    def _advance_sample_generation(self, experiment: Experiment, full: bool, redo: bool):
        if redo:
            experiment.sample_generation_job_ids = []
            store.update_experiment(experiment)

        if experiment.training_samples_gcs_uri and not redo:
            logging.info(
                "Reusing existing training samples: %s", experiment.training_samples_gcs_uri
            )
            experiment.status = ExperimentStatus.CREATE_TUNING_DATA
            store.update_experiment(experiment)
            return

        if not experiment.sample_generation_job_ids:
            self._generate_samples(experiment, full=full)
            # Online generation finishes immediately; batch jobs do not.
            if self.config.online_inference:
                experiment.status = ExperimentStatus.CREATE_TUNING_DATA
                store.update_experiment(experiment)
            return

        statuses = [self._job_status(job) for job in self._sample_generation_jobs(experiment)]
        if any(status in (JobStatus.FAILED, JobStatus.CANCELLED) for status in statuses):
            raise RuntimeError("One or more sample generation jobs failed.")
        if all(status == JobStatus.SUCCEEDED for status in statuses):
            experiment.status = ExperimentStatus.CREATE_TUNING_DATA
            store.update_experiment(experiment)

    def _advance_tuning(self, experiment: Experiment, redo: bool):
        if redo:
            experiment.tuning_job_id = None
            store.update_experiment(experiment)

        if not experiment.tuning_job_id:
            self._tune_model(experiment)
            return

        job = self.provider.get_tuning_job(experiment.tuning_job_id)
        status = job_status_from_state(getattr(job.state, "name", str(job.state)))
        if status == JobStatus.SUCCEEDED:
            experiment.status = ExperimentStatus.INFERENCE
            store.update_experiment(experiment)
        elif status in (JobStatus.FAILED, JobStatus.CANCELLED):
            raise RuntimeError(f"Tuning job failed with status: {status.value}")

    def _advance_inference(self, experiment: Experiment, full: bool, redo: bool):
        if redo:
            experiment.validation_job_ids = []
            store.update_experiment(experiment)

        datasets = self.prepare_datasets(full=full)

        if not experiment.validation_job_ids:
            self._run_inference(experiment, datasets)
            if self.config.online_inference:
                experiment.status = ExperimentStatus.COMPLETE
                store.update_experiment(experiment)
            return

        jobs = [self.provider.get_batch_job(job_id) for job_id in experiment.validation_job_ids]
        statuses = [self._job_status(job) for job in jobs]
        if any(status in (JobStatus.FAILED, JobStatus.CANCELLED) for status in statuses):
            raise RuntimeError("One or more validation jobs failed.")
        if all(status == JobStatus.SUCCEEDED for status in statuses):
            self._process_completed_validation_jobs(
                experiment, jobs, validation_dataset_gcs_uri=datasets.validation_gcs_uri
            )
            experiment.status = ExperimentStatus.COMPLETE
            store.update_experiment(experiment)

    # --- Datasets ----------------------------------------------------------

    def prepare_datasets(self, full: bool = False) -> PreparedDatasets:
        """Stage the training and evaluation datasets in GCS."""
        dataset_provider = HuggingFaceDatasetProvider(
            gcs_client=self.gcs_client, gcs_bucket_name=self.config.gcs_bucket
        )
        training_uri, training_examples, _ = dataset_provider.get_dataset(
            self.config.training_dataset.name,
            self.config.training_dataset.split,
            full=full,
            dev_set_size=self.config.dev_set_size,
            candidates_field=self.config.training_dataset.get("candidates_field"),
            target_field=self.config.training_dataset.answer_field_contains,
        )

        evaluation_dataset = self.config.validation_dataset
        halve = True
        if self.config.test_split and self.config.test_dataset and self.config.test_dataset.name:
            # An explicit test set needs no halving.
            evaluation_dataset = self.config.test_dataset
            halve = False

        validation_uri, validation_examples, _ = dataset_provider.get_dataset(
            evaluation_dataset.name,
            evaluation_dataset.split,
            full=full,
            dev_set_size=self.config.dev_set_size,
            candidates_field=evaluation_dataset.get("candidates_field"),
            target_field=evaluation_dataset.answer_field_contains,
            halve=halve,
            test_split=self.config.test_split,
        )
        return PreparedDatasets(
            training_gcs_uri=training_uri,
            training_examples=training_examples,
            validation_gcs_uri=validation_uri,
            validation_examples=validation_examples,
        )

    def _prompted_dataset(self, gcs_uri: str, **prompt_kwargs) -> list[dict]:
        """Load a staged dataset and set each example's prompt.

        The prompt is the key that joins samples, ground truth and predictions,
        so it is rebuilt the same way everywhere, and duplicate questions are
        dropped so that a prompt identifies exactly one example.
        """
        dataset = read_jsonl(self.gcs_client, gcs_uri)
        for example in dataset:
            example["inputs"] = get_prompt_with_instructions(
                example, self.config, **prompt_kwargs
            )
        return self._deduplicate(dataset)

    @staticmethod
    def _deduplicate(dataset: list[dict]) -> list[dict]:
        seen, deduplicated = set(), []
        for example in dataset:
            if example["inputs"] not in seen:
                seen.add(example["inputs"])
                deduplicated.append(example)
        if len(deduplicated) < len(dataset):
            logging.warning(
                "Dropped %d duplicate prompts from the dataset.",
                len(dataset) - len(deduplicated),
            )
        return deduplicated

    # --- Stage 1: sampling the teacher --------------------------------------

    def _generate_samples(self, experiment: Experiment, full: bool = False):
        """Sample ``n_generations`` responses per training question."""
        logging.info("Generating samples for experiment %s", experiment.id)
        datasets = self.prepare_datasets(full=full)
        training_dataset = self._prompted_dataset(datasets.training_gcs_uri)
        schema = get_mmlu_response_model(
            self.config.confidence_level, include_reasoning=self.config.include_reasoning
        )

        if self.config.online_inference:
            generations = []
            for example in tqdm(training_dataset, desc="Sampling the teacher"):
                for response in self._generate_online(
                    self.config.base_model, example["inputs"], schema, self.config.n_generations
                ):
                    generation = {"inputs": example["inputs"], "targets": response}
                    if "candidates" in example:
                        generation["candidates"] = example["candidates"]
                    generations.append(generation)

            output_gcs_uri = self._uri(experiment, "training_sample_generations/online.jsonl")
            write_jsonl(self.gcs_client, output_gcs_uri, generations)
            logging.info("Wrote %d samples to %s", len(generations), output_gcs_uri)
            # Stored as a list of URIs, which stands in for a finished job.
            experiment.sample_generation_job_ids.append([output_gcs_uri])
        else:
            self._submit_batch_jobs(
                experiment,
                dataset=training_dataset,
                model=self.config.base_model,
                schema=schema,
                n_samples=self.config.n_generations,
                output_prefix="sample_generations",
                display_name_prefix="training",
                job_ids=experiment.sample_generation_job_ids,
            )
        store.update_experiment(experiment)

    def _generate_online(self, model, prompt: str, schema, n_samples: int) -> list[str]:
        """Draw ``n_samples`` responses for one prompt, in as few calls as possible."""
        generation_config = OmegaConf.to_container(self.config.generation_config, resolve=True)
        generation_config["candidate_count"] = min(n_samples, MAX_CANDIDATE_COUNT)
        n_calls = -(-n_samples // generation_config["candidate_count"])

        responses = []
        for _ in range(n_calls):
            candidates, _ = self.provider.generate_online(
                model,
                prompt,
                generation_config=generation_config,
                response_schema=schema,
                response_mime_type="application/json",
            )
            responses.extend(candidates)
        return responses[:n_samples]

    def _submit_batch_jobs(
        self,
        experiment: Experiment,
        dataset: list[dict],
        model,
        schema,
        n_samples: int,
        output_prefix: str,
        display_name_prefix: str,
        job_ids: list,
    ):
        """Shard a dataset into batch prediction jobs."""
        generation_config = OmegaConf.to_container(self.config.generation_config, resolve=True)
        generation_config.pop("thinking_budget", None)
        generation_config["candidate_count"] = min(n_samples, MAX_CANDIDATE_COUNT)
        generation_config["response_mime_type"] = "application/json"
        generation_config["response_schema"] = schema.model_json_schema()
        n_calls = -(-n_samples // generation_config["candidate_count"])

        request_id = 0
        for shard_index, start in enumerate(range(0, len(dataset), BATCH_SHARD_SIZE)):
            requests = []
            for example in dataset[start : start + BATCH_SHARD_SIZE]:
                for _ in range(n_calls):
                    requests.append(
                        {
                            "id": request_id,
                            "request": {
                                "contents": [
                                    {"role": "user", "parts": [{"text": example["inputs"]}]}
                                ],
                                "generation_config": generation_config,
                            },
                        }
                    )
                    request_id += 1

            shard_uri = self._uri(experiment, f"{output_prefix}/{shard_index}/")
            write_jsonl(self.gcs_client, f"{shard_uri}input.jsonl", requests)
            job = self.provider.create_batch_job(
                model,
                f"{shard_uri}input.jsonl",
                shard_uri,
                display_name=f"{experiment.id}_{display_name_prefix}_shard_{shard_index}",
            )
            job_ids.append(job.name)
            logging.info("Created batch job %s for shard %d", job.name, shard_index)

    # --- Stage 2: building the augmented dataset ----------------------------

    def _create_augmented_dataset_from_samples(
        self, experiment: Experiment, jobs: list, full: bool = False
    ):
        """Calibrate the samples and write the augmented training set."""
        logging.info("Creating augmented dataset from %d sample jobs.", len(jobs))

        generations_by_prompt = collections.defaultdict(list)
        for job in jobs:
            for prompt, response in self._job_outputs(job):
                generations_by_prompt[prompt].append(response)

        # More samples than requested means a prompt appeared twice; keep the
        # first n_generations so every prompt carries the same weight.
        generations, raw_generations = {}, []
        for prompt, responses in generations_by_prompt.items():
            responses = responses[: self.config.n_generations]
            generations[prompt] = responses
            raw_generations.extend({"inputs": prompt, "targets": r} for r in responses)
        logging.info("Collected generations for %d unique prompts.", len(generations))

        experiment.training_samples_gcs_uri = self._uri(experiment, "training_samples.jsonl")
        write_jsonl(self.gcs_client, experiment.training_samples_gcs_uri, raw_generations)

        datasets = self.prepare_datasets(full=full)
        training_dataset = self._prompted_dataset(datasets.training_gcs_uri)
        prompt_to_example_map = {example["inputs"]: example for example in training_dataset}
        ground_truth_map = {
            prompt: example["targets"] for prompt, example in prompt_to_example_map.items()
        }

        calibrator, training_generations = train_and_validate_calibrator(
            generations,
            self.config,
            self.augmentation_manager,
            ground_truth_map,
            prompt_to_example_map,
        )

        statistics = self.augmentation_manager.accumulate_statistics(
            training_generations,
            ground_truth_map=ground_truth_map,
            prompt_to_example_map=prompt_to_example_map,
        )

        # How good the teacher's own sampled distribution is: the ceiling that
        # distillation is trying to transfer into the student.
        evaluator = Evaluator(
            confidence_bins=self.augmentation_manager.get_confidence_bins(),
            config=self.config,
            normalizer=self.normalizer,
        )
        experiment.sample_generation_metrics = evaluator.evaluate_from_statistics(
            statistics, ground_truth_map=ground_truth_map, prompt_to_example_map=prompt_to_example_map
        )
        logging.info("Sample generation metrics: %s", experiment.sample_generation_metrics)

        (
            augmented_dataset,
            missing_ground_truth_stats,
            augmentation_stats,
        ) = self.augmentation_manager.create_augmented_training_set(
            statistics,
            training_generations,
            calibrator=calibrator,
            prompt_to_example_map=prompt_to_example_map,
        )
        experiment.missing_ground_truth_stats = missing_ground_truth_stats
        experiment.augmentation_stats = augmentation_stats

        if not augmented_dataset:
            raise ValueError("Augmented dataset is empty.")

        experiment.augmented_training_dataset_gcs_uri = self._uri(
            experiment, "augmented_training.jsonl"
        )
        write_jsonl(
            self.gcs_client, experiment.augmented_training_dataset_gcs_uri, augmented_dataset
        )
        logging.info(
            "Wrote %d augmented examples to %s",
            len(augmented_dataset),
            experiment.augmented_training_dataset_gcs_uri,
        )

    # --- Stage 3: tuning ----------------------------------------------------

    def _tune_model(self, experiment: Experiment):
        """Split the augmented data and submit a supervised tuning job."""
        augmented_dataset = read_jsonl(
            self.gcs_client, experiment.augmented_training_dataset_gcs_uri
        )
        random.Random(42).shuffle(augmented_dataset)

        num_validation = max(
            1, int(len(augmented_dataset) * self.config.tuning_validation_split)
        )
        tuning_validation = augmented_dataset[:num_validation]
        tuning_training = augmented_dataset[num_validation:]
        logging.info(
            "Tuning on %d examples, validating on %d.",
            len(tuning_training),
            len(tuning_validation),
        )

        experiment.tuning_training_dataset_gcs_uri = self._uri(experiment, "tuning_training.jsonl")
        write_jsonl(self.gcs_client, experiment.tuning_training_dataset_gcs_uri, tuning_training)
        experiment.tuning_validation_dataset_gcs_uri = self._uri(
            experiment, "tuning_validation.jsonl"
        )
        write_jsonl(
            self.gcs_client, experiment.tuning_validation_dataset_gcs_uri, tuning_validation
        )
        experiment.tuned_model_output_gcs_uri = self._uri(experiment, "tuned_models")

        logging.info("Submitting tuning job for experiment %s...", experiment.id)
        tuning_job = self.provider.tune(
            self.config.base_model,
            self.provider.get_tuning_dataset(experiment.tuning_training_dataset_gcs_uri),
            self.provider.get_tuning_dataset(experiment.tuning_validation_dataset_gcs_uri),
            self.config.tuned_model_display_name,
            epochs=self.config.epochs,
            learning_rate_multiplier=self.config.learning_rate_multiplier,
            adapter_size=self.config.adapter_size,
        )
        experiment.tuning_job_id = tuning_job.name
        store.update_experiment(experiment)

    # --- Stage 4: inference and evaluation -----------------------------------

    def _run_inference(self, experiment: Experiment, datasets: PreparedDatasets):
        if self.config.experiment_type == ExperimentType.BASELINE:
            model = self.config.base_model
        else:
            model = self.provider.get_tuned_model(experiment.tuning_job_id)
            if not model:
                raise ValueError("Tuned model not available for inference.")
        self._evaluate_model(experiment, model, datasets.validation_gcs_uri)

    def _evaluate_model(self, experiment: Experiment, model, validation_dataset_gcs_uri: str):
        """Predict on the evaluation set, online or as a batch job."""
        use_binned = self.config.augmentation_scheme in BINNED_SCHEMES
        schema = get_mmlu_response_model(
            self.config.confidence_level,
            include_reasoning=self.config.include_reasoning,
            binned_answers=use_binned,
        )
        validation_dataset = self._prompted_dataset(
            validation_dataset_gcs_uri, binned_answers=use_binned
        )

        if self.config.online_inference:
            # One prediction per example: the student answers once and states
            # its confidence, which is exactly what the metrics score.
            predictions = []
            for example in tqdm(validation_dataset, desc="Running inference"):
                for response in self._generate_online(model, example["inputs"], schema, 1):
                    predictions.append({"inputs": example["inputs"], "targets": response})

            output_gcs_uri = self._uri(
                experiment, f"{self._inference_dir()}/online.jsonl"
            )
            write_jsonl(self.gcs_client, output_gcs_uri, predictions)
            self._process_completed_validation_jobs(
                experiment,
                [JobResult(name="online_evaluation", status=JobStatus.SUCCEEDED,
                           dest_uris=[output_gcs_uri])],
                validation_dataset_gcs_uri=validation_dataset_gcs_uri,
            )
        else:
            self._submit_batch_jobs(
                experiment,
                dataset=validation_dataset,
                model=model,
                schema=schema,
                n_samples=1,
                output_prefix=self._inference_dir(),
                display_name_prefix="evaluation",
                job_ids=experiment.validation_job_ids,
            )
            store.update_experiment(experiment)

    def _process_completed_validation_jobs(
        self, experiment: Experiment, jobs: list, validation_dataset_gcs_uri: str
    ):
        """Join predictions back to their examples and compute the metrics."""
        use_binned = self.config.augmentation_scheme in BINNED_SCHEMES
        examples_by_prompt = {
            example["inputs"]: example
            for example in self._prompted_dataset(
                validation_dataset_gcs_uri, binned_answers=use_binned
            )
        }

        warnings = collections.Counter()
        merged_dataset = []
        for job in jobs:
            for prompt, response in self._job_outputs(job):
                example = examples_by_prompt.get(prompt)
                if example is None:
                    warnings["missing_example"] += 1
                    continue
                merged_dataset.append({**example, "prediction": response})

        for reason, count in warnings.items():
            logging.warning("%d predictions skipped (%s).", count, reason)

        logging.info(
            "Scoring %d predictions for %d evaluation examples.",
            len(merged_dataset),
            len(examples_by_prompt),
        )
        if not merged_dataset:
            raise ValueError(
                "No predictions could be matched to evaluation examples. "
                "Check that the inference stage completed."
            )

        evaluator = Evaluator(
            confidence_bins=self.augmentation_manager.get_confidence_bins(),
            config=self.config,
            normalizer=self.normalizer,
        )
        metrics = evaluator.evaluate(merged_dataset)

        experiment.predictions_dataset_gcs_uri = self._uri(experiment, "predictions.jsonl")
        write_jsonl(self.gcs_client, experiment.predictions_dataset_gcs_uri, merged_dataset)

        print("Test metrics:" if self.config.test_split else "Validation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        metrics_uri = self._uri(
            experiment, "test_metrics.jsonl" if self.config.test_split else "validation_metrics.jsonl"
        )
        write_jsonl(self.gcs_client, metrics_uri, [metrics])
        if self.config.test_split:
            experiment.test_metrics_gcs_uri, experiment.test_metrics = metrics_uri, metrics
        else:
            experiment.validation_metrics_gcs_uri, experiment.validation_metrics = metrics_uri, metrics
        store.update_experiment(experiment)

    # --- Job plumbing --------------------------------------------------------

    def _uri(self, experiment: Experiment, suffix: str) -> str:
        return f"gs://{self.config.gcs_bucket}/{experiment.id}/{suffix}"

    def _inference_dir(self) -> str:
        return "test_generations" if self.config.test_split else "validation_generations"

    def _sample_generation_jobs(self, experiment: Experiment) -> list:
        """Rebuild job handles from the ids stored on the experiment."""
        if not experiment.sample_generation_job_ids and experiment.training_samples_gcs_uri:
            return [
                JobResult(
                    name="reused_generations",
                    status=JobStatus.SUCCEEDED,
                    dest_uris=[experiment.training_samples_gcs_uri],
                )
            ]

        jobs = []
        for job_id in experiment.sample_generation_job_ids:
            if isinstance(job_id, str):
                jobs.append(self.provider.get_batch_job(job_id))
            else:
                # A list of output URIs written by online generation.
                jobs.append(
                    JobResult(
                        name="online_generation",
                        status=JobStatus.SUCCEEDED,
                        dest_uris=list(job_id),
                    )
                )
        return jobs

    @staticmethod
    def _job_status(job) -> JobStatus:
        if isinstance(job, JobResult):
            return job.status
        return job_status_from_state(getattr(job.state, "name", str(job.state)))

    def _job_outputs(self, job) -> list[tuple[str, str]]:
        """Return (prompt, response) pairs for a finished job.

        Online output already carries its prompt. Batch output carries only the
        request id, so it is matched against the input file we uploaded.
        """
        if isinstance(job, JobResult):
            pairs = []
            for uri in job.dest_uris:
                for item in read_jsonl(self.gcs_client, uri):
                    pairs.append((item["inputs"], item["targets"]))
            return pairs

        source_uris = job.src.gcs_uri
        source_uris = [source_uris] if isinstance(source_uris, str) else source_uris
        id_to_prompt = {}
        for uri in source_uris:
            for item in read_jsonl(self.gcs_client, uri):
                id_to_prompt[str(item["id"])] = item["request"]["contents"][0]["parts"][0]["text"]

        destination_uris = job.dest.gcs_uri
        destination_uris = (
            [destination_uris] if isinstance(destination_uris, str) else destination_uris
        )
        pairs, warnings = [], collections.Counter()
        for uri in destination_uris:
            for item in read_jsonl(self.gcs_client, uri):
                if "error" in item:
                    warnings["batch error"] += 1
                    continue
                prompt = id_to_prompt.get(str(item["id"]))
                if prompt is None:
                    warnings["unmatched response id"] += 1
                    continue
                for response in self._responses_from(item.get("response", item)):
                    pairs.append((prompt, response))

        for reason, count in warnings.items():
            logging.warning("%d batch responses skipped (%s).", count, reason)
        return pairs

    @staticmethod
    def _responses_from(response_data: dict) -> list[str]:
        """Pull the generated texts out of one batch response row."""
        if response_data.get("candidates"):
            try:
                return [c["content"]["parts"][0]["text"] for c in response_data["candidates"]]
            except (KeyError, IndexError) as error:
                raise ValueError(f"Malformed batch response: {response_data}") from error
        if "targets" in response_data:
            return [response_data["targets"]]
        raise ValueError(f"Unknown response format: {response_data}")
