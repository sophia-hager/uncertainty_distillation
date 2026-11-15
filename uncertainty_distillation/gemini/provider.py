"""Google Gen AI (Vertex AI) provider: sampling, tuning and batch inference.

The Google SDK is imported lazily so that the rest of the package — listing
experiments, inspecting results — works without it installed.
"""

import logging
import time

from omegaconf import OmegaConf

#: Vertex AI names LoRA adapter sizes rather than taking an integer.
ADAPTER_SIZES = {
    1: "ADAPTER_SIZE_ONE",
    2: "ADAPTER_SIZE_TWO",
    4: "ADAPTER_SIZE_FOUR",
    8: "ADAPTER_SIZE_EIGHT",
    16: "ADAPTER_SIZE_SIXTEEN",
    32: "ADAPTER_SIZE_THIRTY_TWO",
}

DEFAULT_LOCATION = "us-east1"

RUNNING_STATES = ("JOB_STATE_PENDING", "JOB_STATE_RUNNING", "JOB_STATE_QUEUED")


def _import_sdk():
    try:
        import vertexai
        from google import genai
        from google.genai import types as genai_types
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The Gemini pipeline needs the Google Gen AI SDK: "
            "pip install -r requirements-gemini.txt"
        ) from error
    return vertexai, genai, genai_types


class GoogleGenAIProvider:
    """Thin wrapper over the Gen AI client for the calls the pipeline makes.

    Credentials come from Application Default Credentials
    (``gcloud auth application-default login``), never from a key in the code.
    """

    def __init__(self, project_id: str | None, location: str | None):
        vertexai, genai, genai_types = _import_sdk()
        self.genai = genai
        self.types = genai_types

        location = location if location and location != "None" else DEFAULT_LOCATION
        vertexai.init(project=project_id, location=location)
        self.client = genai.Client(vertexai=True, project=project_id, location=location)

    # --- Datasets -------------------------------------------------------

    def get_tuning_dataset(self, gcs_uri: str):
        return self.types.TuningDataset(gcs_uri=gcs_uri)

    # --- Tuning ---------------------------------------------------------

    def tune(
        self,
        base_model: str,
        training_dataset,
        validation_dataset,
        tuned_model_display_name: str,
        epochs: int,
        learning_rate_multiplier: float,
        adapter_size: int,
    ):
        if adapter_size not in ADAPTER_SIZES:
            raise ValueError(
                f"Invalid adapter_size: {adapter_size}. Must be one of {sorted(ADAPTER_SIZES)}"
            )
        return self.client.tunings.tune(
            base_model=base_model,
            training_dataset=training_dataset,
            config=self.types.CreateTuningJobConfig(
                tuned_model_display_name=tuned_model_display_name,
                validation_dataset=validation_dataset,
                epoch_count=epochs,
                learning_rate_multiplier=learning_rate_multiplier,
                adapter_size=ADAPTER_SIZES[adapter_size],
            ),
        )

    def get_tuning_job(self, tuning_job_id: str):
        return self.client.tunings.get(name=tuning_job_id)

    def get_tuned_model(self, tuning_job_id: str, timeout: int = 3600):
        """Wait for a tuning job and return its tuned model."""
        job = self.client.tunings.get(name=tuning_job_id)
        start_time = time.time()
        while job.state.name in RUNNING_STATES:
            if time.time() - start_time > timeout:
                raise RuntimeError("Tuning job timed out.")
            logging.info("Waiting for tuning job to complete...")
            time.sleep(10)
            job = self.client.tunings.get(name=job.name)

        if job.state.name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Tuning job failed with state: {job.state.name}")
        return job.tuned_model

    # --- Inference ------------------------------------------------------

    def generate_online(
        self,
        model,
        prompt: str,
        generation_config=None,
        response_schema=None,
        response_mime_type: str | None = None,
    ) -> tuple[list[str], object]:
        """Generate candidates for one prompt and return their texts."""
        model_name = model.endpoint if hasattr(model, "endpoint") else model

        config = None
        if generation_config is not None:
            if OmegaConf.is_config(generation_config):
                generation_config = OmegaConf.to_container(generation_config, resolve=True)
            generation_config = dict(generation_config)
            thinking_budget = generation_config.pop("thinking_budget", 0)
            config = self.types.GenerateContentConfig(
                **generation_config,
                thinking_config=(
                    self.types.ThinkingConfig(thinking_budget=thinking_budget)
                    if thinking_budget > 0
                    else None
                ),
                response_mime_type=response_mime_type,
                response_schema=response_schema,
            )

        response = self.client.models.generate_content(
            model=model_name, contents=prompt, config=config
        )
        return [c.content.parts[0].text for c in response.candidates], response.usage_metadata

    def count_tokens(self, model_name: str, prompt: str) -> int:
        return self.client.models.count_tokens(model=model_name, contents=prompt).total_tokens

    # --- Batch inference -------------------------------------------------

    def create_batch_job(self, model, input_gcs_uri: str, output_gcs_uri: str, display_name: str):
        """Submit a batch prediction job reading from, and writing to, GCS."""
        if hasattr(model, "model"):
            # A tuned model: batch jobs need the model resource, not the endpoint.
            if not model.model:
                raise ValueError(
                    "TunedModel.model is required for batch jobs but is empty. Ensure the "
                    "tuning job completed successfully."
                )
            model = model.model
        elif hasattr(model, "endpoint"):
            raise ValueError(
                "Batch jobs require the TunedModel.model attribute "
                "(projects/.../models/...), but only .endpoint is available."
            )

        return self.client.batches.create(
            model=model,
            src=input_gcs_uri,
            config=self.types.CreateBatchJobConfig(
                dest=output_gcs_uri, display_name=display_name
            ),
        )

    def get_batch_job(self, job_name: str):
        return self.client.batches.get(name=job_name)
