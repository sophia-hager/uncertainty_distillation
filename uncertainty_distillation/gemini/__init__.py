"""Uncertainty distillation for API-served models (Gemini on Vertex AI).

This is the same method as the local-model pipeline in the parent package, but
for a model you can only reach through an API: the teacher is sampled with the
Gemini API, the fine-tuning data is staged in Google Cloud Storage, and the
student is trained with Vertex AI supervised tuning.

The stages mirror the local pipeline:

1. ``GENERATING_SAMPLES``  -- sample ``n_generations`` structured responses per
   question from the base model;
2. ``CREATE_TUNING_DATA``  -- calibrate the resulting answer frequencies, bin
   them, and rewrite each response with its verbalized confidence;
3. ``TUNING``              -- supervised fine-tuning of the base model;
4. ``INFERENCE``           -- predict on held-out data;
5. ``COMPLETE``            -- accuracy, high-confidence accuracy and AUROC.

Because tuning and batch inference are long-running asynchronous jobs, an
experiment is a resumable record: each run advances it as far as it can and
stores its state, and running it again picks up where it left off.
"""

__all__ = [
    "augmentation",
    "calibration",
    "config",
    "datasets",
    "evaluator",
    "experiment",
    "normalization",
    "pipeline",
    "prompts",
    "provider",
    "schemas",
    "storage",
    "store",
]
