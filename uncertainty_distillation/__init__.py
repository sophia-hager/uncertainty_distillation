"""Uncertainty distillation: teach an LLM to verbalize calibrated confidence.

The pipeline has four stages:

1. ``scripts/sample_teacher.py``  -- sample many answers per question from the
   teacher model to obtain an empirical predictive distribution.
2. ``scripts/build_finetuning_data.py`` -- calibrate that distribution, bin it,
   and turn it into (prompt, completion) pairs with verbalized confidence.
3. ``scripts/train.py`` -- supervised fine-tuning of the student on those pairs.
4. ``scripts/predict.py`` + ``scripts/evaluate.py`` -- generate and score
   predictions, including calibration/AUROC of the verbalized confidences.
"""

__all__ = [
    "calibration",
    "chat",
    "data",
    "evaluation",
    "judge",
    "parsing",
    "paths",
    "prompts",
    "semantic_clustering",
]
