"""Scoring the student's predictions and its verbalized confidence."""

import json
import logging

import numpy as np
from sklearn.metrics import roc_auc_score

from .normalization import StringNormalizer

#: Number of top bins treated as "the model said it was confident".
HIGH_CONFIDENCE_BINS = 2


class Evaluator:
    """Accuracy, high-confidence accuracy and AUROC for multiple choice.

    The verbalized bin is mapped back to a number by its position
    (``index / (n_bins - 1)``), which is the score AUROC is computed against.
    """

    def __init__(self, confidence_bins: list[str], config, normalizer=None):
        self.confidence_bins = confidence_bins
        self.config = config
        self.normalizer = normalizer or StringNormalizer()
        self.confidence_bin_to_score = {
            bin_name: index / (len(confidence_bins) - 1)
            for index, bin_name in enumerate(confidence_bins)
        }

    def _answer_to_letter_map(self, candidates: dict[str, str]) -> dict[str, str]:
        return {
            self.normalizer.normalize(text): self.normalizer.normalize(letter)
            for letter, text in candidates.items()
        }

    def _target_answer(self, example: dict, target: str | None = None) -> str:
        target = example["targets"] if target is None else target
        if self.config.training_dataset.answer_field_contains == "full_answer":
            return self._answer_to_letter_map(example["candidates"])[
                self.normalizer.normalize(target)
            ]
        return self.normalizer.normalize(target)

    def _parse_prediction(self, prediction: str) -> tuple[str, str, str] | None:
        """Return (answer, reasoning, confidence bin) for one prediction.

        Handles both response formats: a single answer with a ``confidence``
        field, and the binned format whose keys are the bins themselves (there,
        the model's answer is the one in the highest occupied bin).
        """
        try:
            data = json.loads(prediction)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None

        for bin_name in reversed(self.confidence_bins):
            value = data.get(bin_name.replace(" ", "_"))
            if value:
                return str(value).strip(), "", bin_name

        confidence_bin = str(data.get("confidence", "MEDIUM")).strip()
        if confidence_bin not in self.confidence_bin_to_score:
            confidence_bin = "MEDIUM"
        return (
            str(data.get("answer", "")).strip(),
            str(data.get("reasoning", "")).strip(),
            confidence_bin,
        )

    def evaluate(self, dataset: list[dict]) -> dict:
        """Score predictions that have been joined back to their examples."""
        confidences, correctness, high_confidence_correctness = [], [], []
        non_parsable_count = 0

        for example in dataset:
            parsed = self._parse_prediction(example["prediction"])
            if parsed is None:
                if non_parsable_count == 0:
                    logging.warning("First unparsable prediction: %s", example["prediction"])
                non_parsable_count += 1
                continue

            answer, reasoning, confidence_bin = parsed
            if not answer or (self.config.include_reasoning and not reasoning):
                non_parsable_count += 1
                continue

            confidences.append(self.confidence_bin_to_score[confidence_bin])
            is_correct = int(
                self.normalizer.normalize(answer) == self._target_answer(example)
            )
            correctness.append(is_correct)

            if self.confidence_bins.index(confidence_bin) >= len(self.confidence_bins) - HIGH_CONFIDENCE_BINS:
                high_confidence_correctness.append(is_correct)

        return {
            "accuracy": float(np.mean(correctness)) if correctness else 0.0,
            "high_accuracy": (
                float(np.mean(high_confidence_correctness))
                if high_confidence_correctness
                else 0.0
            ),
            "auroc": (
                float(roc_auc_score(correctness, confidences))
                if len(np.unique(correctness)) > 1
                else 0.0
            ),
            "non_parsable_count": non_parsable_count,
            "total_examples": len(dataset),
        }

    def evaluate_from_statistics(
        self,
        statistics: dict[str, dict[str, dict]],
        ground_truth_map: dict[str, str],
        prompt_to_example_map: dict[str, dict] | None = None,
    ) -> dict:
        """Score the *teacher's* samples, before any distillation.

        The prediction is the most frequently sampled answer and the confidence
        is how often it was sampled, which is the signal distillation transfers.
        """
        confidences, correctness = [], []
        for prompt, answers in statistics.items():
            most_likely = max(answers, key=lambda answer: answers[answer]["count"])
            total = sum(answer["count"] for answer in answers.values())
            confidences.append(answers[most_likely]["count"] / total)

            example = (prompt_to_example_map or {}).get(prompt, {})
            gold = self._target_answer(example, ground_truth_map[prompt])
            correctness.append(int(self.normalizer.normalize(most_likely) == gold))

        return {
            "accuracy": float(np.mean(correctness)) if correctness else 0.0,
            "auroc": (
                float(roc_auc_score(correctness, confidences))
                if len(np.unique(correctness)) > 1
                else 0.0
            ),
        }
