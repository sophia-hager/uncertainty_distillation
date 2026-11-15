"""Turning teacher samples into an augmented, confidence-labelled training set.

For each question the teacher's samples give a distribution over answers.  The
frequency of an answer is calibrated, mapped to a confidence bin, and written
back into one of the sampled responses as its ``confidence`` field.  The result
is a supervised example that keeps the model's own wording and reasoning but
states a confidence the model has no other way of knowing.
"""

import collections
import json
import logging
import random

from .calibration import calibrate
from .normalization import StringNormalizer
from .prompts import get_prompt_with_instructions
from .schemas import get_confidence_bins

#: Schemes whose target is a map from every confidence bin to an answer.
BINNED_SCHEMES = ("BINNED_ANSWERS", "SPARSE_BINNED_ANSWERS")


def _parse_response(response: str) -> tuple[str, str] | None:
    """Return (answer, reasoning) for a JSON response, or None if malformed."""
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return str(data.get("answer", "")).strip(), str(data.get("reasoning", "")).strip()


class AugmentationManager:
    """Builds the augmented training set for multiple-choice tasks."""

    def __init__(self, config, normalizer=None):
        self.config = config
        self.normalizer = normalizer or StringNormalizer()

    # --- Bins -----------------------------------------------------------

    def get_confidence_bins(self) -> list[str]:
        return get_confidence_bins(self.config.confidence_level)

    def bin_index(self, confidence: float) -> int:
        """Linear mapping of a probability in [0, 1] to a bin index."""
        bins = self.get_confidence_bins()
        return min(int(confidence * len(bins)), len(bins) - 1)

    def verbalize(self, confidence: float) -> str:
        return self.get_confidence_bins()[self.bin_index(confidence)]

    # --- Ground truth ----------------------------------------------------

    def _answer_to_letter_map(self, candidates: dict[str, str]) -> dict[str, str]:
        """Map each option's text to its letter, both normalized."""
        return {
            self.normalizer.normalize(text): self.normalizer.normalize(letter)
            for letter, text in candidates.items()
        }

    def ground_truth_answer(self, example: dict, target: str | None = None) -> str:
        """The gold answer in the same space as the model's parsed answers.

        Datasets that give the answer as the option's full text are mapped back
        to the option letter, because that is what the model is asked to emit.
        """
        target = example["targets"] if target is None else target
        if self.config.training_dataset.answer_field_contains == "full_answer":
            return self._answer_to_letter_map(example["candidates"])[
                self.normalizer.normalize(target)
            ]
        return self.normalizer.normalize(target)

    # --- Statistics ------------------------------------------------------

    def accumulate_statistics(
        self,
        generations: dict[str, list[str]],
        ground_truth_map: dict[str, str] | None = None,
        prompt_to_example_map: dict[str, dict] | None = None,
    ) -> dict[str, dict[str, dict]]:
        """Count, per prompt, how often each distinct answer was sampled."""
        statistics = collections.defaultdict(
            lambda: collections.defaultdict(lambda: {"count": 0, "reasonings": [], "is_correct": 0})
        )
        for prompt, responses in generations.items():
            for response in responses:
                parsed = _parse_response(response)
                if parsed is None:
                    continue
                answer, reasoning = parsed
                if not answer:
                    continue
                if self.config.include_reasoning and not reasoning:
                    continue

                normalized = self.normalizer.normalize(answer)
                statistics[prompt][normalized]["count"] += 1
                statistics[prompt][normalized]["reasonings"].append(reasoning)

                if ground_truth_map:
                    try:
                        example = (prompt_to_example_map or {}).get(prompt, {})
                        gold = self.ground_truth_answer(example, ground_truth_map[prompt])
                    except KeyError as error:
                        raise ValueError(
                            "A prompt from the reused generations is not in the current dataset. "
                            "This usually means the prompt configuration (such as include_reasoning) "
                            "changed since the samples were generated, so they cannot be reused."
                        ) from error
                    statistics[prompt][normalized]["is_correct"] = int(normalized == gold)
        return statistics

    # --- Augmented dataset ------------------------------------------------

    def create_augmented_training_set(
        self,
        statistics: dict[str, dict[str, dict]],
        generations: dict[str, list[str]],
        calibrator=None,
        prompt_to_example_map: dict[str, dict] | None = None,
    ) -> tuple[list[dict], dict, dict]:
        """Build the dataset, and report what went into it."""
        augmented_dataset = []
        missing_ground_truth_stats = {"count": 0}
        stats = {
            "total_examples": 0,
            "correct_examples": 0,
            "incorrect_examples": 0,
            "malformed_responses": 0,
            "pre_calibration_bin_counts": collections.defaultdict(int),
            "post_calibration_bin_counts": collections.defaultdict(int),
        }
        use_binned = self.config.augmentation_scheme in BINNED_SCHEMES

        for prompt, answers in statistics.items():
            example = prompt_to_example_map[prompt]
            gold = self.ground_truth_answer(example)
            total = sum(answer["count"] for answer in answers.values())
            prompt_text = (
                get_prompt_with_instructions(example, self.config, binned_answers=True)
                if use_binned
                else prompt
            )

            if gold not in answers:
                missing_ground_truth_stats["count"] += 1

            if use_binned:
                augmented_dataset.append(
                    self._create_binned_example(answers, total, calibrator, prompt_text)
                )
                stats["total_examples"] += 1
                continue

            valid_generations = self._valid_generations(generations[prompt], stats)

            if gold in answers:
                self._record_bins(answers[gold]["count"] / total, calibrator, stats)
                stats["correct_examples"] += 1
                stats["total_examples"] += 1
                augmented_dataset.append(
                    self._create_augmented_example(
                        confidence=answers[gold]["count"] / total,
                        calibrator=calibrator,
                        response=self._response_for(valid_generations, gold),
                        prompt_text=prompt_text,
                    )
                )

            for answer in self._incorrect_answers_to_add(answers, gold, total, calibrator):
                self._record_bins(answers[answer]["count"] / total, calibrator, stats)
                stats["incorrect_examples"] += 1
                stats["total_examples"] += 1
                augmented_dataset.append(
                    self._create_augmented_example(
                        confidence=answers[answer]["count"] / total,
                        calibrator=calibrator,
                        response=self._response_for(valid_generations, answer),
                        prompt_text=prompt_text,
                    )
                )

        logging.info("Augmentation stats: %s", dict(stats))
        return augmented_dataset, missing_ground_truth_stats, dict(stats)

    def _valid_generations(self, responses: list[str], stats: dict) -> list[str]:
        """Drop responses that are not usable, counting them as malformed."""
        valid = []
        for response in responses:
            parsed = _parse_response(response)
            if parsed is None or not parsed[0]:
                stats["malformed_responses"] += 1
                continue
            if self.config.include_reasoning and not parsed[1]:
                stats["malformed_responses"] += 1
                continue
            valid.append(response)
        return valid

    def _response_for(self, valid_generations: list[str], answer: str) -> str:
        """The first sampled response that gave this answer."""
        for response in valid_generations:
            parsed = _parse_response(response)
            if parsed and self.normalizer.normalize(parsed[0]) == self.normalizer.normalize(answer):
                return response
        raise ValueError(f"No sampled response found for answer: {answer}")

    def _incorrect_answers_to_add(self, answers, gold, total, calibrator) -> list[str]:
        """Which wrong answers become training examples, per the scheme."""
        incorrect = [answer for answer in answers if answer != gold]
        scheme = self.config.augmentation_scheme

        if scheme == "CORRECT_PLUS_N":
            return random.sample(
                incorrect, min(self.config.augmentation_scheme_n, len(incorrect))
            )

        if scheme == "STRATIFIED":
            # One wrong answer per occupied confidence bin, so the student sees
            # wrong answers across the whole confidence range.
            by_bin = collections.defaultdict(list)
            for answer in incorrect:
                confidence = calibrate(calibrator, answers[answer]["count"] / total)
                by_bin[self.bin_index(confidence)].append(answer)
            return [random.choice(group) for group in by_bin.values()]

        if scheme == "CORRECT_ONLY":
            return []

        raise ValueError(f"Unknown augmentation scheme: {scheme}")

    def _record_bins(self, confidence: float, calibrator, stats: dict) -> None:
        """Track bin occupancy before and after calibration."""
        bins = self.get_confidence_bins()
        stats["pre_calibration_bin_counts"][bins[self.bin_index(confidence)]] += 1
        if calibrator:
            calibrated = calibrate(calibrator, confidence)
            stats["post_calibration_bin_counts"][bins[self.bin_index(calibrated)]] += 1

    def _create_binned_example(self, answers, total, calibrator, prompt_text) -> dict:
        """One example whose target lists an answer for each confidence bin."""
        bins = self.get_confidence_bins()

        # Sort by sample count so that, when two answers land in the same bin,
        # the more frequent one wins it.
        candidates = sorted(
            (
                (self.bin_index(calibrate(calibrator, stats["count"] / total)), stats["count"], answer)
                for answer, stats in answers.items()
            ),
            key=lambda candidate: candidate[1],
            reverse=True,
        )

        binned = {}
        for bin_index, _, answer in candidates:
            binned.setdefault(bins[bin_index], answer)

        if self.config.augmentation_scheme == "SPARSE_BINNED_ANSWERS":
            target = {name.replace(" ", "_"): answer for name, answer in binned.items()}
        else:
            target = {name.replace(" ", "_"): binned.get(name) for name in bins}

        return _tuning_example(prompt_text, json.dumps(target))

    def _create_augmented_example(
        self, confidence: float, calibrator, response: str, prompt_text: str
    ) -> dict:
        """Rewrite a sampled response with its calibrated, verbalized confidence."""
        confidence_bin = self.verbalize(calibrate(calibrator, confidence))

        response_data = json.loads(response)
        response_data["confidence"] = confidence_bin
        return _tuning_example(prompt_text, json.dumps(response_data))


def _tuning_example(prompt_text: str, target_text: str) -> dict:
    """One example in the format Vertex AI supervised tuning expects."""
    return {
        "contents": [
            {"role": "user", "parts": [{"text": prompt_text}]},
            {"role": "model", "parts": [{"text": target_text}]},
        ]
    }
