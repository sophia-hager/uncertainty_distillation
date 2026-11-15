"""Post-hoc calibration of the teacher's answer frequencies.

The fraction of samples that chose an answer is only a proxy for the
probability that the answer is right.  A calibrator, fit on a held-out slice of
the teacher's own samples, maps that frequency to a probability of being
correct; the mapped value is what gets binned and verbalized.
"""

import logging

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattScalingCalibrator:
    """Logistic regression on the raw frequency."""

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, probs, labels) -> None:
        self.model.fit(np.asarray(probs, dtype=float).reshape(-1, 1), np.asarray(labels))

    def predict(self, probs):
        return self.model.predict_proba(np.asarray(probs, dtype=float).reshape(-1, 1))[:, 1]


class TemperatureScalingCalibrator(PlattScalingCalibrator):
    """Logistic regression on the logit, with no intercept (a temperature)."""

    def __init__(self):
        self.model = LogisticRegression(fit_intercept=False)

    @staticmethod
    def _to_logits(probs):
        probs = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(probs / (1 - probs))

    def fit(self, probs, labels) -> None:
        self.model.fit(self._to_logits(probs).reshape(-1, 1), np.asarray(labels))

    def predict(self, probs):
        return self.model.predict_proba(self._to_logits(probs).reshape(-1, 1))[:, 1]


class IsotonicRegressionCalibrator:
    """Monotone, non-parametric calibration."""

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)

    def fit(self, probs, labels) -> None:
        self.model.fit(np.asarray(probs, dtype=float).ravel(), np.asarray(labels).ravel())

    def predict(self, probs):
        return self.model.predict(np.asarray(probs, dtype=float).ravel())


CALIBRATORS = {
    "platt_scaling": PlattScalingCalibrator,
    "temperature_scaling": TemperatureScalingCalibrator,
    "isotonic_regression": IsotonicRegressionCalibrator,
}


def make_calibrator(method: str):
    if method.lower() not in CALIBRATORS:
        raise ValueError(f"Invalid calibration method: {method}")
    return CALIBRATORS[method.lower()]()


def calibrate(calibrator, confidence: float) -> float:
    """Map one raw confidence through the calibrator."""
    if calibrator is None:
        return float(confidence)
    return float(np.asarray(calibrator.predict([confidence])).ravel()[0])


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    """Average gap between confidence and accuracy, weighted by bin occupancy."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (probs > lower) & (probs <= upper)
        if in_bin.sum():
            ece += abs(probs[in_bin].mean() - labels[in_bin].mean()) * in_bin.sum() / len(probs)
    return float(ece)


def _confidence_and_correctness(statistics) -> tuple[list[float], list[int]]:
    """Flatten per-prompt statistics into (frequency, is_correct) pairs."""
    confidences, correctness = [], []
    for answers in statistics.values():
        total = sum(answer["count"] for answer in answers.values())
        for stats in answers.values():
            confidences.append(stats["count"] / total)
            correctness.append(stats.get("is_correct", 0))
    return confidences, correctness


def train_and_validate_calibrator(
    generations: dict[str, list[str]],
    config,
    augmentation_manager,
    ground_truth_map: dict[str, str],
    prompt_to_example_map: dict[str, dict],
):
    """Fit the calibrator on held-out prompts and report what it changed.

    Prompts reserved for calibration are excluded from the augmented training
    set, so the confidences the student is trained on are never fit on their own
    data.  Within the reserved pool an 80/20 split is used to report ECE before
    and after calibration; the calibrator is then refit on the whole pool.

    Returns the calibrator (or None) and the generations left for augmentation.
    """
    if not config.calibration.method:
        return None, generations

    calibrator = make_calibrator(config.calibration.method)

    prompts = sorted(generations.keys())
    random_state = np.random.RandomState(42)
    random_state.shuffle(prompts)

    num_calibration = int(len(prompts) * config.calibration.validation_split)
    calibration_prompts = prompts[:num_calibration]
    training_generations = {p: generations[p] for p in prompts[num_calibration:]}

    logging.info(
        "Reserved %d prompts for calibration (split %.2f); %d prompts left for augmentation.",
        len(calibration_prompts),
        config.calibration.validation_split,
        len(training_generations),
    )

    def statistics_for(subset):
        return augmentation_manager.accumulate_statistics(
            {p: generations[p] for p in subset},
            ground_truth_map=ground_truth_map,
            prompt_to_example_map=prompt_to_example_map,
        )

    num_fit = int(len(calibration_prompts) * 0.8)
    fit_confidences, fit_correctness = _confidence_and_correctness(
        statistics_for(calibration_prompts[:num_fit])
    )
    if not fit_confidences:
        logging.warning("No data available for calibration training.")
        return calibrator, training_generations

    calibrator.fit(fit_confidences, fit_correctness)

    held_out_confidences, held_out_correctness = _confidence_and_correctness(
        statistics_for(calibration_prompts[num_fit:])
    )
    if held_out_confidences:
        probs = np.asarray(held_out_confidences)
        labels = np.asarray(held_out_correctness)
        calibrated = np.asarray(calibrator.predict(probs)).ravel()
        logging.info("Calibration validation results:")
        logging.info("  Average accuracy:            %.4f", labels.mean())
        logging.info("  Average confidence (before): %.4f", probs.mean())
        logging.info("  Average confidence (after):  %.4f", calibrated.mean())
        logging.info("  ECE (before):                %.4f", expected_calibration_error(probs, labels))
        logging.info("  ECE (after):                 %.4f", expected_calibration_error(calibrated, labels))

    # Refit on the whole reserved pool for the actual augmentation.
    full_confidences, full_correctness = _confidence_and_correctness(
        statistics_for(calibration_prompts)
    )
    if full_confidences:
        calibrator.fit(full_confidences, full_correctness)
        logging.info("Refit calibrator on %d calibration prompts.", len(calibration_prompts))

    return calibrator, training_generations
