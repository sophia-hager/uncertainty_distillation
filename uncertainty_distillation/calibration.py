"""Calibration of the teacher's predictive distribution and binning of it.

The teacher's confidence in an answer is the relative frequency of that answer
among the samples drawn for its question.  Those relative frequencies are
(optionally) recalibrated against correctness with isotonic regression, and then
discretized into the small number of bins the student verbalizes.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import KBinsDiscretizer

from .general_calibration_error import compute_all_metrics


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """A no-op calibrator: the relative frequency is used as the probability."""

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return np.asarray(input_array, dtype=float)

    def predict(self, input_array, y=None):
        return self.transform(input_array)


def fit_calibrator(probs, labels, method: str = "isotonic"):
    """Fit a map from teacher confidence to probability of being correct."""
    if method == "isotonic":
        print(f"Fitting isotonic regression to {len(probs)} examples")
        return IsotonicRegression(out_of_bounds="clip").fit(probs, labels)
    if method == "identity":
        return IdentityTransformer()
    raise ValueError(f"Unknown calibration method: {method}")


def fit_binner(n_bins: int = 5, strategy: str = "uniform") -> KBinsDiscretizer:
    """Fit the discretizer that turns probabilities into confidence bins.

    The binner is fit on the endpoints of the unit interval rather than on the
    observed probabilities, so the bins have a fixed, interpretable meaning that
    does not shift between datasets or models.
    """
    binner = KBinsDiscretizer(encode="ordinal", strategy=strategy, n_bins=n_bins)
    binner.fit(np.array([[0.0], [1.0]]))
    print("Bin edges:", binner.bin_edges_)
    return binner


def calibrate(calibrator, probability: float) -> float:
    return float(np.asarray(calibrator.predict(np.array([probability]))).item())


def to_bin(binner: KBinsDiscretizer, probability: float) -> int:
    return int(binner.transform(np.array([[probability]])).item())


def report_calibration(labels, probs, calibrator) -> dict:
    """Print every GCE variant before and after calibration; return the raw ones."""
    measures = compute_all_metrics(labels, probs)
    post_hoc = compute_all_metrics(labels, calibrator.predict(np.asarray(probs)))
    print(f"{'parameters':<60} {'pre-hoc':>10} {'post-hoc':>10}")
    for key, value in measures.items():
        print(f"{key:<60} {value:>10.4f} {post_hoc[key]:>10.4f}")
    return measures


def fit_calibration(labels, probs, calibration_method="isotonic", n_bins=5,
                    binner_strategy="uniform"):
    """Fit and report the calibrator and the binner for a set of samples."""
    calibrator = fit_calibrator(probs, labels, method=calibration_method)
    report_calibration(labels, probs, calibrator)
    return calibrator, fit_binner(n_bins=n_bins, strategy=binner_strategy)


def print_density_histogram(data, num_bins: int = 7, bar_char: str = "#") -> None:
    """Print an ASCII density histogram, used to sanity-check bin occupancy."""
    data = np.asarray(data, dtype=float).reshape(-1)
    if data.size == 0:
        return
    counts, bin_edges = np.histogram(data, bins=num_bins)
    densities = counts / data.size
    max_density = densities.max() if densities.size else 1
    width = 20

    print("-" * (width + 10))
    for i in range(num_bins):
        bar_length = int(densities[i] / max_density * width) if max_density > 0 else 0
        print(
            f"[{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}): "
            f"{bar_char * bar_length} ({densities[i]:.3f})"
        )
    print("-" * (width + 10))
