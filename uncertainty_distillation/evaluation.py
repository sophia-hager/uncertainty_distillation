"""Scoring of student predictions: accuracy per verbalized confidence bin."""

import string

import numpy as np
from sklearn.metrics import roc_auc_score

#: Bin assigned to completions with no parsable confidence.
UNPARSABLE_BIN = -1


def normalize_for_match(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace, as HF exact_match
    does with ``ignore_case=True, ignore_punctuation=True``."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> int:
    return int(normalize_for_match(prediction) == normalize_for_match(reference))


def report(bins, correctness, num_bins: int) -> dict:
    """Print per-bin accuracy, overall accuracy and AUROC.

    A well-distilled model is both accurate and *ordered*: accuracy should rise
    monotonically with the verbalized confidence bin, which the AUROC of the bin
    index against correctness summarizes in a single number.
    """
    bins = np.asarray(bins)
    correctness = np.asarray(correctness)

    for bin_index in range(num_bins):
        in_bin = correctness[bins == bin_index]
        print(f"BIN {bin_index}")
        print("NUMBER IN BIN:", len(in_bin))
        if len(in_bin):
            print("accuracy:", in_bin.mean())

    unparsable = int((bins == UNPARSABLE_BIN).sum())
    print(f"# completions without a parsable confidence: {unparsable}")

    scored = bins != UNPARSABLE_BIN
    metrics = {
        "num_examples": int(len(bins)),
        "num_unparsable_confidence": unparsable,
        "accuracy": float(correctness.mean()) if len(correctness) else float("nan"),
    }
    print("overall accuracy:", metrics["accuracy"])

    try:
        metrics["auroc"] = float(roc_auc_score(correctness[scored], bins[scored]))
        print("AUROC:", metrics["auroc"])
    except ValueError:
        print("AUROC undefined: predictions are all one class")

    return metrics
