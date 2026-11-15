"""Stage 2 (open ended): turn clustered teacher samples into fine-tuning data.

Identical in spirit to ``build_finetuning_data.py``, but the unit of confidence
is a semantic cluster rather than an answer letter: the fraction of samples in a
cluster is calibrated, binned, verbalized and appended to a representative
answer from that cluster.

Example::

    python scripts/build_finetuning_data_open.py --scheme append_all --num_incorrect 1 \
        --predictions_path 'samples/open_gsm8k_Ministral-8B-Instruct-2410/*.jsonl' \
        --output_path datasets/ministral_open.jsonl --calibration_method isotonic
"""

import argparse
import json
import random

from tqdm import tqdm

import _bootstrap  # noqa: F401

from uncertainty_distillation.calibration import (
    calibrate,
    fit_calibration,
    print_density_histogram,
    to_bin,
)
from uncertainty_distillation.parsing import count_confidences, verbalize_confidence
from uncertainty_distillation.paths import ensure_parent
from build_finetuning_data import load_predictions

#: Semantic cluster id assigned to answers equivalent to the reference.
CORRECT_CLUSTER = "0"


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--calibration_method", choices=["isotonic", "identity"],
                        default="isotonic")
    parser.add_argument("--scheme", default="append_all", choices=["append_all"])
    parser.add_argument("--n_bins", type=int, default=5, choices=[2, 5, 7, 10])
    parser.add_argument("--num_incorrect", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def cluster_frequencies(example):
    """Map each cluster's representative answer to (is_correct, frequency)."""
    total = sum(count for _, count in example["answers"].values())
    return {
        text: (cluster_id == CORRECT_CLUSTER, count / total)
        for cluster_id, (text, count) in example["answers"].items()
    }


def calibration_targets(examples):
    labels, probs = [], []
    for example in examples:
        for correct, frequency in cluster_frequencies(example).values():
            labels.append(int(correct))
            probs.append(frequency)
    return labels, probs


def create_answers_with_uncertainty(examples, calibrator, binner, args, rng):
    new_examples = []
    all_probs, all_bins = [], []

    for example in tqdm(examples):
        clusters = cluster_frequencies(example)
        correct = {text: freq for text, (is_correct, freq) in clusters.items() if is_correct}
        incorrect = {text: freq for text, (is_correct, freq) in clusters.items() if not is_correct}

        keep = dict(correct)
        for text in rng.sample(sorted(incorrect), k=min(args.num_incorrect, len(incorrect))):
            keep[text] = incorrect[text]

        for answer in sorted(keep):
            probability = calibrate(calibrator, keep[answer])
            prob_bin = to_bin(binner, probability)
            all_probs.append(probability)
            all_bins.append(prob_bin)
            confidence = verbalize_confidence(prob_bin, args.n_bins, args.scheme)
            new_examples.append(
                {"prompt": example["prompt"], "completion": f"{answer}{confidence}"}
            )

    print("Calibrated teacher probabilities:")
    print_density_histogram(all_probs)
    print("Confidence bins:")
    print_density_histogram(all_bins)
    return new_examples


def main():
    args = get_args()
    rng = random.Random(args.seed)

    print("Loading predictions...")
    examples = load_predictions(args.predictions_path)

    print("Fitting calibration...")
    labels, probs = calibration_targets(examples)
    calibrator, binner = fit_calibration(
        labels, probs, calibration_method=args.calibration_method, n_bins=args.n_bins
    )

    print("Creating new dataset...")
    dataset = create_answers_with_uncertainty(examples, calibrator, binner, args, rng)
    print(f"New dataset has {len(dataset)} examples")
    if dataset:
        print("Example:", dataset[0])

    print(f"Writing dataset to {args.output_path}")
    with open(ensure_parent(args.output_path), "w") as output_file:
        for example in dataset:
            output_file.write(json.dumps(example) + "\n")
    print(count_confidences(dataset, args.scheme))


if __name__ == "__main__":
    main()
