"""Stage 2 (multiple choice): turn teacher samples into fine-tuning data.

The teacher's relative frequency for an answer is recalibrated against
correctness (isotonic regression by default), discretized into ``--n_bins``
confidence bins, and verbalized.  Each kept answer becomes one training example
whose completion is ``<reasoning>...</reasoning><answer>...</answer>
<confidence>...</confidence>``.

Schemes:

* ``append_correct``    -- keep only the correct answer for each question;
* ``append_all``        -- keep the correct answer plus up to ``--num_incorrect``
  incorrect ones, so the student also learns to be unconfident when wrong;
* ``append_stratified`` -- keep at most one answer per confidence bin.

Example::

    python scripts/build_finetuning_data.py --scheme append_all --num_incorrect 1 \
        --predictions_path 'samples/siqa_Llama-3.2-3B-Instruct/*.jsonl' \
        --output_path datasets/llama_siqa.jsonl --calibration_method isotonic
"""

import argparse
import glob
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
from uncertainty_distillation.parsing import (
    count_confidences,
    extract_answer,
    is_answer_correct,
    normalize_reference_answer,
    verbalize_confidence,
)
from uncertainty_distillation.paths import ensure_parent


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_path", required=True,
                        help="Glob over the teacher sample shards.")
    parser.add_argument("--output_path", required=True, help="Output JSONL.")
    parser.add_argument("--calibration_method", choices=["isotonic", "identity"],
                        default="isotonic")
    parser.add_argument("--scheme", default="append_correct",
                        choices=["append_correct", "append_all", "append_stratified"])
    parser.add_argument("--n_bins", type=int, default=5, choices=[2, 5, 7, 10])
    parser.add_argument("--completion_type", choices=["full", "confidence_only"],
                        default="full",
                        help="'confidence_only' trains the student to emit a "
                             "confidence without restating the answer.")
    parser.add_argument("--num_incorrect", type=int, default=2,
                        help="Incorrect answers kept per question under append_all.")
    parser.add_argument("--add_reasoning", action="store_true", default=True)
    parser.add_argument("--no_reasoning", dest="add_reasoning", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_predictions(predictions_path):
    paths = sorted(glob.glob(predictions_path))
    if not paths:
        raise ValueError(f"No sample files matched: {predictions_path}")
    examples = []
    for path in paths:
        with open(path) as sample_file:
            examples.extend(json.loads(line) for line in sample_file)
    print(f"Loaded {len(examples)} examples from {len(paths)} shards")
    return examples


def collapse_counts(example):
    """Group the sampled answers of one example, keyed by normalized answer."""
    collapsed = {}
    for answer, count in example["counts"].items():
        try:
            normalized = extract_answer(answer)
        except Exception:  # Unparsable samples are kept as-is.
            normalized = answer
        collapsed[normalized] = collapsed.get(normalized, 0) + count
    assert collapsed and sum(collapsed.values()) > 0
    return collapsed


def calibration_targets(examples):
    """(label, teacher confidence) pairs over every answer of every example."""
    labels, probs = [], []
    for example in examples:
        gold_answer = normalize_reference_answer(example["label"])
        collapsed = collapse_counts(example)
        total = sum(collapsed.values())
        for answer, count in collapsed.items():
            labels.append(int(is_answer_correct(answer, gold_answer)))
            probs.append(count / total)
    return labels, probs


def choose_reasoning(reasoning_chains, rng):
    """Pick one non-empty reasoning chain per answer, if any exists."""
    chosen = {}
    for answer, chains in reasoning_chains.items():
        candidates = sorted({chain for chain in chains if chain})
        chosen[answer] = rng.choice(candidates) if candidates else ""
    return chosen


def incorrect_answers_to_keep(gold_answer, answers, reasoning, num_to_keep):
    """Incorrect answers that have a usable reasoning chain."""
    kept = [
        answer
        for answer in answers
        if not is_answer_correct(answer, gold_answer) and reasoning.get(answer)
    ]
    return kept[:num_to_keep] if num_to_keep else kept


def create_answers_with_uncertainty(examples, calibrator, binner, args, rng):
    new_examples = []
    all_probs, all_bins = [], []

    for example in tqdm(examples):
        gold_answer = normalize_reference_answer(example["label"])
        collapsed = collapse_counts(example)
        total = sum(collapsed.values())
        reasoning = choose_reasoning(example.get("reasoning", {}), rng)
        incorrect_to_add = incorrect_answers_to_keep(
            gold_answer, collapsed.keys(), reasoning, args.num_incorrect
        )

        filled_bins = set()
        for answer in sorted(collapsed):
            probability = calibrate(calibrator, collapsed[answer] / total)
            prob_bin = to_bin(binner, probability)
            all_probs.append(probability)
            all_bins.append(prob_bin)

            correct = is_answer_correct(answer, gold_answer)
            if args.scheme == "append_correct" and not correct:
                continue
            if args.scheme == "append_all" and not (correct or answer in incorrect_to_add):
                continue
            if args.scheme == "append_stratified":
                if prob_bin in filled_bins:
                    continue
                filled_bins.add(prob_bin)

            maybe_answer = f"<answer>{answer}</answer>" if args.completion_type == "full" else ""
            confidence = verbalize_confidence(prob_bin, args.n_bins, args.scheme)
            prefix = reasoning.get(answer, "") if args.add_reasoning else ""
            new_examples.append(
                {"prompt": example["prompt"], "completion": f"{prefix}{maybe_answer}{confidence}"}
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
