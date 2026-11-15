"""Stage 4b (multiple choice): score predictions and their verbalized confidence.

Reports exact-match accuracy overall and within each confidence bin, plus the
AUROC of the verbalized confidence against correctness.

Example::

    python scripts/evaluate.py runs/llama_siqa/predictions.jsonl --num_bins 5
"""

import argparse
import json

from tqdm import tqdm

import _bootstrap  # noqa: F401

from uncertainty_distillation.evaluation import UNPARSABLE_BIN, exact_match, report
from uncertainty_distillation.parsing import (
    confidence_to_bin,
    extract_answer,
    normalize_reference_answer,
)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="Predictions JSONL from predict.py.")
    parser.add_argument("--num_bins", type=int, default=5, choices=[2, 5, 7, 10])
    parser.add_argument("--predict_field", default="llm_completion")
    parser.add_argument("--label_field", default="label")
    parser.add_argument("--metrics_path", default=None, help="Optional JSON output.")
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.predictions) as predictions_file:
        predictions = [json.loads(line) for line in predictions_file]

    bins, correctness, unparsable_answers = [], [], 0
    for example in tqdm(predictions):
        completion = example[args.predict_field]
        bins.append(confidence_to_bin(completion, args.num_bins))

        label = normalize_reference_answer(example[args.label_field])
        predicted = extract_answer(completion) if "<answer>" in completion else ""
        if not predicted:
            unparsable_answers += 1
        correctness.append(exact_match(predicted, label))

    print(f"# unparsable answers: {unparsable_answers}")
    metrics = report(bins, correctness, args.num_bins)
    metrics["num_unparsable_answers"] = unparsable_answers

    if args.metrics_path:
        with open(args.metrics_path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)


if __name__ == "__main__":
    main()
