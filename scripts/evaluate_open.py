"""Stage 4b (open ended): score free-text predictions with an LLM judge.

Equivalence to the reference answer cannot be checked by string match, so a
judge model decides it.  Requires ``OPENAI_API_KEY`` in the environment.

Example::

    OPENAI_API_KEY=... python scripts/evaluate_open.py runs/open/predictions.jsonl
"""

import argparse
import json

from tqdm import tqdm

import _bootstrap  # noqa: F401

from uncertainty_distillation.evaluation import report
from uncertainty_distillation.judge import EntailmentLLM
from uncertainty_distillation.parsing import confidence_to_bin


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="Predictions JSONL from predict.py.")
    parser.add_argument("--num_bins", type=int, default=5, choices=[2, 5, 7, 10])
    parser.add_argument("--judge_model", default="gpt-4o-mini")
    parser.add_argument("--predict_field", default="llm_completion")
    parser.add_argument("--label_field", default="label")
    parser.add_argument("--metrics_path", default=None, help="Optional JSON output.")
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.predictions) as predictions_file:
        predictions = [json.loads(line) for line in predictions_file]

    judge = EntailmentLLM(model=args.judge_model)

    bins, correctness, unjudged = [], [], 0
    for example in tqdm(predictions):
        completion = example[args.predict_field]
        verdict = judge.check_implication(
            example[args.label_field], completion, example=example
        )
        if verdict < 0:
            # The judge neither confirmed nor contradicted the answer.
            unjudged += 1
            continue
        bins.append(confidence_to_bin(completion, args.num_bins))
        correctness.append(verdict)

    print(f"# predictions the judge could not score: {unjudged}")
    metrics = report(bins, correctness, args.num_bins)
    metrics["num_unjudged"] = unjudged

    if args.metrics_path:
        with open(args.metrics_path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)


if __name__ == "__main__":
    main()
