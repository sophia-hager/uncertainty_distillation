"""Write a dataset to the ``{"prompt": ..., "label": ...}`` JSONL format.

The same format is used for the student's evaluation data and as the reference
for the teacher samples.

Example::

    python scripts/write_dataset.py --dataset arc --split validation \
        --output data/arc_val.jsonl
"""

import argparse
import json

import _bootstrap  # noqa: F401  (sets up sys.path)

from uncertainty_distillation.data import DATASETS, format_prompt, load_examples
from uncertainty_distillation.paths import DATA_DIR, ensure_parent


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default=None,
                        help="Output JSONL (default: data/<dataset>_<split>.jsonl)")
    parser.add_argument("--shuffle_seed", type=int, default=42,
                        help="Shuffle before slicing; pass -1 to keep dataset order.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Keep at most this many examples.")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip this many examples before taking --limit.")
    return parser.parse_args()


def main():
    args = get_args()
    output_path = args.output or DATA_DIR / f"{args.dataset}_{args.split}.jsonl"

    examples = load_examples(
        args.dataset,
        args.split,
        shuffle_seed=None if args.shuffle_seed < 0 else args.shuffle_seed,
        limit=args.limit,
        offset=args.offset,
    )
    print(f"Writing {len(examples)} examples to {output_path}")

    with open(ensure_parent(output_path), "w") as output_file:
        for example in examples:
            record = {"prompt": format_prompt(example), "label": example["answer"]}
            output_file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
