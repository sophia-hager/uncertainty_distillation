"""Stage 1 (open ended): sample the teacher and cluster answers semantically.

Free-text answers cannot be counted by string equality, so samples are grouped
into semantic equivalence classes with an NLI model.  The class that entails the
reference answer gets the id ``0``; the size of each class relative to the total
is the teacher's confidence in that answer.

Example::

    python scripts/sample_teacher_open.py --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset gsm8k --limit 7000 --num_shards 14 --shard_index 1
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import _bootstrap  # noqa: F401

from uncertainty_distillation.data import OPEN_DATASETS, load_examples, shard_indices
from uncertainty_distillation.paths import SAMPLES_DIR
from uncertainty_distillation.prompts import sampling_prompt
from uncertainty_distillation.semantic_clustering import (
    DEFAULT_ENTAILMENT_MODEL,
    EntailmentDeberta,
    get_semantic_ids,
)
from sample_teacher import DTYPES, sample_completions


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=sorted(OPEN_DATASETS), default="gsm8k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default=None,
                        help="Default: samples/open_<dataset>_<model name>")
    parser.add_argument("--entailment_model", default=DEFAULT_ENTAILMENT_MODEL)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=800)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=1, help="One-based shard index.")
    parser.add_argument("--shuffle_seed", type=int, default=-1,
                        help="Open-ended splits are taken in dataset order by default.")
    parser.add_argument("--limit", type=int, default=7000,
                        help="Examples reserved for teacher sampling; the rest of the "
                             "split can be held out for evaluation.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    output_dir = (
        args.output_dir or SAMPLES_DIR / f"open_{args.dataset}_{args.model.split('/')[-1]}"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"sample_{args.shard_index}.jsonl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPES[args.dtype]
    ).to(device)
    model.eval()
    entailment_model = EntailmentDeberta(args.entailment_model, device=device)

    examples = load_examples(
        args.dataset,
        args.split,
        shuffle_seed=None if args.shuffle_seed < 0 else args.shuffle_seed,
        limit=args.limit,
        offset=args.offset,
    )
    if args.num_shards > 1:
        start, end = shard_indices(len(examples), args.num_shards)[args.shard_index - 1]
        print(f"Selecting examples in range: ({start}, {end})")
        examples = examples[start:end]

    system_message = sampling_prompt("open")
    print(f"Sampling {args.num_samples} completions for {len(examples)} examples")

    with open(output_path, "w") as output_file:
        for example in tqdm(examples):
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": example["question"]},
            ]
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

            completions = sample_completions(model, tokenizer, input_ids, args, device)
            assert len(completions) == args.num_samples

            reference = f"The answer is {example['answer']}"
            semantic_ids = get_semantic_ids(completions, entailment_model, label=reference)

            # One representative completion and a count per equivalence class.
            answers = {}
            for completion, semantic_id in zip(completions, semantic_ids):
                answers.setdefault(str(semantic_id), [completion, 0])
                answers[str(semantic_id)][1] += 1

            record = {
                "prompt": example["question"],
                "label": example["answer"],
                "answers": answers,
            }
            output_file.write(json.dumps(record) + "\n")

    print(f"Wrote samples to: {output_path}")


if __name__ == "__main__":
    main()
