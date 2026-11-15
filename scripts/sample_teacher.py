"""Stage 1 (multiple choice): sample the teacher's predictive distribution.

For every question we draw ``--num_samples`` completions from the teacher with
ancestral sampling and record how often each answer letter occurs, along with a
reasoning chain for each distinct answer.  The relative frequencies are the raw
(uncalibrated) teacher confidences consumed by ``build_finetuning_data.py``.

Sampling is embarrassingly parallel, so the dataset can be split into shards
that are run independently (see ``run/sample_teacher.sh``).

Example::

    python scripts/sample_teacher.py --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset siqa --num_shards 32 --shard_index 1
"""

import argparse
import json
import os
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import _bootstrap  # noqa: F401

from uncertainty_distillation.data import (
    MULTIPLE_CHOICE_DATASETS,
    format_prompt,
    load_examples,
    shard_indices,
)
from uncertainty_distillation.parsing import extract_answer, extract_reasoning
from uncertainty_distillation.paths import SAMPLES_DIR
from uncertainty_distillation.prompts import MC_EXAMPLE_REASONING, sampling_prompt

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Teacher model, e.g. meta-llama/Llama-3.2-3B-Instruct "
                             "or mistralai/Ministral-8B-Instruct-2410")
    parser.add_argument("--dataset", choices=sorted(MULTIPLE_CHOICE_DATASETS), default="siqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default=None,
                        help="Default: samples/<dataset>_<model name>")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Samples generated per forward pass.")
    parser.add_argument("--max_new_tokens", type=int, default=800)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=1, help="One-based shard index.")
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_completions(model, tokenizer, input_ids, args, device):
    """Draw ``args.num_samples`` completions, in batches, and decode them."""
    completions = []
    remaining = args.num_samples
    while remaining > 0:
        size = min(args.batch_size, remaining)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids.to(device),
                max_new_tokens=args.max_new_tokens,
                num_beams=1,      # disable beam search
                do_sample=True,   # categorical sampling
                num_return_sequences=size,
            )
        # Keep only the generated continuation, not the prompt.
        completions.extend(
            tokenizer.batch_decode(outputs[:, input_ids.shape[-1]:], skip_special_tokens=True)
        )
        remaining -= size
    return completions


def summarize(completions):
    """Count answers and collect the reasoning chains that produced each one."""
    counts = Counter()
    reasoning = {}
    for completion in completions:
        answer = extract_answer(completion)
        counts[answer] += 1
        chain = extract_reasoning(completion)
        # Reasoning copied verbatim from the one-shot example is uninformative.
        if MC_EXAMPLE_REASONING in chain:
            chain = ""
        reasoning.setdefault(answer, []).append(chain)
    return dict(counts), reasoning


def main():
    args = get_args()
    set_seed(args.seed)

    output_dir = args.output_dir or SAMPLES_DIR / f"{args.dataset}_{args.model.split('/')[-1]}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"sample_{args.shard_index}.jsonl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPES[args.dtype]
    ).to(device)
    model.eval()

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

    system_message = sampling_prompt("multiple_choice")
    print(f"Sampling {args.num_samples} completions for {len(examples)} examples")

    with open(output_path, "w") as output_file:
        for example in tqdm(examples):
            prompt = format_prompt(example)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Answer this question: {prompt}"},
            ]
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

            completions = sample_completions(model, tokenizer, input_ids, args, device)
            assert len(completions) == args.num_samples
            counts, reasoning = summarize(completions)

            record = {
                "prompt": prompt,
                "label": example["answer"],
                "counts": counts,
                "reasoning": reasoning,
            }
            output_file.write(json.dumps(record) + "\n")

    print(f"Wrote samples to: {output_path}")


if __name__ == "__main__":
    main()
