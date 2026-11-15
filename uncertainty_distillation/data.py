"""Dataset adapters.

Every dataset is normalized to the same record shape so the rest of the
pipeline is dataset-agnostic:

* multiple choice -- ``{"question": str, "choices": list[str], "answer": str}``
  where ``answer`` is the letter of the correct choice;
* open ended      -- ``{"question": str, "answer": str}`` where ``answer`` is
  the reference answer text.
"""

from datasets import load_dataset

ANSWER_LETTERS = "ABCDEFGHIJ"


def _mmlu(split: str):
    dataset = load_dataset("cais/mmlu", "all")[split]
    for example in dataset:
        yield {
            "question": example["question"],
            "choices": list(example["choices"]),
            "answer": ANSWER_LETTERS[example["answer"]],
        }


def _social_i_qa(split: str):
    dataset = load_dataset("allenai/social_i_qa", trust_remote_code=True)[split]
    for example in dataset:
        yield {
            "question": f"{example['context']} {example['question']}",
            "choices": [example["answerA"], example["answerB"], example["answerC"]],
            "answer": ANSWER_LETTERS[int(example["label"]) - 1],
        }


def _arc(split: str):
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")[split]
    for example in dataset:
        yield {
            "question": example["question"],
            "choices": list(example["choices"]["text"]),
            "answer": example["answerKey"],
            "labels": list(example["choices"]["label"]),
        }


def _gsm8k(split: str):
    dataset = load_dataset("openai/gsm8k", "main")[split]
    for example in dataset:
        yield {
            "question": example["question"],
            "answer": example["answer"].split("####")[-1].strip(),
            "solution": example["answer"],
        }


MULTIPLE_CHOICE_DATASETS = {
    "mmlu": _mmlu,
    "siqa": _social_i_qa,
    "arc": _arc,
}

OPEN_DATASETS = {
    "gsm8k": _gsm8k,
}

DATASETS = {**MULTIPLE_CHOICE_DATASETS, **OPEN_DATASETS}


def task_for_dataset(name: str) -> str:
    if name in MULTIPLE_CHOICE_DATASETS:
        return "multiple_choice"
    if name in OPEN_DATASETS:
        return "open"
    raise ValueError(f"Unknown dataset: {name}")


def load_examples(name: str, split: str, shuffle_seed=None, limit=None, offset=0):
    """Load a dataset, optionally shuffled and restricted to a contiguous slice."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name} (known: {sorted(DATASETS)})")
    examples = list(DATASETS[name](split))
    if shuffle_seed is not None:
        import random

        random.Random(shuffle_seed).shuffle(examples)
    examples = examples[offset:]
    if limit is not None:
        examples = examples[:limit]
    return examples


def format_choices(example: dict) -> str:
    """Render the choices of a multiple-choice example as ``A) ...`` lines."""
    labels = example.get("labels") or ANSWER_LETTERS[: len(example["choices"])]
    return "".join(f"\n{label}) {text}" for label, text in zip(labels, example["choices"]))


def format_prompt(example: dict) -> str:
    """The user turn shown to the model for either kind of example."""
    if "choices" in example:
        return f"{example['question']}{format_choices(example)}\n"
    return example["question"]


def shard_indices(n_total: int, num_shards: int):
    """Split ``range(n_total)`` into ``num_shards`` contiguous (start, end) pairs."""
    size = n_total // num_shards
    shards = []
    for index, start in enumerate(range(0, n_total, size), start=1):
        if index == num_shards:
            shards.append((start, n_total))
            break
        shards.append((start, start + size))
    return shards
