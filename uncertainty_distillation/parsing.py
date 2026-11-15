"""Parsing of model completions: tags, answers and verbalized confidences."""

import re

from .prompts import CONFIDENCE_BINS

#: Schemes that append the confidence after the answer.
APPEND_SCHEMES = ("append_all", "append_correct", "append_stratified")


def extract_text_between_tags(text: str, tag: str) -> list[str]:
    """Return every chunk of ``text`` enclosed in ``<tag> ... </tag>``.

    Matching is case-insensitive and spans newlines, since sampled reasoning
    chains are usually multi-line.
    """
    pattern = re.compile(rf"<{tag}\b[^>]*>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    return [match.strip() for match in pattern.findall(text)]


def extract_answer(completion: str) -> str:
    """Extract the normalized answer from a completion.

    For multiple choice we only keep the answer *letter*: the model is asked for
    ``<answer> C) 42 </answer>`` and everything after the parenthesis is
    discarded so that "C) 42" and "C" collapse to the same answer.
    """
    answer = completion.split("<answer>")[-1].split("</answer>")[0]
    return answer.split(")")[0].strip().lower()


def extract_reasoning(completion: str) -> str:
    """Extract the reasoning chain, tags included, or '' when there is none."""
    chains = extract_text_between_tags(completion, "reasoning")
    if not chains:
        return ""
    return f"<reasoning>{chains[-1]}</reasoning>"


def normalize_reference_answer(label: str) -> str:
    """Normalize a gold label so it can be compared against a parsed answer.

    Handles both plain labels ("C") and GSM8K-style references that put the
    final answer after ``####``.
    """
    return label.split("####")[-1].strip().lower()


def is_answer_correct(predicted: str, actual: str) -> bool:
    return predicted == actual


def confidence_to_bin(completion: str, num_bins: int = 5) -> int:
    """Map the verbalized confidence in ``completion`` to its bin index.

    Returns -1 when no parsable confidence is present.
    """
    vocabulary = {word: i for i, word in enumerate(CONFIDENCE_BINS[num_bins])}
    matches = extract_text_between_tags(completion, "confidence")
    if not matches:
        return -1
    return vocabulary.get(matches[-1].strip().lower(), -1)


def verbalize_confidence(prob_bin: int, num_bins: int = 5, scheme: str = "append_all") -> str:
    """Render bin index ``prob_bin`` as the text appended to (or prepended to)
    the answer during fine-tuning data construction."""
    if num_bins not in CONFIDENCE_BINS:
        raise ValueError(f"Unsupported number of bins: {num_bins}")
    word = CONFIDENCE_BINS[num_bins][prob_bin]

    if scheme in APPEND_SCHEMES:
        return f"<confidence>{word}</confidence>"
    if scheme == "prepend":
        if num_bins not in (2, 5):
            raise ValueError(f"Prepending is only defined for 2 or 5 bins, got {num_bins}")
        return f"With {word} confidence:"
    raise ValueError(f"Unknown scheme: {scheme}")


def count_confidences(examples: list[dict], scheme: str = "append_all") -> dict:
    """Count how often each verbalized confidence occurs in a built dataset."""
    if scheme not in APPEND_SCHEMES:
        raise ValueError(f"Unknown scheme: {scheme}")
    counts: dict[str, int] = {}
    for example in examples:
        matches = extract_text_between_tags(example["completion"], "confidence")
        if not matches:
            continue
        counts[matches[-1]] = counts.get(matches[-1], 0) + 1
    return counts
