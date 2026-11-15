"""Staging Hugging Face datasets into GCS in the pipeline's own format.

Every example becomes ``{"inputs": question, "targets": answer, "candidates":
{letter: option}}``.  Vertex AI reads its data from GCS, so the staged file is
also what the batch jobs consume.
"""

import json
import logging
import random
from typing import Optional

from datasets import load_dataset

from .storage import write_jsonl

#: Fields checked, in order, when looking for the question text.
INPUT_FIELDS = ("question", "article", "input", "text", "document", "content")

#: Fields checked, in order, when no explicit target field is configured.
TARGET_FIELDS = ("answer", "solution", "highlights")

#: Seed for the deterministic validation/test halving of the evaluation split.
SPLIT_SEED = 42


def format_example(example: dict, candidates_field: Optional[str], target_field: Optional[str]) -> dict:
    """Normalize one Hugging Face example."""
    inputs = ""
    for field in INPUT_FIELDS:
        if example.get(field):
            inputs = example[field]
            break

    item = {
        "inputs": inputs,
        "original_data": {k: str(v) for k, v in example.items() if k != "inputs"},
    }

    if candidates_field and candidates_field in example:
        options = example[candidates_field]
        item["targets"] = options[example["answer_index"]]
        item["candidates"] = {chr(ord("A") + i): option for i, option in enumerate(options)}
        return item

    if target_field and target_field in example:
        item["targets"] = example[target_field]
    else:
        item["targets"] = next((example[f] for f in TARGET_FIELDS if f in example), "")
    return item


class HuggingFaceDatasetProvider:
    """Loads a dataset (or a local JSONL file) and stages it in GCS."""

    def __init__(self, gcs_client, gcs_bucket_name: str):
        self.gcs_client = gcs_client
        self.gcs_bucket_name = gcs_bucket_name

    def get_dataset(
        self,
        dataset_name: str,
        split: str,
        full: bool = False,
        dev_set_size: int = 10,
        candidates_field: Optional[str] = None,
        target_field: Optional[str] = None,
        halve: bool = False,
        test_split: bool = False,
    ) -> tuple[str, int, int]:
        """Stage a dataset and return (GCS URI, #examples, #characters).

        With ``halve``, the split is shuffled deterministically and cut in two:
        the first half is the validation set and the second the test set, so
        that reporting on test never touches data used while developing.
        """
        if dataset_name.endswith(".jsonl"):
            with open(dataset_name) as dataset_file:
                formatted = [json.loads(line) for line in dataset_file]
        else:
            formatted = [
                format_example(example, candidates_field, target_field)
                for example in load_dataset(dataset_name, split=split)
            ]

        if halve:
            random.Random(SPLIT_SEED).shuffle(formatted)
            middle = len(formatted) // 2
            if test_split:
                logging.info("Using TEST split (second half of the evaluation data).")
                formatted = formatted[middle:]
            else:
                logging.info("Using VALIDATION split (first half of the evaluation data).")
                formatted = formatted[:middle]

        if not full:
            formatted = formatted[:dev_set_size]

        num_characters = sum(
            len(str(item.get("inputs", ""))) + len(str(item.get("targets", "")))
            for item in formatted
        )

        suffix = ("-test" if test_split else "-val") if halve else ""
        suffix += "" if full else "-dev"
        blob_name = f"{dataset_name.replace('/', '_')}-{split}{suffix}.jsonl"
        gcs_uri = write_jsonl(
            self.gcs_client, f"gs://{self.gcs_bucket_name}/{blob_name}", formatted
        )
        return gcs_uri, len(formatted), num_characters
