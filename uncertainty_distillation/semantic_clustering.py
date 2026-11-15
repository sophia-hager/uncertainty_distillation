"""Semantic clustering of free-text answers.

For open-ended questions the teacher's predictive distribution cannot be read
off string equality: "42 eggs" and "She has 42 eggs." are the same answer.  We
group samples into semantic equivalence classes with a bidirectional NLI check,
and give the class that is equivalent to the reference answer the id 0 so that
downstream code can tell correct from incorrect clusters.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_ENTAILMENT_MODEL = "microsoft/deberta-v2-xlarge-mnli"

#: Index of the ``contradiction`` class in the DeBERTa MNLI head.
CONTRADICTION = 0


class EntailmentDeberta:
    """Bidirectional NLI judge used to decide whether two answers agree."""

    def __init__(self, model_name_or_path: str = DEFAULT_ENTAILMENT_MODEL, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        ).to(self.device)

    def check_implication(self, text1: str, text2: str, **kwargs) -> int:
        """Return the MNLI class for ``text1 -> text2`` (0 = contradiction)."""
        del kwargs  # Accepted for interface compatibility.
        inputs = self.tokenizer(
            text1, text2, return_tensors="pt", padding="longest", truncation=True
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return int(torch.argmax(F.softmax(logits, dim=1)).item())


def are_equivalent(model, text1: str, text2: str) -> bool:
    """Two answers agree when neither direction is a contradiction."""
    implications = [
        model.check_implication(text1, text2),
        model.check_implication(text2, text1),
    ]
    return CONTRADICTION not in implications


def get_semantic_ids(predictions: list[str], model, label: str = "") -> list[int]:
    """Cluster ``predictions``; the cluster matching ``label`` gets the id 0."""
    semantic_ids = [-1] * len(predictions)

    for i, prediction in enumerate(predictions):
        if are_equivalent(model, label, prediction):
            semantic_ids[i] = 0

    next_id = 1
    for i, prediction in enumerate(predictions):
        if semantic_ids[i] != -1:
            continue
        semantic_ids[i] = next_id
        for j in range(i + 1, len(predictions)):
            if semantic_ids[j] == -1 and are_equivalent(model, prediction, predictions[j]):
                semantic_ids[j] = next_id
        next_id += 1

    assert -1 not in semantic_ids
    return semantic_ids
