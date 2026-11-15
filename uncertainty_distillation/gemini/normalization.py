"""Answer normalization.

For multiple choice, two answers are the same when they are the same string
after lowercasing and stripping punctuation; this is what collapses "A", "a"
and "(A)" into one answer when counting the teacher's samples.
"""

import string


class StringNormalizer:
    """Normalizes an answer to the canonical form used as its identity."""

    def normalize(self, prediction: str) -> str:
        prediction = prediction.lower()
        prediction = prediction.translate(str.maketrans("", "", string.punctuation))
        return prediction.strip()

    def get_semantic_ids(self, predictions: list[str]) -> list[int]:
        """Assign each prediction the id of its normalized form."""
        normalized = [self.normalize(p) for p in predictions]
        unique = sorted(set(normalized))
        return [unique.index(p) for p in normalized]
