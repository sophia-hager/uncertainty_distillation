"""LLM-as-a-judge scoring for open-ended answers.

Used only at evaluation time, to decide whether a generated free-text answer is
equivalent to the reference.  The API key is read from the ``OPENAI_API_KEY``
environment variable and is never stored in this repository.
"""

import hashlib
import logging
import os


def md5hash(string: str) -> int:
    return int(hashlib.md5(string.encode("utf-8")).hexdigest(), 16)


class EntailmentLLM:
    """Judge two answers for equivalence with a chat model, with a local cache."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.02):
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set the OPENAI_API_KEY environment variable to use the LLM judge."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.prediction_cache: dict[int, str] = {}

    def equivalence_prompt(self, text1: str, text2: str, question: str) -> str:
        return (
            f'We are evaluating answers to the question "{question}"\n'
            "Here are two possible answers:\n"
            f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
            "Is Possible Answer 1 equivalent to Possible Answer 2, or do the answers "
            "contradict? Respond only with 'equivalent' or 'contradictory'."
        )

    def predict(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def check_implication(self, reference: str, prediction: str, example: dict) -> int:
        """Return 1 if equivalent, 0 if contradictory, -1 if unparsable."""
        if example is None:
            raise ValueError("The judge needs the original example for context.")
        prompt = self.equivalence_prompt(reference, prediction, example["prompt"])

        hashed = md5hash(prompt)
        if hashed not in self.prediction_cache:
            self.prediction_cache[hashed] = self.predict(prompt)
        response = self.prediction_cache[hashed].lower()[:30]

        if "equivalent" in response:
            return 1
        if "contradictory" in response:
            return 0
        logging.warning("Unparsable judge response: %s", response)
        return -1
