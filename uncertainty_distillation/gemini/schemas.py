"""Structured-output schemas for the model's JSON responses.

Gemini is asked for structured output, so answers and confidences come back as
JSON rather than as tagged text.  The confidence field is a closed enum, which
is what makes the verbalized confidence parseable without heuristics.
"""

from typing import Literal, Type

from pydantic import BaseModel, Field

Answer = Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

Confidence5 = Literal["VERY LOW", "LOW", "MEDIUM", "HIGH", "VERY HIGH"]

Confidence7 = Literal[
    "VERY LOW", "LOW", "SOMEWHAT LOW", "MEDIUM", "SOMEWHAT HIGH", "HIGH", "VERY HIGH"
]

#: Bin labels in ascending order of confidence, by confidence level.
CONFIDENCE_BINS = {
    "FIVE_BINS": list(Confidence5.__args__),
    "SEVEN_BINS": list(Confidence7.__args__),
}


class MMLUResponse5(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning to arrive at the correct answer.")
    answer: Answer = Field(description="The single correct answer option.")
    confidence: Confidence5 = Field(description="The confidence level in the answer.")


class MMLUResponse5NoReasoning(BaseModel):
    answer: Answer = Field(description="The single correct answer option.")
    confidence: Confidence5 = Field(description="The confidence level in the answer.")


class MMLUResponse7(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning to arrive at the correct answer.")
    answer: Answer = Field(description="The single correct answer option.")
    confidence: Confidence7 = Field(description="The confidence level in the answer.")


class MMLUResponse7NoReasoning(BaseModel):
    answer: Answer = Field(description="The single correct answer option.")
    confidence: Confidence7 = Field(description="The confidence level in the answer.")


class MMLUResponseNoConfidence(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning to arrive at the correct answer.")
    answer: Answer = Field(description="The single correct answer option.")


class MMLUResponseNoConfidenceNoReasoning(BaseModel):
    answer: Answer = Field(description="The single correct answer option.")


class BinnedAnswerResponse5(BaseModel):
    VERY_LOW: Answer | None = Field(default=None, description="The answer associated with VERY LOW confidence.")
    LOW: Answer | None = Field(default=None, description="The answer associated with LOW confidence.")
    MEDIUM: Answer | None = Field(default=None, description="The answer associated with MEDIUM confidence.")
    HIGH: Answer | None = Field(default=None, description="The answer associated with HIGH confidence.")
    VERY_HIGH: Answer | None = Field(default=None, description="The answer associated with VERY HIGH confidence.")


class BinnedAnswerResponse7(BaseModel):
    VERY_LOW: Answer | None = Field(default=None, description="The answer associated with VERY LOW confidence.")
    LOW: Answer | None = Field(default=None, description="The answer associated with LOW confidence.")
    SOMEWHAT_LOW: Answer | None = Field(default=None, description="The answer associated with SOMEWHAT LOW confidence.")
    MEDIUM: Answer | None = Field(default=None, description="The answer associated with MEDIUM confidence.")
    SOMEWHAT_HIGH: Answer | None = Field(default=None, description="The answer associated with SOMEWHAT HIGH confidence.")
    HIGH: Answer | None = Field(default=None, description="The answer associated with HIGH confidence.")
    VERY_HIGH: Answer | None = Field(default=None, description="The answer associated with VERY HIGH confidence.")


def get_confidence_bins(confidence_level: str) -> list[str]:
    """Bin labels, lowest confidence first."""
    level = getattr(confidence_level, "value", confidence_level)
    if level not in CONFIDENCE_BINS:
        raise ValueError(f"Invalid confidence level: {confidence_level}")
    return list(CONFIDENCE_BINS[level])


def get_mmlu_response_model(
    confidence_level: str,
    include_reasoning: bool = True,
    binned_answers: bool = False,
    include_confidence: bool = True,
) -> Type[BaseModel]:
    """The response schema matching a given prompt variant."""
    level = getattr(confidence_level, "value", confidence_level)
    if level not in CONFIDENCE_BINS:
        raise ValueError(f"Invalid confidence level: {confidence_level}")

    if not include_confidence:
        return MMLUResponseNoConfidence if include_reasoning else MMLUResponseNoConfidenceNoReasoning

    if binned_answers:
        return BinnedAnswerResponse5 if level == "FIVE_BINS" else BinnedAnswerResponse7

    if level == "FIVE_BINS":
        return MMLUResponse5 if include_reasoning else MMLUResponse5NoReasoning
    return MMLUResponse7 if include_reasoning else MMLUResponse7NoReasoning
