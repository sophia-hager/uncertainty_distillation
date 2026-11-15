"""Prompts for multiple-choice uncertainty distillation with structured output."""

FORMAT_INSTRUCTIONS = {
    "STANDARD_REASONING": "Answer the following multiple choice question. Return your answer in JSON format with the fields 'reasoning', 'answer', and 'confidence'.",
    "STANDARD_NO_REASONING": "Answer the following multiple choice question. Return your answer in JSON format with the fields 'answer' and 'confidence'.",
    "STANDARD_REASONING_NO_CONFIDENCE": "Answer the following multiple choice question. Return your answer in JSON format with the fields 'reasoning' and 'answer'.",
    "STANDARD_NO_REASONING_NO_CONFIDENCE": "Answer the following multiple choice question. Return your answer in JSON format with the field 'answer'.",
    "BINNED_5": "Answer the following multiple choice question. Output a JSON object where keys are confidence levels ('VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH') and the values are the predictions with that confidence level. Do not repeat the same answer in different bins.",
    "BINNED_7": "Answer the following multiple choice question. Output a JSON object where keys are confidence levels ('VERY_LOW', 'LOW', 'SOMEWHAT_LOW', 'MEDIUM', 'SOMEWHAT_HIGH', 'HIGH', 'VERY_HIGH') and the values are the predictions with that confidence level. Do not repeat the same answer in different bins.",
    "SPARSE_BINNED_5": "Answer the following multiple choice question. Output a JSON object where keys are confidence levels ('VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH') and the values are the predictions with that confidence level. Exclude confidence levels with no corresponding predictions. Do not repeat the same answer in different bins.",
    "SPARSE_BINNED_7": "Answer the following multiple choice question. Output a JSON object where keys are confidence levels ('VERY_LOW', 'LOW', 'SOMEWHAT_LOW', 'MEDIUM', 'SOMEWHAT_HIGH', 'HIGH', 'VERY_HIGH') and the values are the predictions with that confidence level. Exclude confidence levels with no corresponding predictions. Do not repeat the same answer in different bins.",
}

ADDITIONAL_PROMPTS = {
    "PROMPT_DEFAULT": "",
    "PROMPT_COT": "Let's think step by step.",
}


def _format_instruction(config, binned_answers: bool, include_confidence: bool) -> str:
    five_bins = str(getattr(config.confidence_level, "value", config.confidence_level)) == "FIVE_BINS"

    if binned_answers:
        sparse = config.augmentation_scheme == "SPARSE_BINNED_ANSWERS"
        prefix = "SPARSE_BINNED" if sparse else "BINNED"
        return FORMAT_INSTRUCTIONS[f"{prefix}_{5 if five_bins else 7}"]

    reasoning = "REASONING" if config.include_reasoning else "NO_REASONING"
    suffix = "" if include_confidence else "_NO_CONFIDENCE"
    return FORMAT_INSTRUCTIONS[f"STANDARD_{reasoning}{suffix}"]


def get_prompt_with_instructions(
    example: dict,
    config,
    binned_answers: bool = False,
    include_confidence: bool = True,
) -> str:
    """Render the full prompt for one example.

    The prompt is also the key under which samples, ground truth and
    predictions are joined, so it must be built identically at every stage.
    """
    instruction = _format_instruction(config, binned_answers, include_confidence)
    additional = ADDITIONAL_PROMPTS.get(config.prompt_name, "")
    full_instruction = f"{instruction}\n{additional}".strip()

    question = example.get("question", example.get("inputs", ""))

    if "candidates" in example:
        choices = "".join(f"({key}) {value}\n" for key, value in example["candidates"].items())
        return f"{full_instruction}\n\nQuestion: {question}\n{choices}"

    return f"{full_instruction}\n\nQuestion: {question}"
