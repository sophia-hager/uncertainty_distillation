"""System prompts for teacher sampling and for the distilled student.

The teacher is prompted to answer *without* a confidence statement; its
uncertainty is estimated by sampling.  The student is prompted to additionally
verbalize a confidence bin, which is what fine-tuning teaches it to calibrate.
"""

#: Verbalized confidence vocabulary, by number of bins.  Index == bin index.
CONFIDENCE_BINS = {
    2: ["low", "high"],
    5: ["very low", "low", "medium", "high", "very high"],
    7: [
        "extremely low",
        "very low",
        "low",
        "medium",
        "high",
        "very high",
        "extremely high",
    ],
    10: ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"],
}

_MC_EXAMPLE = (
    "\"If each of Lisa's 7 chickens lays 6 eggs, how many eggs does Lisa have? "
    "\n A) 24 \n B) 35 \n C) 42 \n D) 50 \n"
    "<reasoning> This can be solved with multiplication. The answer is 7*6, or 42.</reasoning> "
    "<answer> C) 42 </answer>"
)

#: Phrase from the one-shot example above.  Sampled reasoning chains that merely
#: parrot it carry no information about the question, so we drop them.
MC_EXAMPLE_REASONING = "The answer is 7*6, or 42."

#: Prompt used when sampling the teacher on multiple-choice data (no confidence).
MC_SAMPLING_PROMPT = (
    "Answer the following question. Enclose concise reasoning in "
    "<reasoning> </reasoning> tags and your FINAL answer in <answer> </answer> "
    "tags without any of your work, like this: "
    f"{_MC_EXAMPLE}.\" Your answer should not include words."
)

#: Prompt used when sampling the teacher on open-ended (free-text) data.
OPEN_SAMPLING_PROMPT = (
    "You are a helpful AI assistant. Answer the following math question as "
    "briefly as possible and accurately."
)


def _confidence_vocabulary(num_bins: int) -> str:
    if num_bins not in CONFIDENCE_BINS:
        raise ValueError(f"Unsupported number of bins: {num_bins}")
    if num_bins == 10:
        return "in increments of 10%"
    return ", ".join(CONFIDENCE_BINS[num_bins])


def multiple_choice_prompt(num_bins: int = 5) -> str:
    """System prompt asking for reasoning, an answer letter, and a confidence."""
    vocabulary = _confidence_vocabulary(num_bins)
    example_confidence = CONFIDENCE_BINS[num_bins][-1] if num_bins != 10 else "70%"
    return (
        f"Answer the following question and state confidence in the answer ({vocabulary}). "
        "Enclose concise reasoning in <reasoning> </reasoning> tags, confidence in "
        "<confidence> </confidence> tags, and the letter of your FINAL answer in "
        "<answer> </answer> tags without any of your work, like this: "
        f"{_MC_EXAMPLE} <confidence>{example_confidence}</confidence>.\" "
        "Your answer should not include words."
    )


def open_ended_prompt(num_bins: int = 5) -> str:
    """System prompt for free-text answers followed by a verbalized confidence."""
    vocabulary = _confidence_vocabulary(num_bins)
    example_confidence = CONFIDENCE_BINS[num_bins][-1] if num_bins != 10 else "70%"
    return (
        "You are a helpful AI assistant. Answer the following math question as "
        "briefly as possible and accurately. Enclose confidence in the answer "
        f"({vocabulary}) after the answer in <confidence> </confidence> tags, "
        f"like so: <confidence> {example_confidence} </confidence>."
    )


def system_prompt(task: str, num_bins: int = 5) -> str:
    """Student system prompt for ``task`` in {``multiple_choice``, ``open``}."""
    if task == "multiple_choice":
        return multiple_choice_prompt(num_bins)
    if task == "open":
        return open_ended_prompt(num_bins)
    raise ValueError(f"Unknown task: {task}")


def sampling_prompt(task: str) -> str:
    """Teacher system prompt for ``task`` in {``multiple_choice``, ``open``}."""
    if task == "multiple_choice":
        return MC_SAMPLING_PROMPT
    if task == "open":
        return OPEN_SAMPLING_PROMPT
    raise ValueError(f"Unknown task: {task}")
