"""Chat-template helpers shared by sampling, training and inference."""


def create_conversation(sample, system_message, predict=False):
    """Wrap a ``{prompt, completion}`` example as a chat conversation.

    When ``predict`` is true the assistant turn is omitted so the model is asked
    to produce it.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": sample["prompt"]},
    ]
    if not predict:
        messages.append({"role": "assistant", "content": sample["completion"]})
    return {"messages": messages}


def apply_chat_template(example, tokenizer, tokenize=False, add_generation_prompt=False):
    """Render ``example["messages"]`` with the tokenizer's chat template.

    Chat templates already insert the special tokens a model expects, so when
    rendering to text we disable ``add_special_tokens`` to avoid duplicating
    them (this is not an issue when ``tokenize=True``).
    """
    messages = example["messages"]
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})

    if tokenize:
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt
        )
    else:
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_special_tokens=False,
            add_generation_prompt=add_generation_prompt,
        )
    return example


#: Fallback template for base models whose tokenizer ships without one.
DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n' + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n"
    "{% endfor %}"
)
