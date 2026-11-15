"""Stage 4a: generate predictions with a (fine-tuned) student model.

Reads the ``{prompt, label}`` JSONL written by ``write_dataset.py`` and writes
one record per example with the model's completion, which contains both the
answer and its verbalized confidence.

Example::

    python scripts/predict.py --model_name_or_path runs/llama_siqa/best \
        --data data/arc_validation.jsonl --output_path runs/llama_siqa/predictions.jsonl
"""

import argparse
import json

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import _bootstrap  # noqa: F401

from uncertainty_distillation.chat import apply_chat_template, create_conversation
from uncertainty_distillation.paths import ensure_parent
from uncertainty_distillation.prompts import system_prompt


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--data", required=True, help="Evaluation JSONL.")
    parser.add_argument("--output_path", default="predictions.jsonl")
    parser.add_argument("--task", choices=["multiple_choice", "open"],
                        default="multiple_choice")
    parser.add_argument("--input_field", default="prompt")
    parser.add_argument("--output_field", default="label")
    parser.add_argument("--predict_field", default="llm_completion")
    parser.add_argument("--num_bins", type=int, default=5, choices=[2, 5, 7, 10])
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--decoding_strategy", choices=["greedy", "sample", "beam_search"],
                        default="greedy")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--attn_implementation", default="flash_attention_2",
                        help="Use 'sdpa' if FlashAttention 2 is not installed.")
    parser.add_argument("--print_samples", action="store_true")
    return parser.parse_args()


def generate(model, tokenizer, input_ids, args):
    attention_mask = torch.ones_like(input_ids)
    common = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "tokenizer": tokenizer,
        "stop_strings": [tokenizer.eos_token],
    }
    with torch.inference_mode():
        if args.decoding_strategy == "beam_search":
            return model.generate(**common, do_sample=False, num_beams=args.num_beams,
                                  temperature=args.temperature, early_stopping=True)
        if args.decoding_strategy == "sample":
            return model.generate(**common, do_sample=True, temperature=args.temperature)
        return model.generate(**common, do_sample=False, top_p=None)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    datasets = load_dataset("json", data_files={"predict": args.data})
    system_message = system_prompt(args.task, args.num_bins)

    datasets = datasets.map(
        create_conversation,
        batched=False,
        fn_kwargs={"system_message": system_message, "predict": True},
        num_proc=4,
        desc="Creating conversations.",
    )
    datasets = datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "tokenize": True, "add_generation_prompt": False},
        num_proc=4,
        desc="Applying chat template.",
    )

    predictions = []
    for example in tqdm(datasets["predict"]):
        input_ids = torch.tensor([example["text"]]).to(device)
        outputs = generate(model, tokenizer, input_ids, args)

        # Keep the generated continuation only, without the EOS token.
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:])
        response = response.replace(tokenizer.eos_token, "")

        if args.print_samples:
            print(example[args.input_field])
            print(example[args.output_field])
            print(response)

        predictions.append({
            args.input_field: example[args.input_field],
            args.output_field: example[args.output_field],
            args.predict_field: response,
        })

    print(f"Writing predictions to: {args.output_path}")
    with open(ensure_parent(args.output_path), "w") as output_file:
        for prediction in predictions:
            output_file.write(json.dumps(prediction) + "\n")


if __name__ == "__main__":
    main()
