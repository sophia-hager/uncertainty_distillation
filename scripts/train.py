"""Stage 3: supervised fine-tuning of the student on verbalized confidences.

Trains with TRL's ``SFTTrainer`` on the ``{prompt, completion}`` pairs produced
in stage 2, computing the loss on the completion only.  Full fine-tuning is the
default; ``--use_lora`` trains adapters instead, which is how the larger
(8B-class) models are trained.

Example::

    accelerate launch scripts/train.py \
        --train_data datasets/llama_siqa.jsonl \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --run_name llama_siqa --num_epochs 1 --learning_rate 3e-5
"""

import argparse
import os

import torch
from accelerate import Accelerator, PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import _bootstrap  # noqa: F401

from uncertainty_distillation.chat import (
    DEFAULT_CHAT_TEMPLATE,
    apply_chat_template,
    create_conversation,
)
from uncertainty_distillation.paths import RUNS_DIR
from uncertainty_distillation.prompts import system_prompt


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data", required=True, help="Training JSONL from stage 2.")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Starting point, e.g. mistralai/Ministral-8B-Instruct-2410")
    parser.add_argument("--task", choices=["multiple_choice", "open"],
                        default="multiple_choice")
    parser.add_argument("--run_name", default="uncertainty_distillation")
    parser.add_argument("--output_dir", default=None, help="Default: runs/<run_name>")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_bins", type=int, default=5, choices=[2, 5, 7, 10])
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size.")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--valid_fraction", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="Twice r is a common rule of thumb.")
    parser.add_argument("--attn_implementation", default="flash_attention_2",
                        help="Use 'sdpa' if FlashAttention 2 is not installed.")
    parser.add_argument("--report_to", default="none",
                        help="Trainer logging integration, e.g. 'wandb' or 'none'.")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true", default=True)
    parser.add_argument("--skip_train", action="store_true",
                        help="Set up everything but training (debugging).")
    return parser.parse_args()


def prepare_datasets(args, accelerator, tokenizer):
    dataset = load_dataset("json", data_files={"train": args.train_data}, split="train")
    accelerator.print(f"Example training record: {dataset[0]}")

    raw_datasets = dataset.train_test_split(test_size=args.valid_fraction)
    accelerator.print(f"Training set size: {len(raw_datasets['train'])}")
    accelerator.print(f"Validation set size: {len(raw_datasets['test'])}")

    if getattr(tokenizer, "chat_template", None):
        accelerator.print("Tokenizer has a chat template.")
    else:
        accelerator.print("Tokenizer has no chat template; using the default one.")
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    system_message = system_prompt(args.task, args.num_bins)

    raw_datasets = raw_datasets.map(
        create_conversation,
        remove_columns=list(raw_datasets["train"].features),
        batched=False,
        fn_kwargs={"system_message": system_message},
        num_proc=4,
        load_from_cache_file=not args.overwrite_cache,
        desc="Creating conversations.",
    )
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        remove_columns=list(raw_datasets["train"].features),
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=4,
        load_from_cache_file=not args.overwrite_cache,
        desc="Applying chat template.",
    )
    return raw_datasets


def main():
    args = get_args()
    accelerator = Accelerator()
    accelerator.print(f"{accelerator.num_processes} {accelerator.distributed_type}")

    output_dir = args.output_dir or str(RUNS_DIR / args.run_name)
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.model_max_length > 100_000:
        # Sensible default for models that declare no real maximum.
        tokenizer.model_max_length = 10_000

    with accelerator.main_process_first():
        raw_datasets = prepare_datasets(args, accelerator, tokenizer)

    # Each process loads the model onto its own device.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map={"": PartialState().process_index},
    )

    peft_config = (
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            modules_to_save=None,
            task_type="CAUSAL_LM",
        )
        if args.use_lora
        else None
    )

    accelerator.print(f"Learning rate: {args.learning_rate}")
    accelerator.print(f"Accumulation steps: {args.accumulation_steps}")

    training_args = SFTConfig(
        run_name=args.run_name,
        output_dir=output_dir,
        max_length=args.max_length,
        padding_free=True,
        completion_only_loss=True,                  # train on the completion only
        disable_tqdm=args.disable_tqdm,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        gradient_checkpointing=True,                # trade compute for memory
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=args.logging_steps,
        bf16=True,
        tf32=True,
        learning_rate=args.learning_rate,
        max_grad_norm=0.5,
        warmup_ratio=0.1,
        lr_scheduler_type=args.lr_scheduler_type,
        label_names=[],                             # report perplexity on validation data
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
    )

    trainer = SFTTrainer(
        model,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        args=training_args,
        peft_config=peft_config,
    )
    trainer.can_return_loss = True

    if args.skip_train:
        accelerator.print("Skipping training entirely!")
    else:
        trainer.train()

    best_dir = os.path.join(output_dir, "best")
    accelerator.print(f"Saving best model to {best_dir}")
    trainer.save_model(best_dir)


if __name__ == "__main__":
    main()
