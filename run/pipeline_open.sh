#!/usr/bin/env bash
# End-to-end open-ended (free-text) pipeline on GSM8K.
#
# Differs from the multiple-choice pipeline in two places: samples are grouped
# into semantic clusters rather than counted by answer letter, and evaluation
# uses an LLM judge (set OPENAI_API_KEY in your environment).
#
# Usage: run/pipeline_open.sh [MODEL]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${1:-mistralai/Ministral-8B-Instruct-2410}"
MODEL_NAME="${MODEL##*/}"
NUM_BINS="${NUM_BINS:-5}"
LIMIT="${LIMIT:-500}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}_open}"

# 1. Sample the teacher and cluster the samples semantically.
python3 scripts/sample_teacher_open.py \
    --model "${MODEL}" --dataset gsm8k \
    --limit "${LIMIT}" --num_samples "${NUM_SAMPLES}"

# 2. Calibrate, bin and verbalize cluster frequencies as fine-tuning data.
python3 scripts/build_finetuning_data_open.py \
    --predictions_path "samples/open_gsm8k_${MODEL_NAME}/*.jsonl" \
    --output_path "datasets/${RUN_NAME}.jsonl" \
    --scheme append_all --num_incorrect 1 \
    --calibration_method isotonic --n_bins "${NUM_BINS}"

# 3+4. Fine-tune, predict and score. The held-out slice starts where teacher
# sampling stopped, so the student is never evaluated on its own training data.
python3 scripts/write_dataset.py --dataset gsm8k --split test \
    --shuffle_seed -1 --output data/gsm8k_test.jsonl

TASK=open NUM_BINS="${NUM_BINS}" run/train_and_evaluate.sh \
    "${RUN_NAME}" "datasets/${RUN_NAME}.jsonl" data/gsm8k_test.jsonl \
    --model "${MODEL}" --num_epochs 1 --learning_rate 3e-5 --num_bins "${NUM_BINS}" \
    --use_lora --lora_r 8 --lora_alpha 16
