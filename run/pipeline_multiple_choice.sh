#!/usr/bin/env bash
# End-to-end multiple-choice pipeline on a small slice of data.
#
# This is a worked example of the four stages; scale NUM_SHARDS, NUM_SAMPLES and
# LIMIT up for a real run.
#
# Usage: run/pipeline_multiple_choice.sh [MODEL]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${1:-meta-llama/Llama-3.2-3B-Instruct}"
MODEL_NAME="${MODEL##*/}"
DATASET="${DATASET:-siqa}"
NUM_BINS="${NUM_BINS:-5}"
LIMIT="${LIMIT:-500}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}_${DATASET}}"

# 1. Sample the teacher's predictive distribution.
python3 scripts/sample_teacher.py \
    --model "${MODEL}" --dataset "${DATASET}" \
    --limit "${LIMIT}" --num_samples "${NUM_SAMPLES}"

# 2. Calibrate, bin and verbalize it as fine-tuning data.
python3 scripts/build_finetuning_data.py \
    --predictions_path "samples/${DATASET}_${MODEL_NAME}/*.jsonl" \
    --output_path "datasets/${RUN_NAME}.jsonl" \
    --scheme append_all --num_incorrect 1 \
    --calibration_method isotonic --n_bins "${NUM_BINS}"

# 3+4. Fine-tune the student, then predict and score on held-out data.
python3 scripts/write_dataset.py --dataset arc --split validation \
    --output data/arc_validation.jsonl

NUM_BINS="${NUM_BINS}" run/train_and_evaluate.sh \
    "${RUN_NAME}" "datasets/${RUN_NAME}.jsonl" data/arc_validation.jsonl \
    --model "${MODEL}" --num_epochs 1 --learning_rate 3e-5 --num_bins "${NUM_BINS}"
