#!/usr/bin/env bash
# Fine-tune a student on distilled confidences, then predict and score.
#
# Everything is written under runs/<RUN_NAME>/ inside the repository:
#   best/             the fine-tuned model
#   predictions.jsonl generations on the evaluation set
#   metrics.txt       accuracy per confidence bin and AUROC
#
# Usage: run/train_and_evaluate.sh RUN_NAME TRAIN_DATA EVAL_DATA [TRAIN ARGS...]
# Example:
#   run/train_and_evaluate.sh llama_siqa datasets/llama_siqa.jsonl data/arc_validation.jsonl \
#       --model meta-llama/Llama-3.2-3B-Instruct --num_epochs 1 --learning_rate 3e-5

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 RUN_NAME TRAIN_DATA EVAL_DATA [TRAIN ARGS...]"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_NAME="$1"
TRAIN_DATA="$2"
EVAL_DATA="$3"
shift 3

TASK="${TASK:-multiple_choice}"
NUM_BINS="${NUM_BINS:-5}"
RUN_DIR="runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

accelerate launch scripts/train.py \
    --train_data "${TRAIN_DATA}" \
    --run_name "${RUN_NAME}" \
    --output_dir "${RUN_DIR}" \
    --task "${TASK}" \
    --num_bins "${NUM_BINS}" \
    "$@"

python3 scripts/predict.py \
    --model_name_or_path "${RUN_DIR}/best" \
    --data "${EVAL_DATA}" \
    --task "${TASK}" \
    --num_bins "${NUM_BINS}" \
    --output_path "${RUN_DIR}/predictions.jsonl"

if [ "${TASK}" = "open" ]; then
    EVAL_SCRIPT=scripts/evaluate_open.py
else
    EVAL_SCRIPT=scripts/evaluate.py
fi

python3 "${EVAL_SCRIPT}" "${RUN_DIR}/predictions.jsonl" \
    --num_bins "${NUM_BINS}" \
    --metrics_path "${RUN_DIR}/metrics.json" | tee "${RUN_DIR}/metrics.txt"
