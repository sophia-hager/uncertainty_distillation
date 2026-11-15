#!/usr/bin/env bash
# Sample the teacher over a dataset, one shard at a time.
#
# Sampling is the expensive stage (hundreds of generations per question), and it
# is embarrassingly parallel: each shard writes its own file and the shards are
# globbed together in the next stage. Run the shards sequentially with this
# script, or launch them independently on whatever scheduler you use by passing
# a single --shard_index to scripts/sample_teacher.py.
#
# Usage: run/sample_teacher.sh [MODEL] [DATASET] [NUM_SHARDS]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${1:-meta-llama/Llama-3.2-3B-Instruct}"
DATASET="${2:-siqa}"
NUM_SHARDS="${3:-32}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"

for (( shard=1; shard<=NUM_SHARDS; shard++ )); do
    echo "=== Shard ${shard}/${NUM_SHARDS} ==="
    python3 scripts/sample_teacher.py \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --num_samples "${NUM_SAMPLES}" \
        --num_shards "${NUM_SHARDS}" \
        --shard_index "${shard}"
done
