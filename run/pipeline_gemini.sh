#!/usr/bin/env bash
# End-to-end Gemini pipeline on a small slice of MMLU-SemiPro.
#
# Prerequisites:
#   gcloud auth application-default login
#   export GOOGLE_CLOUD_PROJECT=<your project>
#   export UD_GCS_BUCKET=<a GCS bucket you own>
#
# Tuning and batch inference are asynchronous cloud jobs, so a full run does
# not finish in one invocation: this script starts the experiment, and
# re-running `continue` advances it each time its jobs complete.
#
# Usage: run/pipeline_gemini.sh [CONFIG] [OVERRIDES...]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${1:-configs/gemini_mmlu.yaml}"
shift || true

: "${UD_GCS_BUCKET:?Set UD_GCS_BUCKET to a GCS bucket you own}"
: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT to your GCP project}"

python3 scripts/gemini_experiment.py start --config "${CONFIG}" "$@"
python3 scripts/gemini_experiment.py list

echo
echo "Re-run the following whenever the cloud jobs finish, until the stage is COMPLETE:"
echo "  python3 scripts/gemini_experiment.py continue 0"
