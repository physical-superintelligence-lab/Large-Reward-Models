#!/bin/bash
set -euo pipefail

# Unified VLM Reward Server launcher.
# Usage (single model, one modality):
#   MODEL_PATH=/path/to/vlm_checkpoint bash start_server.sh
#
# Usage (zero-shot base VLM control experiment -- no fine-tuning):
#   MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct bash start_server.sh
#
# Usage (tri mode: three adapters, one shared backbone, one GPU):
#   TRI_CONTRASTIVE_ADAPTER=/path/contrastive \
#   TRI_PROGRESS_ADAPTER=/path/progress \
#   TRI_COMPLETION_ADAPTER=/path/completion \
#     BASE_MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct bash start_server.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="${SCRIPT_DIR}/vlm_reward_server.py"

TRI_CONTRASTIVE_ADAPTER=${TRI_CONTRASTIVE_ADAPTER:-""}
TRI_PROGRESS_ADAPTER=${TRI_PROGRESS_ADAPTER:-""}
TRI_COMPLETION_ADAPTER=${TRI_COMPLETION_ADAPTER:-""}
TRI_MODE=0
if [ -n "${TRI_CONTRASTIVE_ADAPTER}" ] || [ -n "${TRI_PROGRESS_ADAPTER}" ] || [ -n "${TRI_COMPLETION_ADAPTER}" ]; then
  if [ -z "${TRI_CONTRASTIVE_ADAPTER}" ] || [ -z "${TRI_PROGRESS_ADAPTER}" ] || [ -z "${TRI_COMPLETION_ADAPTER}" ]; then
    echo "ERROR: tri mode requires all three of TRI_CONTRASTIVE_ADAPTER, TRI_PROGRESS_ADAPTER, TRI_COMPLETION_ADAPTER" >&2
    exit 1
  fi
  TRI_MODE=1
fi
# MODEL_PATH is not needed in tri mode.
if [ "${TRI_MODE}" -eq 0 ]; then
  MODEL_PATH=${MODEL_PATH:?Please set MODEL_PATH (or the three TRI_*_ADAPTER vars for tri mode)}
else
  MODEL_PATH=${MODEL_PATH:-""}
fi
BASE_MODEL_PATH=${BASE_MODEL_PATH:-""}
PORT=${PORT:-5002}
HOST=${HOST:-0.0.0.0}
GPU_ID=${GPU_ID:-0}
PYTHON_BIN=${PYTHON_BIN:-python}

if [ -n "${VLM_CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${VLM_CONDA_ENV}"
fi

if [ ! -f "${SERVER_SCRIPT}" ]; then
  echo "ERROR: Missing server script: ${SERVER_SCRIPT}"
  exit 1
fi

# Local paths must exist; hub model ids (e.g. org/model) are allowed.
if [ -n "${MODEL_PATH}" ] && [[ "${MODEL_PATH}" == /* ]] && [ ! -e "${MODEL_PATH}" ]; then
  echo "ERROR: Local MODEL_PATH does not exist: ${MODEL_PATH}"
  exit 1
fi

echo "Starting VLM Reward Server"
echo "  host=${HOST} port=${PORT} gpu=${GPU_ID}"
if [ "${TRI_MODE}" -eq 1 ]; then
  echo "  tri adapters: contrastive=${TRI_CONTRASTIVE_ADAPTER} progress=${TRI_PROGRESS_ADAPTER} completion=${TRI_COMPLETION_ADAPTER}"
else
  echo "  model=${MODEL_PATH}"
fi
if [ -n "${BASE_MODEL_PATH}" ]; then
  echo "  base_model=${BASE_MODEL_PATH}"
fi

CMD=("${PYTHON_BIN}" "${SERVER_SCRIPT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --gpu_id "${GPU_ID}")

if [ "${TRI_MODE}" -eq 1 ]; then
  CMD+=(
    --tri_contrastive_adapter "${TRI_CONTRASTIVE_ADAPTER}"
    --tri_progress_adapter "${TRI_PROGRESS_ADAPTER}"
    --tri_completion_adapter "${TRI_COMPLETION_ADAPTER}"
  )
else
  CMD+=(--model_path "${MODEL_PATH}")
fi

if [ -n "${BASE_MODEL_PATH}" ]; then
  CMD+=(--base_model_path "${BASE_MODEL_PATH}")
fi

exec "${CMD[@]}"
