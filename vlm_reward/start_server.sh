#!/bin/bash
set -euo pipefail

# Unified VLM Reward Server launcher.
# Usage:
#   # For contrastive / completion / progress / roboreward (Qwen3-VL-8B based):
#   MODEL_PATH=/path/to/vlm_checkpoint bash start_server.sh
#
#   # For robometer (Qwen3-VL-4B + reward heads):
#   MODEL_PATH=/path/to/Robometer-4B BASE_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct bash start_server.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="${SCRIPT_DIR}/vlm_reward_server.py"

MODEL_PATH=${MODEL_PATH:?Please set MODEL_PATH}
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
if [[ "${MODEL_PATH}" == /* ]] && [ ! -e "${MODEL_PATH}" ]; then
  echo "ERROR: Local MODEL_PATH does not exist: ${MODEL_PATH}"
  exit 1
fi

echo "Starting VLM Reward Server"
echo "  host=${HOST} port=${PORT} gpu=${GPU_ID}"
echo "  model=${MODEL_PATH}"
if [ -n "${BASE_MODEL_PATH}" ]; then
  echo "  base_model=${BASE_MODEL_PATH}"
fi

CMD=("${PYTHON_BIN}" "${SERVER_SCRIPT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --gpu_id "${GPU_ID}" \
  --model_path "${MODEL_PATH}")

if [ -n "${BASE_MODEL_PATH}" ]; then
  CMD+=(--base_model_path "${BASE_MODEL_PATH}")
fi

exec "${CMD[@]}"
