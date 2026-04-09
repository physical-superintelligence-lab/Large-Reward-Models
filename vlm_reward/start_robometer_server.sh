#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="${SCRIPT_DIR}/rlinf_integration/vlm_reward_server.py"

MODEL_PATH=${MODEL_PATH:-/data/yanruwu/models/Robometer-4B}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/data/yanruwu/models/Qwen3-VL-4B-Instruct}
PORT=${PORT:-5003}
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

if [[ "${BASE_MODEL_PATH}" == /* ]] && [ ! -e "${BASE_MODEL_PATH}" ]; then
  echo "ERROR: Local BASE_MODEL_PATH does not exist: ${BASE_MODEL_PATH}"
  exit 1
fi

echo "Starting Robometer VLM server"
echo "  host=${HOST} port=${PORT} gpu=${GPU_ID}"
echo "  model=${MODEL_PATH}"
echo "  base_model=${BASE_MODEL_PATH}"

exec "${PYTHON_BIN}" "${SERVER_SCRIPT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --gpu_id "${GPU_ID}" \
  --base_model_path "${BASE_MODEL_PATH}" \
  --model_path "${MODEL_PATH}"