#!/usr/bin/env bash
# Repeat closed-loop evaluation across several seeds and lay results out as
# <results-dir>/<method>/seed_<N>/, the layout eval/summarize_closed_loop.py
# expects for mean +/- 95% CI (and paired comparisons across methods, if you
# run this once per method into the same --results-dir).
#
# Why repeat at all: env.eval.use_fixed_reset_state_ids keeps the evaluation
# episodes identical across runs, but action sampling during rollout
# (do_sample=True) is not seeded to a fixed outcome, so a single 320-env
# closed-loop evaluation of the SAME checkpoint can vary by several points of
# success rate. A single run is not a reliable estimate; report the mean and
# CI over multiple seeds instead.
#
# Usage (inside the RLinf Docker container, matching the single-eval example
# in the README):
#   source switch_env openpi
#   bash run_closed_loop_seeds.sh \
#     --rlinf-dir /workspace/RLinf \
#     --results-dir ./results/closed_loop \
#     --method lrm_completion \
#     --model-path /path/to/SFT_or_checkpoint_dir \
#     --ckpt-path /path/to/checkpoint/actor/model_state_dict/full_weights.pt \
#     --seeds "0 1 2 3 4"
#
# Omit --ckpt-path to evaluate the SFT baseline directly (model-path alone).
set -euo pipefail

RLINF_DIR=""
RESULTS_DIR=""
METHOD=""
MODEL_PATH=""
CKPT_PATH=""
SEEDS="0 1 2 3 4"
TOTAL_NUM_ENVS=320

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rlinf-dir) RLINF_DIR="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --ckpt-path) CKPT_PATH="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --total-num-envs) TOTAL_NUM_ENVS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for name in RLINF_DIR RESULTS_DIR METHOD MODEL_PATH; do
  [[ -n "${!name}" ]] || { echo "Missing required --${name,,} (dashes for underscores)" >&2; exit 2; }
done

EMBODIED_PATH="${RLINF_DIR}/examples/embodiment"
export EMBODIED_PATH
export PYTHONPATH="${RLINF_DIR}:${PYTHONPATH:-}"
export ROBOT_PLATFORM=MANISKILL MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

METHOD_DIR="${RESULTS_DIR}/${METHOD}"
CKPT_ARGS=()
[[ -n "${CKPT_PATH}" ]] && CKPT_ARGS=(runner.ckpt_path="${CKPT_PATH}")

FAILED=()
for seed in ${SEEDS}; do
  out="${METHOD_DIR}/seed_${seed}"
  if [[ -d "${out}" ]] && compgen -G "${out}/**/events.out.tfevents.*" >/dev/null 2>&1; then
    echo "[skip] seed ${seed} already done: ${out}"
    continue
  fi
  # A closed-loop eval occasionally dies on a transient Ray/gloo worker crash
  # unrelated to the policy or reward model. Retry once before giving up, and
  # keep going on the remaining seeds rather than aborting the whole sweep.
  ok=0
  for attempt in 1 2; do
    rm -rf "${out}"; mkdir -p "${out}"
    echo "=== ${METHOD} seed=${seed} (attempt ${attempt}) ==="
    if python "${EMBODIED_PATH}/eval_embodied_agent.py" \
        --config-path "${EMBODIED_PATH}/config/" \
        --config-name closed_loop_eval \
        runner.logger.log_path="${out}" \
        "${CKPT_ARGS[@]}" \
        actor.seed="${seed}" \
        env.eval.seed="${seed}" \
        algorithm.eval_rollout_epoch=1 \
        env.eval.total_num_envs="${TOTAL_NUM_ENVS}" \
        env.eval.use_fixed_reset_state_ids=true \
        rollout.model.model_path="${MODEL_PATH}" \
        2>&1 | tee "${out}/eval.log" \
       && compgen -G "${out}/**/events.out.tfevents.*" >/dev/null 2>&1; then
      ok=1; break
    fi
    echo "[warn] seed ${seed} attempt ${attempt} failed" >&2
    sleep 5
  done
  [[ ${ok} -eq 1 ]] || { echo "[FAIL] seed ${seed} gave up after 2 attempts" >&2; FAILED+=("${seed}"); }
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[FAILED SEEDS]: ${FAILED[*]}" >&2
fi

echo
echo "Done. Summarize with:"
echo "  python ${EMBODIED_PATH}/eval/summarize_closed_loop.py --results-dir ${RESULTS_DIR} --methods ${METHOD}"
