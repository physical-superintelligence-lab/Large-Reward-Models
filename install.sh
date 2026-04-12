#!/bin/bash
# Install Large-Reward-Models into an existing RLinf installation.
# Usage: bash install.sh /path/to/RLinf

set -e

RLINF_DIR="${1:?Usage: bash install.sh /path/to/RLinf}"

if [ ! -f "$RLINF_DIR/rlinf/envs/__init__.py" ]; then
    echo "Error: $RLINF_DIR does not look like a valid RLinf directory."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Installing Large-Reward-Models into $RLINF_DIR ==="

# 1. Copy env file
echo "[1/4] Copying vlm_maniskill_env.py ..."
cp "$SCRIPT_DIR/envs/vlm_maniskill_env.py" "$RLINF_DIR/rlinf/envs/maniskill/vlm_maniskill_env.py"

echo "[1/4] Copying recording_maniskill_env.py ..."
cp "$SCRIPT_DIR/envs/recording_maniskill_env.py" "$RLINF_DIR/rlinf/envs/maniskill/recording_maniskill_env.py"

# 2. Copy VLM reward client
echo "[2/4] Copying vlm_reward_client.py ..."
cp "$SCRIPT_DIR/vlm_reward/vlm_reward_client.py" "$RLINF_DIR/rlinf/envs/maniskill/vlm_reward_client.py"

# 3. Copy configs
echo "[3/5] Copying config files ..."
cp "$SCRIPT_DIR/config/"*.yaml "$RLINF_DIR/examples/embodiment/config/"
cp "$SCRIPT_DIR/config/env/"*.yaml "$RLINF_DIR/examples/embodiment/config/env/"
cp "$SCRIPT_DIR/config/eval/"*.yaml "$RLINF_DIR/examples/embodiment/config/"
cp "$SCRIPT_DIR/config/model/"*.yaml "$RLINF_DIR/examples/embodiment/config/model/"
cp "$SCRIPT_DIR/config/training_backend/"*.yaml "$RLINF_DIR/examples/embodiment/config/training_backend/"

# 4. Copy eval scripts
echo "[4/5] Copying eval scripts ..."
mkdir -p "$RLINF_DIR/examples/embodiment/eval"
cp "$SCRIPT_DIR/eval/"*.py "$RLINF_DIR/examples/embodiment/eval/"

# 5. Patch RLinf to register vlm_maniskill env type
echo "[5/5] Patching RLinf for vlm_maniskill support ..."

python3 - "$RLINF_DIR" <<'PYEOF'
import sys, os, re

rlinf_dir = sys.argv[1]

# 4a. Patch rlinf/envs/__init__.py
init_file = os.path.join(rlinf_dir, "rlinf", "envs", "__init__.py")
with open(init_file, "r") as f:
    content = f.read()

if "VLM_MANISKILL" not in content:
    # Add enum entry after MANISKILL = "maniskill"
    content = content.replace(
        'MANISKILL = "maniskill"',
        'MANISKILL = "maniskill"\n    VLM_MANISKILL = "vlm_maniskill"\n    RECORDING_MANISKILL = "recording_maniskill"'
    )
    # Add env class registration before "else:\n        raise NotImplementedError"
    content = content.replace(
        '    else:\n        raise NotImplementedError',
        '    elif env_type == SupportedEnvType.VLM_MANISKILL:\n'
        '        from rlinf.envs.maniskill.vlm_maniskill_env import VLMManiskillEnv\n'
        '\n'
        '        return VLMManiskillEnv\n'
        '    elif env_type == SupportedEnvType.RECORDING_MANISKILL:\n'
        '        from rlinf.envs.maniskill.recording_maniskill_env import RecordingManiskillEnv\n'
        '\n'
        '        return RecordingManiskillEnv\n'
        '    else:\n        raise NotImplementedError'
    )
    with open(init_file, "w") as f:
        f.write(content)
    print(f"  Patched {init_file}")
else:
    print(f"  {init_file} already patched, skipping.")

# 4b. Patch rlinf/envs/action_utils.py
action_file = os.path.join(rlinf_dir, "rlinf", "envs", "action_utils.py")
with open(action_file, "r") as f:
    content = f.read()

if "VLM_MANISKILL" not in content:
    content = content.replace(
        "== SupportedEnvType.MANISKILL",
        "in (SupportedEnvType.MANISKILL, SupportedEnvType.VLM_MANISKILL, SupportedEnvType.RECORDING_MANISKILL)"
    )
    with open(action_file, "w") as f:
        f.write(content)
    print(f"  Patched {action_file}")
else:
    print(f"  {action_file} already patched, skipping.")

# 4c. Patch rlinf/config.py
config_file = os.path.join(rlinf_dir, "rlinf", "config.py")
with open(config_file, "r") as f:
    content = f.read()

if "VLM_MANISKILL" not in content:
    content = content.replace(
        "== SupportedEnvType.MANISKILL",
        "in (SupportedEnvType.MANISKILL, SupportedEnvType.VLM_MANISKILL, SupportedEnvType.RECORDING_MANISKILL)"
    )
    with open(config_file, "w") as f:
        f.write(content)
    print(f"  Patched {config_file}")
else:
    print(f"  {config_file} already patched, skipping.")

PYEOF

echo ""
echo "=== Installation complete ==="
echo ""
echo "Next steps:"
echo "  1. Update model paths in config/*.yaml (actor.model.model_path, rollout.model.model_path)"
echo "  2. Start VLM reward server: cd $SCRIPT_DIR/vlm_reward && bash start_server.sh"
echo "  3. Start training (inside Docker): bash run_embodiment.sh <mode> MANISKILL"
echo "     Modes: contrastive, completion, progress, roboreward, robometer"
