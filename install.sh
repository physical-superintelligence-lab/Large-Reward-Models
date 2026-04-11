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

# 2. Copy VLM reward client
echo "[2/4] Copying vlm_reward_client.py ..."
cp "$SCRIPT_DIR/vlm_reward/vlm_reward_client.py" "$RLINF_DIR/rlinf/envs/maniskill/vlm_reward_client.py"

# 3. Copy configs
echo "[3/4] Copying config files ..."
cp "$SCRIPT_DIR/config/"*.yaml "$RLINF_DIR/examples/embodiment/config/"
cp "$SCRIPT_DIR/config/env/"*.yaml "$RLINF_DIR/examples/embodiment/config/env/"
cp "$SCRIPT_DIR/config/model/"*.yaml "$RLINF_DIR/examples/embodiment/config/model/"
cp "$SCRIPT_DIR/config/training_backend/"*.yaml "$RLINF_DIR/examples/embodiment/config/training_backend/"

# 4. Patch RLinf to register vlm_maniskill env type
echo "[4/4] Patching RLinf for vlm_maniskill support ..."

# 4a. Patch rlinf/envs/__init__.py - add VLM_MANISKILL to enum
INIT_FILE="$RLINF_DIR/rlinf/envs/__init__.py"
if ! grep -q "VLM_MANISKILL" "$INIT_FILE"; then
    # Add enum entry after MANISKILL line
    sed -i '/MANISKILL = "maniskill"/a\    VLM_MANISKILL = "vlm_maniskill"' "$INIT_FILE"

    # Add env class registration before the final else clause
    sed -i '/raise NotImplementedError.*Environment type/i\
    elif env_type == SupportedEnvType.VLM_MANISKILL:\
        from rlinf.envs.maniskill.vlm_maniskill_env import VLMManiskillEnv\
        return VLMManiskillEnv' "$INIT_FILE"

    echo "  Patched $INIT_FILE"
else
    echo "  $INIT_FILE already patched, skipping."
fi

# 4b. Patch rlinf/envs/action_utils.py - add VLM_MANISKILL to action handling
ACTION_FILE="$RLINF_DIR/rlinf/envs/action_utils.py"
if ! grep -q "VLM_MANISKILL" "$ACTION_FILE"; then
    # Replace "== SupportedEnvType.MANISKILL" with "in (SupportedEnvType.MANISKILL, SupportedEnvType.VLM_MANISKILL)"
    sed -i 's/== SupportedEnvType\.MANISKILL/in (SupportedEnvType.MANISKILL, SupportedEnvType.VLM_MANISKILL)/g' "$ACTION_FILE"
    echo "  Patched $ACTION_FILE"
else
    echo "  $ACTION_FILE already patched, skipping."
fi

# 4c. Patch rlinf/config.py - add VLM_MANISKILL to validation
CONFIG_FILE="$RLINF_DIR/rlinf/config.py"
if ! grep -q "VLM_MANISKILL" "$CONFIG_FILE"; then
    # Replace "== SupportedEnvType.MANISKILL" with "in (SupportedEnvType.MANISKILL, SupportedEnvType.VLM_MANISKILL)"
    sed -i 's/== SupportedEnvType\.MANISKILL/in (SupportedEnvType.MANISKILL, SupportedEnvType.VLM_MANISKILL)/g' "$CONFIG_FILE"
    echo "  Patched $CONFIG_FILE"
else
    echo "  $CONFIG_FILE already patched, skipping."
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "Next steps:"
echo "  1. Update model paths in config/*.yaml (actor.model.model_path, rollout.model.model_path)"
echo "  2. Start VLM reward server: cd $SCRIPT_DIR/vlm_reward && bash start_server.sh"
echo "  3. Start training (inside Docker): bash run_embodiment.sh <mode> MANISKILL"
echo "     Modes: contrastive, completion, progress, roboreward, robometer"
