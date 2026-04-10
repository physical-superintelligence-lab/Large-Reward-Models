# Large Reward Models

RL training code for robotic manipulation with VLM-based reward models. Supports 5 reward modes that can be switched via YAML configuration.

Built on [RLinf](https://github.com/RLinf/RLinf) framework with [ManiSkill](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/maniskill.html) environments and [Pi0.5](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html) policy.

## Reward Modes

| Mode | VLM Model | Signal Type | Description |
|---|---|---|---|
| **Contrastive** | LRM-contrastive (Qwen3-VL-8B) | +1 / -1 / 0 | Compares two consecutive frames to determine progress direction |
| **Completion** | LRM-completion (Qwen3-VL-8B) | Binary 0 / 1 | Single-frame yes/no task completion judgment |
| **Progress** | LRM-progress (Qwen3-VL-8B) | Continuous 0~1 | Single-frame task completion progress estimation |
| **RoboReward** | LRM-roboreward (Qwen3-VL-8B) | Discrete 1-5 → 0~1 | Scores a video trajectory on a 1-5 rubric |
| **Robometer** | Robometer-4B (Qwen3-VL-4B + reward heads) | Continuous 0~1 per frame | Per-frame progress via custom reward heads |

## Prerequisites

- [RLinf](https://github.com/RLinf/RLinf) framework
- VLM reward model (see [Models](#models) below)

## Models

**Policy model (required for all modes):**

Download the Pi0.5 SFT checkpoint following the [RLinf Pi0.5 guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html):

```bash
git lfs install
git clone https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-SFT
```

**VLM reward models (download the one for your chosen mode):**

| Mode | Model | Source |
|---|---|---|
| Contrastive | LRM-contrastive | [HuggingFace](https://huggingface.co/USC-PSI-Lab/LRM-models/tree/main/contrastive) |
| Completion | LRM-completion | [HuggingFace](https://huggingface.co/USC-PSI-Lab/LRM-models/tree/main/completion) |
| Progress | LRM-progress | [HuggingFace](https://huggingface.co/USC-PSI-Lab/LRM-models/tree/main/progress) |
| RoboReward | LRM-roboreward | [HuggingFace](https://huggingface.co/USC-PSI-Lab/LRM-models/tree/main/roboreward) |
| Robometer | Robometer-4B + Qwen3-VL-4B-Instruct | [HuggingFace](https://huggingface.co/robometer/Robometer-4B) |

## Setup

### 1. Set up RLinf with ManiSkill

```bash
git clone https://github.com/RLinf/RLinf.git
cd RLinf

# Start Docker
docker run -it --rm --gpus all \
   --shm-size 20g \
   --network host \
   --name rlinf \
   -v .:/workspace/RLinf \
   rlinf/rlinf:agentic-rlinf0.2-maniskill_libero

# Inside Docker: switch to OpenPI environment (for Pi0.5)
source switch_env openpi
```

Download ManiSkill assets (inside Docker):

```bash
cd /workspace/RLinf/rlinf/envs/maniskill
huggingface-cli download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets
```

### 2. Copy files into RLinf

```bash
# Outside Docker
git clone https://github.com/physical-superintelligence-lab/Large-Reward-Models.git

# Copy unified env into RLinf
cp Large-Reward-Models/envs/vlm_maniskill_env.py RLinf/rlinf/envs/maniskill/vlm_maniskill_env.py

# Copy VLM reward client into RLinf (accessible from Docker)
cp Large-Reward-Models/vlm_reward/vlm_reward_client.py RLinf/rlinf/envs/maniskill/vlm_reward_client.py

# Copy training configs
cp Large-Reward-Models/config/*.yaml RLinf/examples/embodiment/config/
cp Large-Reward-Models/config/env/*.yaml RLinf/examples/embodiment/config/env/
cp Large-Reward-Models/config/model/*.yaml RLinf/examples/embodiment/config/model/
cp Large-Reward-Models/config/training_backend/*.yaml RLinf/examples/embodiment/config/training_backend/
```

### 3. Update model paths

Edit the training YAML for your chosen mode (e.g., `config/contrastive.yaml`) and set the policy model path:

```yaml
actor:
  model:
    model_path: "/path/to/RLinf-Pi05-ManiSkill-25Main-SFT"
rollout:
  model:
    model_path: "/path/to/RLinf-Pi05-ManiSkill-25Main-SFT"
```

## Training

Training requires two processes: a **VLM reward server** on the host and the **RL training** inside Docker.

### Step 1: Start VLM Reward Server (on host, outside Docker)

```bash
cd Large-Reward-Models/vlm_reward

# For contrastive / completion / progress / roboreward (Qwen3-VL-8B based):
MODEL_PATH=/path/to/LRM-contrastive GPU_ID=0 bash start_server.sh

# For robometer (Qwen3-VL-4B + reward heads):
MODEL_PATH=/path/to/Robometer-4B BASE_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct GPU_ID=0 bash start_server.sh
```

The server starts on port 5002 by default. Verify it's running:

```bash
curl http://localhost:5002/health
```

### Step 2: Start RL Training (inside Docker)

```bash
# Inside RLinf Docker container
source switch_env openpi
cd /workspace/RLinf/examples/embodiment

# Train with your chosen reward mode
bash run_embodiment.sh contrastive      # contrastive mode
bash run_embodiment.sh completion       # completion mode
bash run_embodiment.sh progress         # progress mode
bash run_embodiment.sh roboreward       # roboreward mode
bash run_embodiment.sh robometer        # robometer mode
```

### Monitor Training

```bash
tensorboard --logdir ./results --port 6006
```

## Configuration

### Switching Reward Modes

Each mode has a dedicated config file under `config/`. The key parameters that differ:

| Parameter | contrastive | completion | progress | roboreward | robometer |
|---|---|---|---|---|---|
| `vlm_use_comparison` | true | - | - | - | - |
| `vlm_use_roboreward` | - | - | - | true | - |
| `vlm_use_robometer` | - | - | - | - | true |
| `vlm_reward_type` | - | completion | progress | - | progress |
| `vlm_call_interval` | 10 | 5 | 10 | 10 | 10 |
| `vlm_include_initial_image` | - | - | true | - | true |
| `vlm_roboreward_max_frames` | - | - | - | 16 | - |
| `vlm_robometer_max_frames` | - | - | - | - | 16 |

### Common Parameters

| Parameter | Default | Description |
|---|---|---|
| `vlm_reward_scale` | 1.0 | Multiplier applied to VLM reward |
| `vlm_reward_weight` | 1.0 | Weight of VLM reward in final reward |
| `vlm_call_interval` | 10 | Call VLM every N environment steps |
| `vlm_pure_reward` | true | Use only VLM reward (ignore env reward) |
| `vlm_non_call_reward_mode` | hold | Reward on non-VLM steps: hold / zero / env |
| `vlm_sample_envs` | 0 | Number of envs to sample per VLM call (0 = all) |
| `vlm_server_url` | http://localhost:5002 | VLM reward server URL |

## Architecture

```
Host Machine                          Docker Container (RLinf)
┌─────────────────────┐              ┌──────────────────────────┐
│  VLM Reward Server  │              │  vlm_maniskill_env.py    │
│  (vlm_reward_server │◄── HTTP ───► │    └─ vlm_reward_client  │
│   .py + VLM model)  │   :5002      │                          │
│                     │              │  train_embodied_agent.py  │
│  GPU: VLM inference │              │  GPU: Policy training     │
└─────────────────────┘              └──────────────────────────┘
```

The VLM reward server runs on the host with its own GPU for VLM inference. The RL training runs inside the RLinf Docker container. They communicate via HTTP on port 5002 (Docker uses `--network host`).

## File Structure

```
Large-Reward-Models/
├── README.md
├── config/
│   ├── contrastive.yaml          # Training config: contrastive mode
│   ├── completion.yaml           # Training config: completion mode
│   ├── progress.yaml             # Training config: progress mode
│   ├── roboreward.yaml           # Training config: roboreward mode
│   ├── robometer.yaml            # Training config: robometer mode
│   ├── env/
│   │   └── maniskill_put_on_plate_vlm.yaml  # Environment config (shared)
│   ├── model/
│   │   └── pi0_5.yaml            # Policy model config
│   └── training_backend/
│       └── fsdp.yaml             # FSDP training backend config
├── envs/
│   └── vlm_maniskill_env.py      # Unified VLM ManiSkill environment (all 5 modes)
└── vlm_reward/
    ├── start_server.sh           # VLM reward server launcher
    ├── vlm_reward_server.py      # VLM reward server (all 5 endpoints)
    └── vlm_reward_client.py      # VLM reward client (called from env)
```

## Citation

```bibtex
TODO
```
