# Large Reward Models

RL training code for robotic manipulation with VLM-based reward models. Supports 5 reward modes that can be switched via YAML configuration.

Built on [RLinf](https://github.com/RLinf/RLinf) framework with [ManiSkill](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/maniskill.html) environments and [Pi0.5](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html) policy.

## Reward Modes

| Mode | VLM Model | Signal Type | Description |
|---|---|---|---|
| **Contrastive** | LRM-contrastive (Qwen3-VL-8B) | +1 / -1 / 0 | Compares two consecutive frames to determine progress direction |
| **Completion** | LRM-completion | Binary 0 / 1 | Single-frame yes/no task completion judgment |
| **Progress** | LRM-progress | Continuous 0~1 | Single-frame task completion progress estimation |
| **RoboReward** | RoboReward-8B | Discrete 1-5 → 0~1 | Scores a video trajectory on a 1-5 rubric |
| **Robometer** | Robometer-4B | Continuous 0~1 per frame | Per-frame progress via custom reward heads |

## Prerequisites

- [RLinf](https://github.com/RLinf/RLinf) framework (for RL training, inside Docker)
- A separate Python environment for the VLM reward server (on host, outside Docker)
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
| RoboReward | RoboReward-8B | [HuggingFace](https://huggingface.co/teetone/RoboReward-8B) |
| Robometer | Robometer-4B | [HuggingFace](https://huggingface.co/aliangdw/Robometer-4B) |

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

### 2. Set up VLM Reward Server Environment (on host, outside Docker)

```bash
conda create -n vlm_reward python=3.10
conda activate vlm_reward
cd Large-Reward-Models
pip install -r vlm_reward/requirements.txt
```

### 3. Install into RLinf

The install script copies files into RLinf and patches it to register the `vlm_maniskill` environment type:

```bash
git clone https://github.com/physical-superintelligence-lab/Large-Reward-Models.git
cd Large-Reward-Models
bash install.sh /path/to/RLinf
```

### 4. Update model paths

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

Each mode has a dedicated config file under `config/`. To switch modes, simply use a different config file — no manual parameter tuning needed:

```bash
bash run_embodiment.sh contrastive    # uses config/contrastive.yaml
bash run_embodiment.sh completion     # uses config/completion.yaml
bash run_embodiment.sh progress       # uses config/progress.yaml
bash run_embodiment.sh roboreward     # uses config/roboreward.yaml
bash run_embodiment.sh robometer      # uses config/robometer.yaml
```

Remember to start the corresponding VLM reward server with the matching model before training (see [Step 1](#step-1-start-vlm-reward-server-on-host-outside-docker)).

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

## Evaluation

### Closed-Loop Evaluation (Success Rate)

Run the trained policy in ManiSkill and measure success rate:

```bash
# Inside RLinf Docker container
source switch_env openpi

# Edit config/eval/closed_loop_eval.yaml to set ckpt_path and model_path
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_embodiment.sh closed_loop_eval
```

### Open-Loop Evaluation (Reward Quality Metrics)

Evaluate VLM reward quality by scoring recorded trajectories and computing discriminative accuracy, correlation, and temporal consistency metrics.

**Step 1: Collect trajectories**

```bash
# Inside Docker - runs policy and saves frames + oracle rewards
bash run_embodiment.sh open_loop_collect
```

**Step 2: Score with VLM** (on host, with VLM server running)

```bash
cd RLinf/scripts

# Contrastive mode
python score_with_vlm.py --data_dir <trajectory_dir>/worker_0 --mode comparison

# Completion mode
python score_with_vlm.py --data_dir <trajectory_dir>/worker_0 --mode completion

# Progress mode
python score_with_vlm.py --data_dir <trajectory_dir>/worker_0 --mode progress
```

**Step 3: Compute metrics**

```bash
python compute_openloop_metrics.py --scores_path <trajectory_dir>/worker_0/vlm_scores.json
```

Outputs `openloop_metrics.json` with discriminative accuracy (ROC-AUC, pairwise ranking), correlation (Pearson, Spearman, Kendall), and temporal consistency metrics.

## File Structure

```
Large-Reward-Models/
├── README.md
├── install.sh                    # Auto-install into RLinf
├── config/
│   ├── contrastive.yaml          # Training config: contrastive mode
│   ├── completion.yaml           # Training config: completion mode
│   ├── progress.yaml             # Training config: progress mode
│   ├── roboreward.yaml           # Training config: roboreward mode
│   ├── robometer.yaml            # Training config: robometer mode
│   ├── env/
│   │   ├── maniskill_put_on_plate_vlm.yaml        # VLM environment config
│   │   └── maniskill_put_on_plate_recording.yaml   # Recording env config (open-loop eval)
│   ├── eval/
│   │   ├── closed_loop_eval.yaml   # Closed-loop eval config
│   │   └── open_loop_collect.yaml  # Open-loop trajectory collection config
│   ├── model/
│   │   └── pi0_5.yaml            # Policy model config
│   └── training_backend/
│       └── fsdp.yaml             # FSDP training backend config
├── envs/
│   └── vlm_maniskill_env.py      # Unified VLM ManiSkill environment (all 5 modes)
├── eval/
│   ├── score_with_vlm.py         # Score trajectories with VLM (contrastive/completion/progress)
│   └── compute_openloop_metrics.py  # Compute open-loop metrics
└── vlm_reward/
    ├── requirements.txt          # Python dependencies for VLM server
    ├── start_server.sh           # VLM reward server launcher
    ├── vlm_reward_server.py      # VLM reward server (all 5 endpoints)
    └── vlm_reward_client.py      # VLM reward client (called from env)
```

## Citation

```bibtex
@article{wu2026large,
  title={Large Reward Models: Generalizable Online Robot Reward Generation with Vision-Language Models},
  author={Wu, Yanru and Yuan, Weiduo and Qi, Ang and Guizilini, Vitor and Mao, Jiageng and Wang, Yue},
  journal={arXiv preprint arXiv:2603.16065},
  year={2026}
}
```
