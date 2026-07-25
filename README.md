# Large Reward Models

RL training code for robotic manipulation with VLM-based reward models. Supports 7 reward modes that can be switched via YAML configuration.

Built on [RLinf](https://github.com/RLinf/RLinf) framework with [ManiSkill](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/maniskill.html) environments and [Pi0.5](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html) policy.

## Reward Modes

| Mode | VLM Model | Signal Type | Description |
|---|---|---|---|
| **Contrastive** | LRM-contrastive (Qwen3-VL-8B) | +1 / -1 / 0 | Compares two consecutive frames to determine progress direction |
| **Completion** | LRM-completion | Binary 0 / 1 | Single-frame yes/no task completion judgment |
| **Progress** | LRM-progress | Continuous 0~1 | Single-frame task completion progress estimation |
| **Tri** | LRM-contrastive + LRM-progress + LRM-completion | Continuous 0~1 | Equal-weight combination of all three LRM heads on one shared backbone; contrastive's signed [-1,1] score is rescaled to [0,1] before averaging so it does not cancel the other two heads' progress signal |
| **RoboReward** | RoboReward-8B | Discrete 1-5 | Scores a video trajectory on a 1-5 rubric |
| **Robometer** | Robometer-4B | Continuous 0~1 per frame | Per-frame progress via custom reward heads |
| **TOPReward** | Qwen3-VL-8B | Continuous 0~1 | Reads log P("True" \| frames, task) off the base VLM's logits (frames via Qwen-VL's video input) |

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
| Tri | (reuses the three LRM adapters above) | No separate download |
| RoboReward | RoboReward-8B | [HuggingFace](https://huggingface.co/teetone/RoboReward-8B) |
| Robometer | Robometer-4B | [HuggingFace](https://huggingface.co/aliangdw/Robometer-4B) |
| TOPReward | Qwen3-VL-8B-Instruct | No separate download |

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

MODEL_PATH=/path/to/VLMrewardmodel GPU_ID=0 bash start_server.sh

```

For **tri mode**, the server loads all three LRM adapters onto one shared backbone instead of a single `MODEL_PATH`:

```bash
TRI_CONTRASTIVE_ADAPTER=/path/to/LRM-contrastive \
TRI_PROGRESS_ADAPTER=/path/to/LRM-progress \
TRI_COMPLETION_ADAPTER=/path/to/LRM-completion \
BASE_MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct GPU_ID=0 bash start_server.sh
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
bash run_embodiment.sh tri              # tri mode (all three LRM heads combined)
bash run_embodiment.sh roboreward       # roboreward mode
bash run_embodiment.sh robometer        # robometer mode
bash run_embodiment.sh topreward        # topreward mode (zero-shot, base VLM)
```

## Configuration

### Switching Reward Modes

Each mode has a dedicated config file under `config/`. To switch modes, simply use a different config file — no manual parameter tuning needed:

```bash
bash run_embodiment.sh contrastive    # uses config/contrastive.yaml
bash run_embodiment.sh completion     # uses config/completion.yaml
bash run_embodiment.sh progress       # uses config/progress.yaml
bash run_embodiment.sh tri            # uses config/tri.yaml
bash run_embodiment.sh roboreward     # uses config/roboreward.yaml
bash run_embodiment.sh robometer      # uses config/robometer.yaml
bash run_embodiment.sh topreward      # uses config/topreward.yaml
```

Remember to start the corresponding VLM reward server with the matching model before training (see [Step 1](#step-1-start-vlm-reward-server-on-host-outside-docker)).


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

Run the trained policy in ManiSkill and measure success rate.

**A single evaluation run is noisy.** `env.eval.use_fixed_reset_state_ids` keeps
the evaluation episodes identical across runs, but action sampling during
rollout (`do_sample=True`) is not pinned to a fixed outcome, so the success
rate of the SAME checkpoint can vary by several points between runs. Evaluate
each checkpoint across multiple seeds and report the mean and 95% CI rather
than trusting a single run.

```bash
# Inside RLinf Docker container
source switch_env openpi
cd Large-Reward-Models

bash eval/run_closed_loop_seeds.sh \
  --rlinf-dir /workspace/RLinf \
  --results-dir ./results/closed_loop \
  --method lrm_completion \
  --model-path /path/to/SFT_or_checkpoint_dir \
  --ckpt-path /path/to/checkpoint/actor/model_state_dict/full_weights.pt \
  --seeds "0 1 2 3 4"

# Mean, 95% CI, and (if you evaluate more than one method into the same
# --results-dir) paired seed-level comparisons against a reference method
python eval/summarize_closed_loop.py \
  --results-dir ./results/closed_loop \
  --methods lrm_completion
```

Omit `--ckpt-path` to evaluate the SFT baseline directly.

<details>
<summary>Single-run evaluation (quick check only — see the noise warning above)</summary>

```bash
# Edit config/eval/closed_loop_eval.yaml to set ckpt_path and model_path, then:
EMBODIED_PATH=/workspace/RLinf/examples/embodiment ROBOT_PLATFORM=MANISKILL \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=/workspace/RLinf:$PYTHONPATH \
python eval_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config/ \
  --config-name closed_loop_eval \
  runner.logger.log_path=../results
```

</details>

> **Note:** Use `eval_embodied_agent.py` (not `run_embodiment.sh`) for closed-loop eval. The training script spawns an Actor worker for IPC weight sync which causes GPU OOM on 40 GB cards. The eval script loads the checkpoint directly into the rollout worker without an Actor.

### Open-Loop Evaluation (Reward Quality Metrics)

Evaluate VLM reward quality by scoring recorded trajectories and computing metrics on how well the VLM reward correlates with ground-truth oracle rewards.

**Step 1: Collect trajectories** (inside Docker, VLM server not needed)

```bash
bash eval_embodiment.sh open_loop_collect MANISKILL \
  actor.model.model_path=/path/to/checkpoint_or_SFT \
  rollout.model.model_path=/path/to/checkpoint_or_SFT \
  env.eval.record_output_dir=./logs/openloop_data/my_exp
```

For the SFT baseline, pass the SFT model path. For an RL checkpoint, pass the checkpoint directory (which must contain `actor/model_state_dict/full_weights.pt`).

**Step 2: Score with VLM** (on host, with VLM server running)

Run for each worker directory under `record_output_dir`:

```bash
python eval/score_with_vlm.py \
  --data_dir ./logs/openloop_data/my_exp/worker_0 \
  --mode <progress|completion|comparison>
```

**Step 3: Compute metrics**

Pass all workers' `vlm_scores.json` as `--scores_path` to merge results before computing metrics:

```bash
python eval/compute_openloop_metrics.py \
  --scores_path ./logs/openloop_data/my_exp/worker_0/vlm_scores.json \
               ./logs/openloop_data/my_exp/worker_1/vlm_scores.json
```

The script auto-detects the scoring mode and outputs mode-specific metrics:

**Progress / Completion metrics:**

| Metric | Description |
|---|---|
| `roc_auc` | AUC for success vs failure classification |
| `pairwise_acc_pct` | Fraction of success/failure pairs correctly ranked (%) |
| `global_pearson` | Pearson correlation across all trajectories |
| `per_traj_pearson` | Per-trajectory Pearson correlation, averaged |

**Contrastive (comparison) metrics:**

| Metric | Description |
|---|---|
| `direction_acc_pct` | Fraction of VLM direction predictions matching oracle (%) |
| `progress_recall_pct` | When oracle shows progress, how often VLM predicts positive (%) |
| `monotonicity_success` | Temporal consistency of cumulative score in successful episodes |

## File Structure

```
Large-Reward-Models/
├── README.md
├── install.sh                    # Auto-install into RLinf
├── config/
│   ├── contrastive.yaml          # Training config: contrastive mode
│   ├── completion.yaml           # Training config: completion mode
│   ├── progress.yaml             # Training config: progress mode
│   ├── tri.yaml                  # Training config: tri mode (all three LRM heads)
│   ├── roboreward.yaml           # Training config: roboreward mode
│   ├── robometer.yaml            # Training config: robometer mode
│   ├── topreward.yaml            # Training config: topreward mode (zero-shot)
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
│   └── vlm_maniskill_env.py      # Unified VLM ManiSkill environment (all 7 modes)
├── eval/
│   ├── score_with_vlm.py         # Score trajectories with VLM (contrastive/completion/progress)
│   ├── compute_openloop_metrics.py  # Compute open-loop metrics
│   ├── run_closed_loop_seeds.sh  # Repeat closed-loop eval across seeds
│   └── summarize_closed_loop.py  # Mean/95% CI (and paired comparisons) across seeds
└── vlm_reward/
    ├── requirements.txt          # Python dependencies for VLM server
    ├── start_server.sh           # VLM reward server launcher
    ├── vlm_reward_server.py      # VLM reward server (all 7 endpoints)
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
