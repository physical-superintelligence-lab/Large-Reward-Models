#!/usr/bin/env python3
"""
VLM Reward Server - provides VLM reward computation as an HTTP service.

Usage:
    # Start the server inside a conda environment
    conda activate your_env
    python vlm_reward_server.py --port 5001 --model_path /path/to/vlm_checkpoint

The RLinf process (e.g. inside a docker container) obtains VLM rewards via HTTP requests.
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import threading
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

# Global model instance
vlm_model = None
vlm_processor = None
device = None
robometer_progress_head = None
robometer_frame_pool_attn = None
robometer_head_mode = False
robometer_frame_pooling = "mean"
robometer_frame_pooling_attn_temperature = 1.0
robometer_use_per_frame_progress_token = False
robometer_prog_token_id = None
robometer_vision_start_token_id = None
robometer_vision_end_token_id = None
inference_lock = threading.Lock()
vlm_adapter_names = {}


def _activate_adapter(reward_head: str) -> None:
    """Select a LoRA adapter when the server was started in tri-head mode."""
    adapter_name = vlm_adapter_names.get(reward_head)
    if adapter_name is not None:
        vlm_model.set_adapter(adapter_name)


class RobometerProgressHead(torch.nn.Module):
    """Minimal reward head used by Robometer checkpoints."""

    def __init__(self, hidden_size: int, mid_size: int, num_bins: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_size, mid_size, bias=True)
        self.norm = torch.nn.LayerNorm(mid_size)
        self.fc2 = torch.nn.Linear(mid_size, num_bins, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = F.silu(x)
        return self.fc2(x)


def _load_robometer_heads(model_path: str, compute_device: str) -> tuple[Optional[RobometerProgressHead], Optional[torch.Tensor]]:
    """Load Robometer reward heads from safetensors shards if present."""
    try:
        from safetensors import safe_open
    except Exception as e:
        print(f"Robometer head load skipped: safetensors unavailable: {e}")
        return None, None

    required_keys = {
        "progress_head.0.weight",
        "progress_head.0.bias",
        "progress_head.1.weight",
        "progress_head.1.bias",
        "progress_head.4.weight",
        "progress_head.4.bias",
        "frame_pool_attn.weight",
    }
    loaded = {}
    shard_paths = sorted(
        os.path.join(model_path, name)
        for name in os.listdir(model_path)
        if name.endswith(".safetensors")
    )
    for shard_path in shard_paths:
        if required_keys.issubset(loaded.keys()):
            break
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            shard_keys = set(sf.keys())
            for key in required_keys - set(loaded.keys()):
                if key in shard_keys:
                    loaded[key] = sf.get_tensor(key)

    if not required_keys.issubset(loaded.keys()):
        missing = sorted(required_keys - set(loaded.keys()))
        print(f"Robometer head load skipped: missing keys: {missing}")
        return None, None

    hidden_size = int(loaded["progress_head.0.weight"].shape[1])
    mid_size = int(loaded["progress_head.0.weight"].shape[0])
    num_bins = int(loaded["progress_head.4.weight"].shape[0])
    head = RobometerProgressHead(hidden_size, mid_size, num_bins)
    head.fc1.weight.data.copy_(loaded["progress_head.0.weight"])
    head.fc1.bias.data.copy_(loaded["progress_head.0.bias"])
    head.norm.weight.data.copy_(loaded["progress_head.1.weight"])
    head.norm.bias.data.copy_(loaded["progress_head.1.bias"])
    head.fc2.weight.data.copy_(loaded["progress_head.4.weight"])
    head.fc2.bias.data.copy_(loaded["progress_head.4.bias"])

    dtype = torch.bfloat16 if "cuda" in str(compute_device) else torch.float32
    head = head.to(device=compute_device, dtype=dtype).eval()
    frame_pool = loaded["frame_pool_attn.weight"].to(device=compute_device, dtype=dtype)
    return head, frame_pool


def _load_robometer_runtime_config(model_path: str) -> None:
    """Load Robometer runtime options from config.yaml when available."""
    global robometer_frame_pooling
    global robometer_frame_pooling_attn_temperature
    global robometer_use_per_frame_progress_token

    cfg_path = os.path.join(model_path, "config.yaml")
    if not os.path.exists(cfg_path):
        return
    try:
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        model_cfg = cfg.get("model", {}) or {}
        data_cfg = cfg.get("data", {}) or {}

        robometer_frame_pooling = str(model_cfg.get("frame_pooling", "mean")).lower()
        robometer_frame_pooling_attn_temperature = float(
            model_cfg.get("frame_pooling_attn_temperature", 1.0)
        )
        robometer_use_per_frame_progress_token = bool(
            data_cfg.get(
                "use_per_frame_progress_token",
                model_cfg.get("use_per_frame_progress_token", False),
            )
        )
    except Exception as e:
        print(f"Warning: failed to parse Robometer config.yaml: {e}")
        robometer_frame_pooling = "mean"
        robometer_frame_pooling_attn_temperature = 1.0
        robometer_use_per_frame_progress_token = False


def load_vlm_model(
    model_path: str,
    base_model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
    gpu_id: int = 0,
    processor_path_override: str = None,
):
    """Load VLM reward model (supports LoRA adapter via manual merge)."""
    global vlm_model, vlm_processor, device
    global robometer_progress_head, robometer_frame_pool_attn, robometer_head_mode
    global robometer_prog_token_id, robometer_vision_start_token_id, robometer_vision_end_token_id
    
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from safetensors.torch import load_file
    
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Loading VLM model on {device}...")
    
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path)
    
    if is_lora:
        print(f"Detected LoRA adapter at {model_path}")
        print(f"Loading base model from {base_model_path}...")
        
        vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        
        print(f"Loading LoRA adapter from {model_path}...")
        adapter_path = os.path.join(model_path, "adapter_model.safetensors")
        adapter_weights = load_file(adapter_path)
        
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        lora_alpha = adapter_config.get('lora_alpha', 128)
        r = adapter_config.get('r', 64)
        scaling = lora_alpha / r
        print(f"LoRA config: alpha={lora_alpha}, r={r}, scaling={scaling:.2f}")
        
        # Separate LoRA weight pairs from full-rank module weights
        lora_pairs = {}
        full_modules = {}
        
        for key, value in adapter_weights.items():
            if 'lora_A' in key:
                base_key = key.replace('.lora_A.weight', '').replace('.lora_A.default.weight', '').replace('base_model.model.', '')
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]['A'] = value
            elif 'lora_B' in key:
                base_key = key.replace('.lora_B.weight', '').replace('.lora_B.default.weight', '').replace('base_model.model.', '')
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]['B'] = value
            elif 'lora' not in key.lower():
                clean_key = key.replace('base_model.model.', '')
                full_modules[clean_key] = value
        
        if full_modules:
            print(f"Loading {len(full_modules)} full module parameters...")
            vlm_model.load_state_dict(full_modules, strict=False)
        
        print(f"Merging {len(lora_pairs)} LoRA adapters...")
        model_state_dict = vlm_model.state_dict()
        merged_count = 0
        
        for base_key, lora_weights in lora_pairs.items():
            if 'A' in lora_weights and 'B' in lora_weights:
                weight_key = base_key + '.weight' if not base_key.endswith('.weight') else base_key
                if weight_key in model_state_dict:
                    base_weight = model_state_dict[weight_key]
                    lora_A = lora_weights['A'].to(base_weight.device, base_weight.dtype)
                    lora_B = lora_weights['B'].to(base_weight.device, base_weight.dtype)
                    delta = (lora_B @ lora_A) * scaling
                    model_state_dict[weight_key] = base_weight + delta
                    merged_count += 1
        
        vlm_model.load_state_dict(model_state_dict, strict=False)
        print(f"✅ Successfully merged {merged_count} LoRA adapters")
    else:
        print(f"Loading full model from {model_path}...")
        vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
    
    vlm_model.eval()
    
    processor_path = (
        processor_path_override
        if processor_path_override
        else (model_path if not is_lora else base_model_path)
    )
    try:
        vlm_processor = AutoProcessor.from_pretrained(
            processor_path, trust_remote_code=True
        )
    except OSError as e:
        # Some fine-tuned checkpoints do not ship preprocessor/tokenizer files.
        # Fall back to base model processor.
        print(
            f"Processor load failed from {processor_path}: {e}\n"
            f"Falling back to base model processor: {base_model_path}"
        )
        vlm_processor = AutoProcessor.from_pretrained(
            base_model_path, trust_remote_code=True
        )

    # Use Robometer reward heads when present (official-style head inference path).
    _load_robometer_runtime_config(model_path)
    robometer_progress_head, robometer_frame_pool_attn = _load_robometer_heads(
        model_path, device
    )
    robometer_head_mode = (
        robometer_progress_head is not None and robometer_frame_pool_attn is not None
    )
    if robometer_head_mode:
        tok = getattr(vlm_processor, "tokenizer", None)
        if tok is not None:
            robometer_prog_token_id = tok.convert_tokens_to_ids("<|prog_token|>")
            robometer_vision_start_token_id = tok.convert_tokens_to_ids("<|vision_start|>")
            robometer_vision_end_token_id = tok.convert_tokens_to_ids("<|vision_end|>")
        if robometer_prog_token_id in (None, tok.unk_token_id if tok is not None else -1):
            robometer_prog_token_id = None
        print("Robometer reward-head mode enabled (non-generative progress inference).")
        print(
            f"Robometer runtime cfg: frame_pooling={robometer_frame_pooling}, "
            f"use_per_frame_progress_token={robometer_use_per_frame_progress_token}, "
            f"has_prog_token={robometer_prog_token_id is not None}"
        )
    else:
        print("Robometer reward-head mode disabled; fallback to text-generation parsing.")
    
    print("VLM model loaded successfully!")
    return vlm_model, vlm_processor


def load_vlm_multi_adapter(
    base_model_path: str,
    contrastive_adapter_path: str,
    progress_adapter_path: str,
    completion_adapter_path: str,
    gpu_id: int = 0,
):
    """Load one Qwen base and three switchable LoRA adapters for tri-head inference."""
    global vlm_model, vlm_processor, device, vlm_adapter_names

    from peft import PeftConfig, PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Loading tri-head base model from {base_model_path} on {device}...")
    # Load on CPU and move to the GPU only after every adapter is attached.
    # Passing device_map here makes PEFT silently skip loading the LoRA weights:
    # from_pretrained returns with lora_B still at its zero init, no warning and
    # no unexpected_keys. Since LoRA is B @ A, a zeroed B is an exact identity,
    # so all three heads would serve the unmodified base model while /health
    # cheerfully reports the adapters as loaded.
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    adapter_paths = {
        "contrastive": contrastive_adapter_path,
        "progress": progress_adapter_path,
        "completion": completion_adapter_path,
    }
    # Progress contains modules_to_save entries for the visual tower/merger.
    # It must be the first adapter so PEFT constructs those wrappers before
    # adding the two language-only adapters.
    first_name = "progress"
    progress_config = PeftConfig.from_pretrained(adapter_paths[first_name])
    # The published progress adapter lists both `visual` and its nested
    # `merger` module. PEFT 0.18 cannot construct overlapping
    # ModulesToSaveWrapper objects, while wrapping `visual` already includes
    # every merger weight in the checkpoint.
    progress_config.modules_to_save = ["visual"]
    vlm_model = PeftModel.from_pretrained(
        base_model,
        adapter_paths[first_name],
        adapter_name=first_name,
        is_trainable=False,
        config=progress_config,
    )
    for name in ("contrastive", "completion"):
        print(f"Loading tri-head {name} adapter from {adapter_paths[name]}...")
        vlm_model.load_adapter(adapter_paths[name], adapter_name=name, is_trainable=False)
    vlm_model = vlm_model.to(device)
    vlm_model.eval()

    # Fail loudly rather than silently serving the base model: a zeroed lora_B
    # is indistinguishable from the base at inference time.
    for name in adapter_paths:
        vlm_model.set_adapter(name)
        if not any(
            param.abs().any().item()
            for key, param in vlm_model.named_parameters()
            if f"lora_B.{name}.weight" in key
        ):
            raise RuntimeError(
                f"Adapter '{name}' loaded with an all-zero lora_B: its weights were "
                f"not applied, so this head would silently serve the base model."
            )
    # Adapter exports include the processor files even when a local cached base
    # snapshot contains only model/tokenizer shards.
    vlm_processor = AutoProcessor.from_pretrained(
        progress_adapter_path, trust_remote_code=True
    )
    vlm_adapter_names = {name: name for name in adapter_paths}
    _activate_adapter(first_name)
    print(f"Tri-head VLM loaded with adapters: {sorted(vlm_adapter_names)}")
    return vlm_model, vlm_processor


# Robometer-style video prompt adapted to single-rollout RL training.
ROBOMETER_VIDEO_PROMPT = (
    'Given this trajectory for the task "{task}", '
    "predict task progress for each frame as a "
    "float between 0 and 1, where 0 is the initial state and 1 is completed.\n"
    "If the robot is not performing the specified task, assign 0 progress.\n"
    "Output strict JSON in the following format:\n"
    '{{"progress_per_frame":[p1,p2,...],"final_progress":p_last}}'
)


# TOPReward (github.com/TOPReward/TOPReward). Reproduced verbatim from
# topreward/clients/qwen.py:compute_instruction_reward with the repository's
# default options (use_video_description=false, add_chat_template=false,
# reduction=mean, fps=2.0). Despite the paper's phrasing, the implementation
# scores a single token: everything but the final "True" is masked out, so the
# reward is log P("True" | video, task) -- how confident the VLM is that the
# trajectory completes the instruction.
TOPREWARD_PREFIX = (
    "The above video shows a robot manipulation trajectory that completes the "
    "following task: "
)
TOPREWARD_SUFFIX = (
    " Decide whether the above statement is True or not. The answer is: True"
)


# Affine map from log P("True") to [0, 1]. TOPReward's own normalisation is
# min-max across a trajectory's prefixes, which is retrospective and unavailable
# online, and exp() is useless here: measured log-probabilities run about -26 to
# -7, so exp() collapses every score to ~0. A fixed monotone rescale keeps the
# ordering the method actually provides (its reported metric, VOC, is a rank
# correlation) while putting the reward on the same [0, 1] scale as every other
# arm. Bounds are round numbers bracketing the observed range over SFT
# trajectories (min -26.1, p5 -25.1, p95 -7.7, max -6.8).
TOPREWARD_LOGP_LO = -25.0
TOPREWARD_LOGP_HI = -5.0


def compute_vlm_topreward(
    frames: list,
    task_description: str,
    fps: float = 2.0,
) -> dict:
    """Score a frame sequence with TOPReward: log P("True" | video, instruction).

    Returns the raw log-probability plus a monotone rescaling of it into [0, 1]
    (see TOPREWARD_LOGP_LO/HI) for use as an RL reward.
    """
    from qwen_vl_utils import process_vision_info

    global vlm_model, vlm_processor, device

    if vlm_model is None:
        raise RuntimeError("VLM model not loaded!")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "fps": fps},
                {"type": "text", "text": TOPREWARD_PREFIX},
            ],
        }
    ]

    prompt_chat = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    eos_token = getattr(vlm_processor.tokenizer, "eos_token", None)
    if eos_token is not None:
        prompt_chat = prompt_chat.split(eos_token)[0]
    full_text = f"{prompt_chat}{task_description}{TOPREWARD_SUFFIX}"

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = vlm_processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(device)

    # Mask everything but the last token, so only "True" is scored.
    labels = inputs["input_ids"].clone()
    prompt_length = inputs["input_ids"].shape[1] - 1
    labels[:, :prompt_length] = -100
    if "attention_mask" in inputs:
        labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

    with torch.no_grad():
        outputs = vlm_model(**inputs)

    logits = outputs.logits[:, :-1, :]
    target_labels = labels[:, 1:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    mask = target_labels != -100
    safe_targets = target_labels.masked_fill(~mask, 0)
    token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    masked_log_probs = token_log_probs[mask]

    if masked_log_probs.numel() == 0:
        return {"score": 0.0, "log_prob": None, "token_count": 0, "response": ""}

    log_prob = masked_log_probs.mean().item()
    score = (log_prob - TOPREWARD_LOGP_LO) / (TOPREWARD_LOGP_HI - TOPREWARD_LOGP_LO)
    score = float(min(1.0, max(0.0, score)))
    return {
        "score": score,
        "log_prob": log_prob,
        "token_count": int(masked_log_probs.numel()),
        "response": f"logP(True)={log_prob:.4f}",
    }


ROBOREWARD_PROMPT = (
    "Given the task, assign a discrete progress score reward (1,2,3,4,5) "
    "for the robot in the video in the format: ANSWER: <score>\n"
    "Rubric for end-of-episode progress (judge only the final state without time limits):\n"
    "1 - No Success: Final state shows no goal-relevant change for the command.\n"
    "2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.\n"
    "3 - Partial Completion: The final state shows good progress toward the goal but violates "
    "more than one requirement or a major requirement.\n"
    "4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.\n"
    "5 - Perfect Completion: Final state satisfies all requirements.\n\n"
    "Task: {task}"
)


def extract_roboreward_score(response: str) -> tuple:
    """Extract 1-5 score from RoboReward output, return (raw_score, normalized_score)."""
    m = re.search(r"ANSWER:\s*([1-5])", response)
    if m:
        score = int(m.group(1))
        return score, (score - 1) / 4.0
    m = re.search(r"\b([1-5])\b", response)
    if m:
        score = int(m.group(1))
        return score, (score - 1) / 4.0
    return None, 0.0


def compute_vlm_roboreward(
    frames: list,
    task_description: str,
) -> dict:
    """Score a trajectory via RoboReward-style video prompt (1-5 discrete, normalized to 0-1).

    Args:
        frames: PIL Image list (chronologically ordered)
        task_description: Task description

    Returns:
        dict: {"score": float (0-1), "raw_score": int (1-5), "response": str}
    """
    from qwen_vl_utils import process_vision_info

    global vlm_model, vlm_processor, device

    if vlm_model is None:
        raise RuntimeError("VLM model not loaded!")

    prompt = ROBOREWARD_PROMPT.format(task=task_description)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "fps": 1.0,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    inputs = vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        **video_kwargs,
    ).to(device)

    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = vlm_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    raw_score, normalized = extract_roboreward_score(response)

    if os.environ.get("VLM_DEBUG", "0") == "1":
        print(f"[RoboReward Debug] Task: {task_description[:50]}...")
        print(f"[RoboReward Debug] Response: {response[:200]}...")
        print(f"[RoboReward Debug] Raw score: {raw_score}, Normalized: {normalized}")

    return {"score": normalized, "raw_score": raw_score, "response": response}


def extract_comparison_result(response: str) -> Optional[str]:
    """Extract comparison result from model output (ImageA or ImageB)."""
    try:
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            data = json.loads(json_match.group(0))
            if "more_complete_image" in data:
                val = data["more_complete_image"]
                if val in ("ImageA", "ImageB"):
                    return val
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    m = re.search(r'"more_complete_image"\s*:\s*"(Image[AB])"', response)
    if m:
        return m.group(1)

    last_a = response.rfind("ImageA")
    last_b = response.rfind("ImageB")
    if last_b > last_a:
        return "ImageB"
    if last_a > last_b:
        return "ImageA"

    return None


def compute_vlm_comparison(
    image_a: np.ndarray,
    image_b: np.ndarray,
    task_description: str,
) -> dict:
    """Compare two images and determine which is closer to task completion.

    Args:
        image_a: First observation image (earlier frame)
        image_b: Second observation image (later frame)
        task_description: Task description

    Returns:
        dict: {"result": "ImageA"/"ImageB"/None, "response": str, "score": float}
              score: +1.0 = ImageB closer, -1.0 = ImageA closer, 0.0 = uncertain
    """
    global vlm_model, vlm_processor, device

    if vlm_model is None:
        raise RuntimeError("VLM model not loaded!")
    _activate_adapter("contrastive")

    def to_pil(img):
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            return Image.fromarray(img)
        return img

    pil_a = to_pil(image_a)
    pil_b = to_pil(image_b)
    images_list = [pil_a, pil_b]

    prompt = (
        "Task: Compare the completion progress.\n\n"
        f"The task is: {task_description}.\n\n"
        "You are given:\n"
        "- Later observation (Image A): <image>\n"
        "- Later observation (Image B): <image>\n\n"
        "Question: Which of Image A or Image B is closer to completing the task?\n"
        "Select one value from the following list:\n"
        '["ImageA", "ImageB"]\n\n'
        "Please provide a step-by-step visual analysis first, and then output your answer in the following JSON format:\n"
        '{ "more_complete_image": "selected_value" }'
    )

    content = []
    prompt_segments = prompt.split("<image>")
    for i, segment in enumerate(prompt_segments):
        if segment.strip():
            content.append({"type": "text", "text": segment})
        if i < len(images_list):
            content.append({"type": "image", "image": images_list[i]})

    messages = [{"role": "user", "content": content}]

    text = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = vlm_processor(
        text=[text],
        images=images_list,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = vlm_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    result = extract_comparison_result(response)

    if result == "ImageB":
        score = 1.0
    elif result == "ImageA":
        score = -1.0
    else:
        score = 0.0

    if os.environ.get("VLM_DEBUG", "0") == "1":
        print(f"[VLM Debug] Comparison - Task: {task_description[:50]}...")
        print(f"[VLM Debug] Response: {response[:200]}...")
        print(f"[VLM Debug] Result: {result}, Score: {score}")

    return {
        "result": result,
        "response": response,
        "score": score,
    }


def extract_robometer_score(response: str) -> tuple[Optional[float], float]:
    """Extract scalar progress in [0,1] from Robometer-style JSON or fallbacks."""
    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    # Prefer JSON output with per-frame progress / final_progress.
    for m in re.finditer(r"\{[^{}]*\}", response, re.DOTALL):
        try:
            data = json.loads(m.group(0))
        except Exception:
            continue

        final = data.get("final_progress", None)
        if final is not None:
            try:
                score = _clamp01(float(final))
                return final, score
            except Exception:
                pass

        progress_list = data.get("progress_per_frame", None)
        if isinstance(progress_list, list) and progress_list:
            vals = []
            for v in progress_list:
                try:
                    vals.append(_clamp01(float(v)))
                except Exception:
                    continue
            if vals:
                return vals[-1], vals[-1]

    # Fallback: grab last numeric token in [0,1].
    float_tokens = re.findall(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", response)
    if float_tokens:
        score = _clamp01(float(float_tokens[-1]))
        return score, score

    # Backward compatibility fallback: 1-5 discrete score.
    m = re.search(r"ANSWER:\s*([1-5])", response)
    if not m:
        m = re.search(r"\b([1-5])\b", response)
    if m:
        raw_score = int(m.group(1))
        return float(raw_score), (raw_score - 1) / 4.0

    return None, 0.0


def _extract_frame_embeddings_from_vision_pairs(
    hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Extract per-frame embeddings from <|vision_start|> ... <|vision_end|> token spans."""
    global robometer_frame_pooling, robometer_frame_pooling_attn_temperature
    global robometer_frame_pool_attn, robometer_vision_start_token_id, robometer_vision_end_token_id

    if robometer_vision_start_token_id is None or robometer_vision_end_token_id is None:
        return torch.empty(0, hidden_state.shape[-1], device=hidden_state.device, dtype=hidden_state.dtype)

    start_positions = (input_ids == robometer_vision_start_token_id).nonzero(as_tuple=True)[0]
    end_positions = (input_ids == robometer_vision_end_token_id).nonzero(as_tuple=True)[0]
    if len(start_positions) == 0 or len(end_positions) == 0:
        return torch.empty(0, hidden_state.shape[-1], device=hidden_state.device, dtype=hidden_state.dtype)

    pair_count = min(len(start_positions), len(end_positions))
    frame_embeddings = []
    for i in range(pair_count):
        start_pos = int(start_positions[i].item())
        end_pos = int(end_positions[i].item())
        if start_pos >= end_pos:
            continue
        frame_tokens = hidden_state[start_pos + 1 : end_pos]
        if frame_tokens.shape[0] == 0:
            continue

        if robometer_frame_pooling == "boundary":
            frame_embedding = frame_tokens[-1]
        elif robometer_frame_pooling == "attention" and robometer_frame_pool_attn is not None:
            temp = max(1e-6, float(robometer_frame_pooling_attn_temperature))
            scores = torch.matmul(frame_tokens, robometer_frame_pool_attn.t()).squeeze(-1) / temp
            weights = torch.softmax(scores, dim=0).unsqueeze(-1)
            frame_embedding = (weights * frame_tokens).sum(dim=0)
        else:
            # Default and official config fallback.
            frame_embedding = frame_tokens.mean(dim=0)
        frame_embeddings.append(frame_embedding)

    if not frame_embeddings:
        return torch.empty(0, hidden_state.shape[-1], device=hidden_state.device, dtype=hidden_state.dtype)
    return torch.stack(frame_embeddings, dim=0)


def _preprocess_robometer_frames(frames: list[Image.Image]) -> list[Image.Image]:
    """Subsample and resize frames to keep Robometer inference memory-bounded."""
    if not frames:
        return frames

    max_frames = int(os.environ.get("ROBOMETER_MAX_FRAMES", "8"))
    max_side = int(os.environ.get("ROBOMETER_MAX_SIDE", "320"))
    max_frames = max(1, max_frames)
    max_side = max(64, max_side)

    if len(frames) > max_frames:
        keep = np.linspace(0, len(frames) - 1, max_frames, dtype=int).tolist()
        frames = [frames[i] for i in keep]

    resized = []
    for frm in frames:
        if not isinstance(frm, Image.Image):
            frm = Image.fromarray(np.asarray(frm))
        frm = frm.convert("RGB")
        w, h = frm.size
        scale = min(1.0, float(max_side) / float(max(w, h)))
        if scale < 1.0:
            frm = frm.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
        resized.append(frm)
    return resized


def compute_vlm_robometer(
    frames: list[Image.Image],
    task_description: str,
) -> dict:
    """Score a frame sequence via Robometer-style video prompt (progress in [0,1])."""
    global vlm_model, vlm_processor, device
    global robometer_progress_head, robometer_frame_pool_attn, robometer_head_mode
    global robometer_use_per_frame_progress_token, robometer_prog_token_id

    if vlm_model is None:
        raise RuntimeError("VLM model not loaded!")
    if not frames:
        return {"score": 0.0, "raw_score": None, "response": "", "progress_per_frame": []}
    frames = _preprocess_robometer_frames(frames)

    prompt = ROBOMETER_VIDEO_PROMPT.format(task=task_description)

    # Official Robometer collator style prompt (progress-only).
    official_prompt = (
        f"The task for the robot is '{task_description}'. Given the trajectory video, "
        "predict the task progress at each frame, how far along the robot is towards "
        "completing the task, a float between 0 and 1, where 0 is the starting state "
        "and 1 is when the task is completed. If the robot is not performing the same task, "
        "predict 0 progress."
    )

    if robometer_head_mode:
        # Match Robometer multi-image inference path as closely as possible.
        content = [{"type": "text", "text": official_prompt}]
        for frm in frames:
            content.append({"type": "image", "image": frm})
            if robometer_use_per_frame_progress_token and robometer_prog_token_id is not None:
                content.append({"type": "text", "text": "<|prog_token|>"})
        messages = [{"role": "user", "content": content}]
        try:
            text = vlm_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                add_vision_id=True,
                enable_thinking=False,
                fps=1,
            )
        except TypeError:
            text = vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        inputs = vlm_processor(
            text=[text],
            images=frames,
            padding=True,
            return_tensors="pt",
        ).to(device)
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "fps": 1.0},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Prefer qwen_vl_utils video packing when available; fallback to image-sequence packing.
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except ImportError:
            process_vision_info = None

        if process_vision_info is not None:
            text = vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            inputs = vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                **video_kwargs,
            ).to(device)
        else:
            fallback_content = [{"type": "text", "text": prompt}]
            for frm in frames:
                fallback_content.append({"type": "image", "image": frm})
            fallback_messages = [{"role": "user", "content": fallback_content}]
            text = vlm_processor.apply_chat_template(
                fallback_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = vlm_processor(
                text=[text],
                images=frames,
                padding=True,
                return_tensors="pt",
            ).to(device)

    if robometer_head_mode:
        try:
            model_inputs = dict(inputs)
            with torch.no_grad():
                outputs = vlm_model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
            if not getattr(outputs, "hidden_states", None):
                raise RuntimeError("No hidden_states returned by base model.")

            hidden = outputs.hidden_states[-1][0]  # [L, H]
            input_ids = model_inputs["input_ids"][0]

            frame_hidden_states = None
            # Preferred path for official Robometer config when prog token exists.
            if robometer_use_per_frame_progress_token and robometer_prog_token_id is not None:
                prog_pos = (input_ids == robometer_prog_token_id).nonzero(as_tuple=True)[0]
                if len(prog_pos) > 0:
                    frame_hidden_states = hidden[prog_pos]

            # Fallback path based on vision token spans.
            if frame_hidden_states is None or frame_hidden_states.shape[0] == 0:
                frame_hidden_states = _extract_frame_embeddings_from_vision_pairs(
                    hidden, input_ids
                )

            # Last fallback to global pooled score to avoid hard failures.
            if frame_hidden_states is None or frame_hidden_states.shape[0] == 0:
                pooled = hidden.mean(dim=0, keepdim=True)
                frame_hidden_states = pooled

            progress_logits = robometer_progress_head(frame_hidden_states)  # [T, bins]
            probs = torch.softmax(progress_logits.float(), dim=-1)
            bin_values = torch.linspace(0.0, 1.0, progress_logits.shape[-1], device=probs.device)
            progress_per_frame = (probs * bin_values).sum(dim=-1)

            score = float(progress_per_frame[-1].item())
            pred_bin = int(torch.argmax(probs[-1], dim=-1).item())
            raw_score = float(pred_bin) / max(1, progress_logits.shape[-1] - 1)
            progress_list = [float(x) for x in progress_per_frame.tolist()]

            response = json.dumps(
                {
                    "progress_per_frame": [round(float(x), 4) for x in progress_list],
                    "final_progress": round(score, 4),
                    "mode": "robometer_head",
                },
                ensure_ascii=False,
            )
        except torch.OutOfMemoryError as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            response = json.dumps(
                {
                    "progress_per_frame": [0.0],
                    "final_progress": 0.0,
                    "mode": "robometer_head_oom",
                    "error": str(e).split("\n")[0][:200],
                },
                ensure_ascii=False,
            )
            raw_score, score, progress_list = None, 0.0, [0.0]
        except Exception as e:
            print(f"[VLM] Robometer head inference failed: {e}")
            response = json.dumps(
                {
                    "progress_per_frame": [0.0],
                    "final_progress": 0.0,
                    "mode": "robometer_head_error",
                    "error": str(e)[:200],
                },
                ensure_ascii=False,
            )
            raw_score, score, progress_list = None, 0.0, [0.0]
    else:
        with torch.no_grad():
            generated_ids = vlm_model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        raw_score, score = extract_robometer_score(response)
        progress_list = [score]

    if os.environ.get("VLM_DEBUG", "0") == "1":
        print(
            f"[VLM Robometer] frames={len(frames)} raw_score={raw_score} "
            f"score={score:.4f} response={response[:200]}"
        )

    return {
        "score": score,
        "raw_score": raw_score,
        "response": response,
        "progress_per_frame": progress_list,
    }


def extract_progress_score(response: str) -> float:
    """Extract progress score from model output.

    Tries JSON format {"completion_progress": value} first, then falls back to
    regex-based numeric matching.
    """
    import json
    
    try:
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            if "completion_progress" in data:
                val = data["completion_progress"]
                if val == "NA" or val == "na":
                    return 0.0
                return max(0.0, min(1.0, float(val)))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # Fallback: match "completion_progress": value pattern without full JSON parse
    try:
        m = re.search(r'"completion_progress"\s*:\s*"?(NA|([0-9.]+))"?', response, re.IGNORECASE)
        if m:
            captured = m.group(1)
            if captured.upper() == "NA":
                return 0.0
            return max(0.0, min(1.0, float(captured)))
    except (ValueError, TypeError):
        pass
    
    # Fallback: simple numeric pattern matching
    patterns = [
        r"\b(0(\.[0-9]+)?|1(\.0)?)\b",
        r"Progress[:\s]*([0-9]*\.?[0-9]+)",
        r"progress[:\s]*([0-9]*\.?[0-9]+)",
        r"Score[:\s]*([0-9]*\.?[0-9]+)",
        r"score[:\s]*([0-9]*\.?[0-9]+)",
        r"([0-9]*\.?[0-9]+)\s*$",  # trailing number
        r"^([0-9]*\.?[0-9]+)",  # leading number
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                continue
    
    return 0.0


def compute_vlm_reward(
    image: np.ndarray,
    task_description: str,
    reward_type: str = "progress",
    goal_image: np.ndarray = None,
    initial_image: np.ndarray = None
) -> dict:
    """Compute VLM reward for a single image.

    Args:
        image: Current observation image (required)
        task_description: Description of the task
        reward_type: Type of reward ("progress" or "quality")
        goal_image: Goal/final state image (optional but recommended)
        initial_image: Initial state image (optional)
    """
    global vlm_model, vlm_processor, device
    
    if vlm_model is None:
        raise RuntimeError("VLM model not loaded!")
    if reward_type == "progress":
        _activate_adapter("progress")
    
    def to_pil(img):
        """Convert numpy array to PIL Image"""
        if img is None:
            return None
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            return Image.fromarray(img)
        return img
    
    current_pil = to_pil(image)
    goal_pil = to_pil(goal_image)
    initial_pil = to_pil(initial_image)
    
    images_list = []

    if reward_type == "progress":
        prompt_parts = []
        prompt_parts.append("Task: Estimate the completion progress.")
        prompt_parts.append(f"\nThe task is: {task_description}")
        prompt_parts.append("\nYou are given:")
        
        if goal_pil is not None:
            prompt_parts.append("\n- Goal Standard: <image>")
            images_list.append(goal_pil)
            
            if initial_pil is not None:
                prompt_parts.append("\n- Initial observation: <image>")
                images_list.append(initial_pil)
            
            prompt_parts.append("\n- Current observation: <image>")
            images_list.append(current_pil)
            
            prompt_parts.append("""

Question: Compare the Current Observation with the Goal Standard. Estimate the completion progress as a value between 0.0 and 1.0.
Select one value from the following list:
[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

Definitions:
- 0.0: Task has just started.
- 0.1 - 0.9: Intermediate progress steps (assess visual distance to Goal).
- 1.0: Task is Finished.
Output your answer in the following JSON format:
{ "completion_progress": selected_value }""")
        else:
            if initial_pil is not None:
                prompt_parts.append("\n- Initial observation: <image>")
                images_list.append(initial_pil)
            
            prompt_parts.append("\n- Current observation: <image>")
            images_list.append(current_pil)
            
            prompt_parts.append("""

Question: Based on the task description, estimate the completion progress as a value between 0.0 and 1.0.
Select one value from the following list:
[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

Definitions:
- 0.0: Task has just started.
- 0.1 - 0.9: Intermediate progress steps (assess how close to completion).
- 1.0: Task is Finished.
Output your answer in the following JSON format:
{ "completion_progress": selected_value }""")
        
        prompt = "".join(prompt_parts)
    else:
        images_list.append(current_pil)
        prompt = f"""Task: {task_description}

Evaluate how well the robot is performing this task.
Output a score between 0.0 (poor) and 1.0 (excellent)."""
    
    # Replace <image> placeholders with actual image entries
    content = []
    prompt_segments = prompt.split("<image>")
    for i, segment in enumerate(prompt_segments):
        if segment.strip():
            content.append({"type": "text", "text": segment})
        if i < len(images_list):
            content.append({"type": "image", "image": images_list[i]})
    
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    
    text = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = vlm_processor(
        text=[text],
        images=images_list,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = vlm_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    score = extract_progress_score(response)

    import os
    if os.environ.get("VLM_DEBUG", "0") == "1":
        print(f"[VLM Debug] Task: {task_description[:50]}...")
        print(f"[VLM Debug] Has goal image: {goal_image is not None}")
        print(f"[VLM Debug] Response: {response[:200]}...")
        print(f"[VLM Debug] Score: {score}")
    
    return {
        "score": score,
        "response": response,
    }


def compute_vlm_rewards_batch(
    images: list,
    task_descriptions: list,
    reward_type: str = "progress"
) -> list:
    """Compute VLM rewards for a batch of images."""
    results = []
    for img, task in zip(images, task_descriptions):
        result = compute_vlm_reward(img, task, reward_type)
        results.append(result)
    return results


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": vlm_model is not None,
        "adapters": sorted(vlm_adapter_names),
    })


@app.route('/compute_reward', methods=['POST'])
def compute_reward_endpoint():
    """Compute VLM reward for a single image."""
    try:
        data = request.json

        def decode_image(img_data, fallback_shape=None, fallback_dtype='uint8'):
            """Decode image from either a dict or flat base64 payload."""
            if isinstance(img_data, dict):
                image_bytes = base64.b64decode(img_data['data'])
                shape = tuple(img_data['shape'])
                dtype = img_data.get('dtype', 'uint8')
            else:
                image_bytes = base64.b64decode(img_data)
                shape = tuple(fallback_shape)
                dtype = fallback_dtype
            return np.frombuffer(image_bytes, dtype=dtype).reshape(shape)
        
        if isinstance(data['image'], dict):
            image = decode_image(data['image'])
            default_shape = data['image']['shape']
            default_dtype = data['image'].get('dtype', 'uint8')
        else:
            image_shape = data['image_shape']
            image_dtype = data.get('image_dtype', 'uint8')
            image = decode_image(data['image'], image_shape, image_dtype)
            default_shape = image_shape
            default_dtype = image_dtype
        
        task_description = data['task_description']
        reward_type = data.get('reward_type', 'progress')
        
        goal_image = None
        if 'goal_image' in data and data['goal_image']:
            if isinstance(data['goal_image'], dict):
                goal_image = decode_image(data['goal_image'])
            else:
                goal_shape = data.get('goal_image_shape', default_shape)
                goal_dtype = data.get('goal_image_dtype', default_dtype)
                goal_image = decode_image(data['goal_image'], goal_shape, goal_dtype)
        
        initial_image = None
        if 'initial_image' in data and data['initial_image']:
            if isinstance(data['initial_image'], dict):
                initial_image = decode_image(data['initial_image'])
            else:
                initial_shape = data.get('initial_image_shape', default_shape)
                initial_dtype = data.get('initial_image_dtype', default_dtype)
                initial_image = decode_image(data['initial_image'], initial_shape, initial_dtype)
        
        with inference_lock:
            result = compute_vlm_reward(
                image, task_description, reward_type,
                goal_image=goal_image,
                initial_image=initial_image
            )
        
        return jsonify({
            "success": True,
            "score": result["score"],
            "response": result["response"],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route('/compute_rewards_batch', methods=['POST'])
def compute_rewards_batch_endpoint():
    """Batch VLM reward computation endpoint."""
    try:
        data = request.json
        
        images = []
        for img_data in data['images']:
            image_bytes = base64.b64decode(img_data['data'])
            image_shape = img_data['shape']
            image_dtype = img_data.get('dtype', 'uint8')
            image = np.frombuffer(image_bytes, dtype=image_dtype).reshape(image_shape)
            images.append(image)
        
        task_descriptions = data['task_descriptions']
        reward_type = data.get('reward_type', 'progress')
        
        results = compute_vlm_rewards_batch(images, task_descriptions, reward_type)
        
        return jsonify({
            "success": True,
            "results": results,
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route('/compute_robometer', methods=['POST'])
def compute_robometer_endpoint():
    """Video-mode Robometer reward endpoint."""
    try:
        data = request.json
        frames = []
        for frame_data in data.get("frames", []):
            if isinstance(frame_data, dict):
                image_bytes = base64.b64decode(frame_data["data"])
                shape = tuple(frame_data["shape"])
                dtype = frame_data.get("dtype", "uint8")
                arr = np.frombuffer(image_bytes, dtype=dtype).reshape(shape)
            else:
                # Keep compatibility with flat base64 payload.
                image_bytes = base64.b64decode(frame_data)
                shape = tuple(data["frame_shape"])
                dtype = data.get("frame_dtype", "uint8")
                arr = np.frombuffer(image_bytes, dtype=dtype).reshape(shape)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(arr))

        task_description = data["task_description"]
        # Single-flight inference avoids concurrent multi-request OOM spikes.
        with inference_lock:
            result = compute_vlm_robometer(frames, task_description)

        return jsonify(
            {
                "success": True,
                "score": result["score"],
                "raw_score": result["raw_score"],
                "response": result["response"],
                "progress_per_frame": result.get("progress_per_frame", []),
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return jsonify({"success": False, "error": str(e)}), 500


def compute_vlm_completion(
    image: np.ndarray,
    task_description: str,
) -> dict:
    """Determine whether the task is completed (yes/no) and return 1.0 or 0.0."""
    global vlm_model, vlm_processor, device

    if vlm_model is None:
        raise RuntimeError("VLM model not loaded!")
    _activate_adapter("completion")

    # Convert to PIL
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    current_pil = Image.fromarray(image)

    # Matches the published model-card prompt (no reasoning field). A
    # reasoning-first variant was A/B tested against this one on real
    # trajectories and changed neither the yes/no decision nor accuracy, so
    # this is the version that stays aligned with how the adapter was trained.
    prompt_text = (
        f"Task: Whether the task has been completed.\n\n"
        f"The task is: {task_description}\n\n"
        f"You are given:\n"
        f"- Current observation: <image>\n\n"
        f"Has the task been completed?\n"
        f'Output your answer in the following JSON format:\n'
        f'{{ "task_completed": "yes" or "no" }}'
    )

    # Build message with image inline (matching test code format)
    content = [
        {"type": "image", "image": current_pil},
        {"type": "text", "text": prompt_text.replace("<image>", "")},
    ]
    # Split by <image> tag like the test code
    segments = prompt_text.split("<image>")
    content = []
    img_idx = 0
    for i, seg in enumerate(segments):
        if seg.strip():
            content.append({"type": "text", "text": seg.strip()})
        if i < len(segments) - 1:  # insert image between segments
            content.append({"type": "image", "image": current_pil})
            img_idx += 1

    messages = [{"role": "user", "content": content}]

    # Use apply_chat_template with tokenize=True (matching test code)
    inputs = vlm_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    tensor_inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    with torch.no_grad():
        out = vlm_model.generate(
            **tensor_inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    in_len = tensor_inputs["input_ids"].shape[1]
    gen_ids = out[0, in_len:]
    response = vlm_processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Parse yes/no and reasoning from JSON output
    is_complete = False
    reasoning = ""
    try:
        # Try to extract reasoning
        reasoning_pattern = r'"reasoning"\s*:\s*"([^"]*)"'
        reasoning_matches = re.findall(reasoning_pattern, response, re.IGNORECASE)
        if reasoning_matches:
            reasoning = reasoning_matches[-1]

        # Extract task_completed
        pattern = r'"task_completed"\s*:\s*"(yes|no)"'
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            is_complete = matches[-1].lower() == "yes"
        else:
            resp_lower = response.lower()
            if "yes" in resp_lower and "no" not in resp_lower:
                is_complete = True
            elif "no" in resp_lower:
                is_complete = False
    except Exception:
        is_complete = "yes" in response.lower()

    # Always log reasoning for debugging
    import sys
    debug_msg = f"""
[VLM Completion] Task: {task_description}
[VLM Completion] Result: {'COMPLETED' if is_complete else 'NOT COMPLETED'}
[VLM Completion] Reasoning: {reasoning}
[VLM Completion] Full Response: {response}
{"-" * 80}
"""
    print(debug_msg, flush=True)
    sys.stdout.flush()

    # Also write to file
    try:
        with open("/data/vlm_completion_debug.log", "a") as f:
            f.write(debug_msg + "\n")
    except:
        pass

    return {
        "score": 1.0 if is_complete else 0.0,
        "response": response,
        "reasoning": reasoning,
        "completed": is_complete,
    }


@app.route('/compute_completion', methods=['POST'])
def compute_completion_endpoint():
    """Task completion check endpoint."""
    try:
        data = request.json

        # Decode image
        if isinstance(data['image'], dict):
            image_bytes = base64.b64decode(data['image']['data'])
            shape = tuple(data['image']['shape'])
            dtype = data['image'].get('dtype', 'uint8')
        else:
            image_bytes = base64.b64decode(data['image'])
            shape = tuple(data['image_shape'])
            dtype = data.get('image_dtype', 'uint8')
        image = np.frombuffer(image_bytes, dtype=dtype).reshape(shape)

        task_description = data['task_description']

        with inference_lock:
            result = compute_vlm_completion(image, task_description)

        return jsonify({
            "success": True,
            "score": result["score"],
            "response": result["response"],
            "completed": result["completed"],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/compute_comparison', methods=['POST'])
def compute_comparison_endpoint():
    """Compare two images to determine which is closer to task completion."""
    try:
        data = request.json

        def decode_image(img_b64, shape, dtype='uint8'):
            image_bytes = base64.b64decode(img_b64)
            return np.frombuffer(image_bytes, dtype=dtype).reshape(shape)

        image_a = decode_image(
            data['image_a'], data['image_a_shape'],
            data.get('image_a_dtype', 'uint8')
        )
        image_b = decode_image(
            data['image_b'], data['image_b_shape'],
            data.get('image_b_dtype', 'uint8')
        )
        task_description = data['task_description']

        with inference_lock:
            result = compute_vlm_comparison(image_a, image_b, task_description)

        return jsonify({
            "success": True,
            "result": result["result"],
            "response": result["response"],
            "score": result["score"],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route('/compute_roboreward', methods=['POST'])
def compute_roboreward_endpoint():
    """Score a trajectory via RoboReward (1-5 discrete, normalized to 0-1)."""
    try:
        data = request.json

        frames = []
        for frame_data in data['frames']:
            image_bytes = base64.b64decode(frame_data['data'])
            shape = tuple(frame_data['shape'])
            dtype = frame_data.get('dtype', 'uint8')
            arr = np.frombuffer(image_bytes, dtype=dtype).reshape(shape)
            frames.append(Image.fromarray(arr))

        task_description = data['task_description']

        with inference_lock:
            result = compute_vlm_roboreward(frames, task_description)

        return jsonify({
            "success": True,
            "score": result["score"],
            "raw_score": result["raw_score"],
            "response": result["response"],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/compute_topreward', methods=['POST'])
def compute_topreward_endpoint():
    """Score a frame sequence via TOPReward: log P("True" | video, instruction)."""
    try:
        data = request.json

        frames = []
        for frame_data in data['frames']:
            image_bytes = base64.b64decode(frame_data['data'])
            shape = tuple(frame_data['shape'])
            dtype = frame_data.get('dtype', 'uint8')
            arr = np.frombuffer(image_bytes, dtype=dtype).reshape(shape)
            frames.append(Image.fromarray(arr))

        task_description = data['task_description']
        fps = float(data.get('fps', 2.0))

        with inference_lock:
            result = compute_vlm_topreward(frames, task_description, fps=fps)

        return jsonify({
            "success": True,
            "score": result["score"],
            "log_prob": result["log_prob"],
            "token_count": result["token_count"],
            "response": result["response"],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="VLM Reward Server")
    parser.add_argument("--port", type=int, default=5001, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/data/yanruwu/vlm_reward/qwen-vl-finetune/checkpoints_estimate_cot60k_withoutna/checkpoint-469",
        help="Path to VLM model or LoRA adapter"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Path to base model (for LoRA adapter)"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--processor_path",
        type=str,
        default=None,
        help="Optional local processor/tokenizer export (useful with weights-only base caches)",
    )
    parser.add_argument("--tri_contrastive_adapter", type=str, default=None)
    parser.add_argument("--tri_progress_adapter", type=str, default=None)
    parser.add_argument("--tri_completion_adapter", type=str, default=None)
    args = parser.parse_args()

    tri_paths = (
        args.tri_contrastive_adapter,
        args.tri_progress_adapter,
        args.tri_completion_adapter,
    )
    if any(tri_paths) and not all(tri_paths):
        parser.error("tri-head mode requires all three --tri_*_adapter paths")
    if all(tri_paths):
        load_vlm_multi_adapter(
            args.base_model_path,
            args.tri_contrastive_adapter,
            args.tri_progress_adapter,
            args.tri_completion_adapter,
            args.gpu_id,
        )
    else:
        load_vlm_model(
            args.model_path,
            args.base_model_path,
            args.gpu_id,
            processor_path_override=args.processor_path,
        )

    print(f"Starting VLM Reward Server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
