"""
VLM Reward Model for RLinf Integration

支持的模型类型:
1. completion - 判断单帧是否完成任务
2. compare - 比较两帧哪个更接近完成
3. progress - 估计任务完成进度 (0.0-1.0)

Checkpoints:
- completion: checkpoints_completion_new/checkpoint-5472
- compare: checkpoints_compare_title_task/checkpoint-2135
- compare_hard: checkpoints_compare_hard_dpo_v3/checkpoint-455
- progress: checkpoints_estimate_cot60k_withoutna/checkpoint-469
"""

import os
import re
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import AutoModelForCausalLM as Qwen3VLForConditionalGeneration


class VLMRewardModel:
    """
    VLM-based Reward Model for embodied AI tasks.
    
    Supports both:
    - Full fine-tuned models
    - LoRA adapters (automatically detected and merged)
    
    Usage:
        model = VLMRewardModel(
            model_path="/path/to/checkpoint",
            reward_type="progress",  # or "completion", "compare"
            device="cuda:0"
        )
        
        # For progress estimation
        reward = model.compute_reward(image, task_description)
        
        # For comparison
        better_image = model.compare_frames(image_a, image_b, task_description)
    """
    
    # Default base model
    BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
    
    def __init__(
        self,
        model_path: str,
        reward_type: str = "progress",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 256,
        base_model: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to fine-tuned Qwen-VL checkpoint (can be full model or LoRA adapter)
            reward_type: One of "completion", "compare", "compare_hard", "progress"
            device: CUDA device
            torch_dtype: Model precision
            max_new_tokens: Max tokens to generate
            base_model: Base model ID (auto-detect from adapter config if not provided)
        """
        self.reward_type = reward_type
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.base_model = base_model or self.BASE_MODEL
        
        print(f"[VLMRewardModel] Loading model from {model_path}...")
        
        # Check if this is a LoRA adapter
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        is_lora = os.path.exists(adapter_config_path)
        
        if is_lora:
            self._load_lora_model(model_path)
        else:
            self._load_full_model(model_path)
        
        print(f"[VLMRewardModel] Model loaded on {device}, reward_type={reward_type}")
    
    def _load_full_model(self, model_path: str):
        """Load a fully fine-tuned model."""
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            device_map=self.device
        ).eval()
    
    def _load_lora_model(self, model_path: str):
        """Load base model and merge LoRA adapter."""
        from safetensors.torch import load_file
        
        # Load adapter config
        config_path = os.path.join(model_path, "adapter_config.json")
        with open(config_path, 'r') as f:
            adapter_config = json.load(f)
        
        # Check for base model in adapter config
        if 'base_model_name_or_path' in adapter_config:
            self.base_model = adapter_config['base_model_name_or_path']
        
        print(f"[VLMRewardModel] Loading base model: {self.base_model}")
        
        # Load processor from base model
        self.processor = AutoProcessor.from_pretrained(
            self.base_model, trust_remote_code=True
        )
        
        # Load base model
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            device_map=self.device
        )
        
        # Load adapter weights
        adapter_path = os.path.join(model_path, "adapter_model.safetensors")
        print(f"[VLMRewardModel] Loading LoRA from: {adapter_path}")
        adapter_weights = load_file(adapter_path)
        
        # Get LoRA scaling
        lora_alpha = adapter_config.get('lora_alpha', 128)
        r = adapter_config.get('r', 64)
        scaling = lora_alpha / r
        
        # Separate LoRA and full module weights
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
        
        # Load full modules
        if full_modules:
            print(f"[VLMRewardModel] Loading {len(full_modules)} full module parameters...")
            self.model.load_state_dict(full_modules, strict=False)
        
        # Merge LoRA weights
        print(f"[VLMRewardModel] Merging {len(lora_pairs)} LoRA adapters (scaling={scaling:.2f})...")
        model_state_dict = self.model.state_dict()
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
        
        self.model.load_state_dict(model_state_dict, strict=False)
        print(f"[VLMRewardModel] ✅ Successfully merged {merged_count} LoRA adapters")
        self.model.eval()
    
    def _load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """Load image from various formats."""
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")
    
    def _build_progress_prompt(self, task_description: str) -> str:
        """Build prompt for progress estimation."""
        return f"""You are a progress evaluator for robotic manipulation tasks.

The task is: {task_description}

Based on the image, estimate the completion progress of this task.
Output a single decimal number between 0.0 and 1.0, where:
- 0.0 means the task has not started
- 0.5 means the task is halfway complete  
- 1.0 means the task is fully complete

Analyze the image and provide your estimate.

Progress:"""

    def _build_completion_prompt(self, task_description: str) -> str:
        """Build prompt for completion detection."""
        return f"""You are evaluating whether a robotic manipulation task is complete.

The task is: {task_description}

Based on the image, determine if the task is complete.
Answer with only "Yes" or "No".

Is the task complete?"""

    def _build_compare_prompt(self, task_description: str, is_hard: bool = False) -> str:
        """Build prompt for frame comparison."""
        difficulty = " (hard)" if is_hard else ""
        return f"""Task: Compare the completion progress{difficulty}.

The task is: {task_description}

You are given two images:
- Image A (first image)
- Image B (second image)

Determine which image shows more progress toward completing the task.

**INSTRUCTIONS:**
1. Identify the target object and goal from the task description.
2. Evaluate Image A: Is the agent interacting with the correct object? What progress is shown?
3. Evaluate Image B: Is the agent interacting with the correct object? What progress is shown?
4. Compare purely based on task alignment.

You MUST end with: {{"more_complete_image": "ImageA"}} or {{"more_complete_image": "ImageB"}}

Analysis:"""

    def _parse_progress(self, response: str) -> float:
        """Parse progress value from model response."""
        # Try to find decimal number
        matches = re.findall(r'(\d+\.?\d*)', response)
        for match in matches:
            try:
                value = float(match)
                if 0.0 <= value <= 1.0:
                    return value
                elif 0 <= value <= 100:
                    return value / 100.0
            except:
                continue
        return 0.5  # Default if parsing fails

    def _parse_completion(self, response: str) -> bool:
        """Parse completion status from model response."""
        response_lower = response.lower().strip()
        if "yes" in response_lower:
            return True
        elif "no" in response_lower:
            return False
        # Check for other positive indicators
        if any(word in response_lower for word in ["complete", "done", "finished", "success"]):
            return True
        return False

    def _parse_comparison(self, response: str) -> str:
        """Parse comparison result from model response."""
        # Look for JSON format
        pattern = r'"more_complete_image"\s*:\s*"(Image[AB])"'
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1]
        
        # Fallback: look for ImageA/ImageB mentions
        response_upper = response.upper()
        if "IMAGEA" in response_upper and "IMAGEB" not in response_upper:
            return "ImageA"
        elif "IMAGEB" in response_upper and "IMAGEA" not in response_upper:
            return "ImageB"
        
        # Check last 200 chars for conclusion
        tail = response[-200:].upper()
        if "IMAGEA" in tail:
            return "ImageA"
        elif "IMAGEB" in tail:
            return "ImageB"
        
        return "unknown"

    @torch.no_grad()
    def _generate(self, messages: List[Dict]) -> str:
        """Generate response from the model."""
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # Greedy for consistency
        )
        
        # Decode only generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0, input_len:]
        response = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()

    def compute_progress(
        self,
        image: Union[str, np.ndarray, Image.Image],
        task_description: str
    ) -> float:
        """
        Estimate task completion progress (0.0 to 1.0).
        
        Args:
            image: Current observation image
            task_description: Natural language task description
            
        Returns:
            Progress value between 0.0 and 1.0
        """
        img = self._load_image(image)
        prompt = self._build_progress_prompt(task_description)
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]
        }]
        
        response = self._generate(messages)
        return self._parse_progress(response)

    def check_completion(
        self,
        image: Union[str, np.ndarray, Image.Image],
        task_description: str
    ) -> Tuple[bool, float]:
        """
        Check if task is complete.
        
        Args:
            image: Current observation image
            task_description: Natural language task description
            
        Returns:
            Tuple of (is_complete, confidence)
        """
        img = self._load_image(image)
        prompt = self._build_completion_prompt(task_description)
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]
        }]
        
        response = self._generate(messages)
        is_complete = self._parse_completion(response)
        confidence = 1.0 if is_complete else 0.0
        
        return is_complete, confidence

    def compare_frames(
        self,
        image_a: Union[str, np.ndarray, Image.Image],
        image_b: Union[str, np.ndarray, Image.Image],
        task_description: str,
        is_hard: bool = False
    ) -> Tuple[str, float]:
        """
        Compare two frames to determine which shows more progress.
        
        Args:
            image_a: First observation image
            image_b: Second observation image
            task_description: Natural language task description
            is_hard: Whether to use hard comparison (smaller frame differences)
            
        Returns:
            Tuple of (better_image, reward_diff)
            better_image: "ImageA" or "ImageB"
            reward_diff: Positive if ImageB is better, negative if ImageA is better
        """
        img_a = self._load_image(image_a)
        img_b = self._load_image(image_b)
        prompt = self._build_compare_prompt(task_description, is_hard)
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_a},
                {"type": "image", "image": img_b},
                {"type": "text", "text": prompt}
            ]
        }]
        
        response = self._generate(messages)
        better = self._parse_comparison(response)
        
        # Convert to reward difference
        if better == "ImageA":
            reward_diff = -1.0  # A is better, so moving from A to B is negative
        elif better == "ImageB":
            reward_diff = 1.0   # B is better, so moving from A to B is positive
        else:
            reward_diff = 0.0   # Unknown/tie
        
        return better, reward_diff

    def compute_reward(
        self,
        image: Union[str, np.ndarray, Image.Image],
        task_description: str,
        prev_image: Optional[Union[str, np.ndarray, Image.Image]] = None
    ) -> float:
        """
        Unified interface to compute reward based on reward_type.
        
        Args:
            image: Current observation image
            task_description: Natural language task description
            prev_image: Previous observation (required for compare mode)
            
        Returns:
            Reward value
        """
        if self.reward_type == "progress":
            return self.compute_progress(image, task_description)
        
        elif self.reward_type == "completion":
            is_complete, _ = self.check_completion(image, task_description)
            return 1.0 if is_complete else 0.0
        
        elif self.reward_type in ["compare", "compare_hard"]:
            if prev_image is None:
                raise ValueError("prev_image is required for comparison reward")
            is_hard = (self.reward_type == "compare_hard")
            _, reward_diff = self.compare_frames(prev_image, image, task_description, is_hard)
            return reward_diff
        
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

    def compute_batch_rewards(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        task_descriptions: List[str],
        prev_images: Optional[List[Union[str, np.ndarray, Image.Image]]] = None
    ) -> List[float]:
        """
        Compute rewards for a batch of observations.
        
        Args:
            images: List of current observation images
            task_descriptions: List of task descriptions
            prev_images: List of previous observations (for compare mode)
            
        Returns:
            List of reward values
        """
        rewards = []
        for i, (img, task) in enumerate(zip(images, task_descriptions)):
            prev_img = prev_images[i] if prev_images else None
            reward = self.compute_reward(img, task, prev_img)
            rewards.append(reward)
        return rewards


# Convenience factory functions
def create_progress_reward_model(
    checkpoint_path: str = "/data/yanruwu/vlm_reward/qwen-vl-finetune/checkpoints_estimate_cot60k_withoutna/checkpoint-469",
    device: str = "cuda:0"
) -> VLMRewardModel:
    """Create a progress estimation reward model."""
    return VLMRewardModel(checkpoint_path, reward_type="progress", device=device)


def create_completion_reward_model(
    checkpoint_path: str = "/data/yanruwu/vlm_reward/qwen-vl-finetune/checkpoints_completion_new/checkpoint-5472",
    device: str = "cuda:0"
) -> VLMRewardModel:
    """Create a completion detection reward model."""
    return VLMRewardModel(checkpoint_path, reward_type="completion", device=device)


def create_compare_reward_model(
    checkpoint_path: str = "/data/yanruwu/vlm_reward/qwen-vl-finetune/checkpoints_compare_title_task/checkpoint-2135",
    device: str = "cuda:0",
    hard_mode: bool = False
) -> VLMRewardModel:
    """Create a frame comparison reward model."""
    if hard_mode:
        checkpoint_path = "/data/yanruwu/vlm_reward/qwen-vl-finetune/checkpoints_compare_hard_dpo_v3/checkpoint-455"
    reward_type = "compare_hard" if hard_mode else "compare"
    return VLMRewardModel(checkpoint_path, reward_type=reward_type, device=device)


if __name__ == "__main__":
    # Quick test
    print("Testing VLM Reward Model...")
    
    # Test progress model
    model = create_progress_reward_model()
    
    # You would replace this with an actual test image
    # reward = model.compute_progress("test_image.png", "pick up the red block")
    # print(f"Progress: {reward}")
    
    print("VLM Reward Model loaded successfully!")
