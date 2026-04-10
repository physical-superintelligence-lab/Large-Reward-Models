# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VLM-Enhanced ManiSkill Environment.

Extends ManiskillEnv with VLM-based reward via HTTP server.
Supports 5 VLM reward modes: contrastive, completion, progress, roboreward, robometer.
Blends with env reward or runs in pure VLM reward mode.

Config keys (in env YAML):
    use_vlm_reward: true
    vlm_server_url: "http://localhost:5002"
    vlm_reward_scale: 1.0
    vlm_reward_weight: 1.0       # blend: (1-w)*env + w*vlm
    vlm_reward_type: completion  # completion | progress | quality (for single-frame modes)
    vlm_use_comparison: false    # contrastive mode (compares consecutive frames)
    vlm_use_roboreward: false    # roboreward mode (1-5 video scoring)
    vlm_use_robometer: false     # robometer mode (per-frame progress heads)
    vlm_completion_binary_reward: false  # if true, completion reward is exactly 0/1
    vlm_success_threshold: 0.5
    vlm_include_initial_image: false
    vlm_call_interval: 10        # call VLM every N steps
    vlm_pure_reward: false       # if true, ignore env reward entirely
"""

import os
import re
import sys
from typing import Optional, Union
from datetime import datetime

import numpy as np
import torch
from mani_skill.utils.structs.types import Array
from PIL import Image

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv

# Add VLM integration path
VLM_INTEGRATION_PATH = "/data/vlm_reward/rlinf_integration"
if os.path.exists(VLM_INTEGRATION_PATH) and VLM_INTEGRATION_PATH not in sys.path:
    sys.path.insert(0, VLM_INTEGRATION_PATH)

try:
    from vlm_reward_client import VLMRewardClient
    VLM_CLIENT_AVAILABLE = True
except ImportError:
    print("Warning: VLMRewardClient not available")
    VLM_CLIENT_AVAILABLE = False


class VLMManiskillEnv(ManiskillEnv):
    """
    ManiSkill environment augmented with VLM-based rewards.

    Uses HTTP calls to a VLM Reward Server to compute completion or
    dense progress signals from visual observations.
    """

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        # Initialize base ManiSkill environment
        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info,
            record_metrics=record_metrics,
        )

        # VLM reward config
        self.use_vlm_reward = cfg.get("use_vlm_reward", True)
        self.vlm_reward_weight = cfg.get("vlm_reward_weight", 0.5)
        self.vlm_reward_scale = cfg.get("vlm_reward_scale", 5.0)
        self.vlm_call_interval = cfg.get("vlm_call_interval", 1)
        self.vlm_use_comparison = bool(cfg.get("vlm_use_comparison", False))
        self.vlm_use_roboreward = bool(cfg.get("vlm_use_roboreward", False))
        self.vlm_use_robometer = bool(cfg.get("vlm_use_robometer", False))
        self.vlm_roboreward_max_frames = max(
            1, int(cfg.get("vlm_roboreward_max_frames", 16))
        )
        self.vlm_robometer_max_frames = max(
            1, int(cfg.get("vlm_robometer_max_frames", 16))
        )
        self.vlm_sample_envs = int(cfg.get("vlm_sample_envs", 0))
        self.vlm_calls_per_episode = int(cfg.get("vlm_calls_per_episode", 0))
        self.vlm_use_rel_reward = bool(cfg.get("vlm_use_rel_reward", False))
        self._max_episode_steps = int(
            cfg.get("max_episode_steps", cfg.get("max_steps_per_rollout_epoch", 80))
        )
        self.vlm_pure_reward = bool(cfg.get("vlm_pure_reward", False))
        self.vlm_non_call_reward_mode = str(
            cfg.get("vlm_non_call_reward_mode", "env")
        ).lower()
        if self.vlm_non_call_reward_mode not in {"env", "hold", "zero"}:
            print(
                f"[VLM] WARNING: unsupported vlm_non_call_reward_mode={self.vlm_non_call_reward_mode}, fallback to env"
            )
            self.vlm_non_call_reward_mode = "env"
        # In pure VLM mode, avoid any env-reward fallback on interval-skipped steps.
        if self.vlm_pure_reward and self.vlm_non_call_reward_mode == "env":
            self.vlm_non_call_reward_mode = "hold"
        self.vlm_require_connection = bool(
            cfg.get("vlm_require_connection", self.vlm_pure_reward)
        )
        self.vlm_server_url = cfg.get("vlm_server_url", "http://localhost:5002")
        # Fallback task description for envs without get_language_instruction()
        self.vlm_task_description = cfg.get(
            "vlm_task_description", "Pick up the cube"
        )
        self.vlm_task_prompt_template = cfg.get("vlm_task_prompt_template", None)
        self.vlm_reward_type = str(cfg.get("vlm_reward_type", "completion")).lower()
        self.vlm_completion_binary_reward = bool(
            cfg.get("vlm_completion_binary_reward", False)
        )
        if self.vlm_reward_type not in {"completion", "progress", "quality"}:
            print(
                f"[VLM] WARNING: unsupported vlm_reward_type={self.vlm_reward_type}, fallback to completion"
            )
            self.vlm_reward_type = "completion"
        self.vlm_success_threshold = float(cfg.get("vlm_success_threshold", 0.5))
        self._vlm_include_initial_image = cfg.get("vlm_include_initial_image", False)

        # Internal state
        self._vlm_client: Optional[VLMRewardClient] = None
        self._vlm_connected = False
        self._vlm_step_counter = 0
        self._vlm_initial_images = [None] * self.num_envs
        self._vlm_prev_images: dict[int, np.ndarray] = {}  # for contrastive mode
        self._vlm_frame_buffers = {}  # for robometer video mode
        self._vlm_roboreward_frame_buffers: dict[int, list] = {}  # for roboreward video mode
        self._vlm_active_steps = set()
        self._vlm_last_rewards = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        # VLM logging
        self.vlm_log_file = cfg.get(
            "vlm_log_file",
            cfg.get("vlm_log_path", "/data/RLinf/vlm_maniskill_detailed.log"),
        )
        self._vlm_log_enabled = cfg.get("vlm_log_enabled", True)
        if self._vlm_log_enabled:
            log_dir = os.path.dirname(self.vlm_log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

        # Optional open-loop analysis log. Each record stores per-step
        # VLM score and env-side signals for offline diagnostics.
        self._vlm_openloop_log_enabled = bool(
            cfg.get("vlm_openloop_log_enabled", False)
        )
        self._vlm_openloop_log_file = cfg.get(
            "vlm_openloop_log_file", "/data/RLinf/vlm_openloop_eval.tsv"
        )
        if self._vlm_openloop_log_enabled:
            openloop_dir = os.path.dirname(self._vlm_openloop_log_file)
            if openloop_dir:
                os.makedirs(openloop_dir, exist_ok=True)

        # VLM image dump
        self._vlm_save_images = cfg.get("vlm_save_images", False)
        self._vlm_image_dir = cfg.get("vlm_image_dir", "/data/RLinf/vlm_images")
        self._vlm_image_interval = cfg.get("vlm_image_interval", 1)
        self._vlm_image_max = cfg.get("vlm_image_max", 5000)
        self._vlm_save_success_only = cfg.get("vlm_save_success_only", True)
        self._vlm_success_source = cfg.get("vlm_success_source", "vlm")
        self._vlm_success_metric = cfg.get("vlm_success_metric", "success")
        self._vlm_rgb_source = cfg.get("vlm_rgb_source", "obs")
        self._vlm_image_count = 0
        if self._vlm_save_images:
            os.makedirs(self._vlm_image_dir, exist_ok=True)

        if self.use_vlm_reward and VLM_CLIENT_AVAILABLE:
            self._init_vlm_client()

    def _init_vlm_client(self):
        """Initialize HTTP client to VLM reward server."""
        print(f"[VLM] Connecting to VLM server at {self.vlm_server_url}")
        self._vlm_client = VLMRewardClient(
            server_url=self.vlm_server_url, timeout=60.0
        )
        try:
            self._vlm_connected = self._vlm_client.health_check()
            if self._vlm_connected:
                print(f"[VLM] Connected to reward server at {self.vlm_server_url}")
            else:
                print(f"[VLM] WARNING: Server at {self.vlm_server_url} not responding")
                if self.vlm_require_connection:
                    raise RuntimeError(
                        f"VLM server not available at {self.vlm_server_url}, aborting because vlm_require_connection=true"
                    )
        except Exception as e:
            print(f"[VLM] WARNING: Failed to connect: {e}")
            self._vlm_connected = False
            if self.vlm_require_connection:
                raise

    def _get_task_descriptions(self) -> list:
        """Get task descriptions, with fallback for envs without language instructions."""
        try:
            descs = self.env.unwrapped.get_language_instruction()
            if isinstance(descs, str):
                return [descs] * self.num_envs
            return list(descs)
        except (AttributeError, NotImplementedError):
            return [self.vlm_task_description] * self.num_envs

    def _format_vlm_prompt(self, task_desc: str) -> str:
        if not self.vlm_task_prompt_template:
            return task_desc
        try:
            return self.vlm_task_prompt_template.format(task=task_desc)
        except Exception:
            return task_desc

    def _select_env_indices(self, total: int) -> list[int]:
        """Select which envs to score this call."""
        if total <= 0:
            return []
        if self.vlm_sample_envs <= 0 or self.vlm_sample_envs >= total:
            return list(range(total))
        return list(np.random.choice(total, size=self.vlm_sample_envs, replace=False))

    def _wrap_obs(self, raw_obs):
        """Override to add task_descriptions and wrist_images for Pi0 compatibility.

        Pi0's obs_processor expects 'task_descriptions' and 'wrist_images' keys,
        but the 'simple' wrap mode only provides {main_images, extra_view_images, states}.
        """
        wrapped_obs = super()._wrap_obs(raw_obs)
        # Add task descriptions (required by Pi0 obs_processor → "prompt")
        if "task_descriptions" not in wrapped_obs:
            wrapped_obs["task_descriptions"] = self._get_task_descriptions()
        # Add wrist_images placeholder (obs_processor checks for it)
        if "wrist_images" not in wrapped_obs:
            wrapped_obs["wrist_images"] = None
        return wrapped_obs

    def _update_initial_images_from_obs(self, extracted_obs, env_indices=None):
        """Cache per-env initial observation image for progress-style prompts."""
        if not self._vlm_include_initial_image:
            return
        rgb_list = self._extract_rgb_from_obs(extracted_obs)
        if env_indices is None:
            env_indices = range(min(self.num_envs, len(rgb_list)))
        for idx in env_indices:
            if idx >= len(rgb_list):
                continue
            rgb = rgb_list[idx]
            if rgb is None:
                continue
            self._vlm_initial_images[idx] = np.array(rgb, copy=True)

    def _extract_rgb_for_vlm(self, extracted_obs) -> list:
        """
        Extract per-environment RGB images from observations for VLM scoring.

        Returns list of numpy arrays (H, W, 3) uint8, one per environment.
        """
        images = None
        if self._vlm_rgb_source == "render":
            try:
                from mani_skill.utils import common
                img = self.env.render()
                img = common.to_numpy(img)
                if img.ndim == 3:
                    img = img[None]  # Add batch dim
                images = img
            except Exception:
                images = None

        if images is None:
            images = extracted_obs.get("main_images", None)
            if images is None:
                # Try rendering directly
                try:
                    from mani_skill.utils import common
                    img = self.env.render()
                    img = common.to_numpy(img)
                    if img.ndim == 3:
                        img = img[None]  # Add batch dim
                    images = img
                except Exception:
                    return [None] * self.num_envs

        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        # Ensure uint8
        if images.dtype != np.uint8:
            if images.max() <= 1.0:
                images = (images * 255).astype(np.uint8)
            else:
                images = images.astype(np.uint8)

        # images shape: (B, H, W, C)
        return [images[i] for i in range(min(self.num_envs, len(images)))]

    def _update_roboreward_buffers(self, rgb_list: list) -> None:
        """Append current trigger-point frame to each env buffer for roboreward video mode."""
        total = min(self.num_envs, len(rgb_list))
        for env_idx in range(total):
            rgb = rgb_list[env_idx]
            if rgb is None or rgb.size == 0:
                continue
            if env_idx not in self._vlm_roboreward_frame_buffers:
                self._vlm_roboreward_frame_buffers[env_idx] = []
            self._vlm_roboreward_frame_buffers[env_idx].append(rgb.copy())
            buf = self._vlm_roboreward_frame_buffers[env_idx]
            if len(buf) > self.vlm_roboreward_max_frames:
                keep = np.linspace(
                    0, len(buf) - 1, self.vlm_roboreward_max_frames, dtype=int
                )
                self._vlm_roboreward_frame_buffers[env_idx] = [buf[i] for i in keep]

    def _compute_vlm_comparison_rewards(
        self,
        rgb_list: list,
        task_descs: list,
        vlm_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VLM rewards using frame comparison (contrastive mode)."""
        total = min(self.num_envs, len(rgb_list))
        selected = self._select_env_indices(total)

        # Always update prev_images for ALL envs (so sampling doesn't desync)
        for env_idx in range(total):
            rgb = rgb_list[env_idx]
            if rgb is not None and rgb.size > 0 and env_idx not in selected:
                self._vlm_prev_images[env_idx] = rgb.copy()

        for env_idx in selected:
            try:
                rgb = rgb_list[env_idx]
                if rgb is None or rgb.size == 0:
                    continue

                prev_rgb = self._vlm_prev_images.get(env_idx, None)
                self._vlm_prev_images[env_idx] = rgb.copy()

                if prev_rgb is None:
                    continue

                task = task_descs[env_idx] if env_idx < len(task_descs) else task_descs[0]

                result = self._vlm_client.compute_comparison(prev_rgb, rgb, task)
                score = result.get("score", 0.0) if isinstance(result, dict) else 0.0

                vlm_rewards[env_idx] = score * self.vlm_reward_scale

                if self._vlm_log_enabled:
                    try:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        log_line = (
                            f"{timestamp},env_{env_idx},step_{self._vlm_step_counter},"
                            f"comparison,{task},result={result.get('result', 'N/A')},"
                            f"score={score:.1f},reward={vlm_rewards[env_idx]:.4f},"
                            f"{result.get('response', '')[:120]}\n"
                        )
                        with open(self.vlm_log_file, "a") as f:
                            f.write(log_line)
                    except Exception:
                        pass

            except Exception as e:
                print(f"[VLM] Warning: comparison reward failed for env {env_idx}: {e}")

        return vlm_rewards

    def _compute_vlm_roboreward_rewards(
        self,
        rgb_list: list,
        task_descs: list,
        vlm_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VLM rewards using RoboReward video scoring (1-5 → normalized 0-1)."""
        total = min(self.num_envs, len(rgb_list))
        selected = self._select_env_indices(total)

        # Keep previous values for envs not sampled this round.
        if self.vlm_sample_envs > 0 and self.vlm_sample_envs < total:
            vlm_rewards[:] = self._vlm_last_rewards

        for env_idx in selected:
            frames = self._vlm_roboreward_frame_buffers.get(env_idx, [])
            if not frames:
                continue

            task = task_descs[env_idx] if env_idx < len(task_descs) else task_descs[0]

            try:
                result = self._vlm_client.compute_roboreward(frames, task)
                score = result.get("score", None)

                if score is not None:
                    vlm_rewards[env_idx] = float(score) * self.vlm_reward_scale

                    if self._vlm_log_enabled:
                        try:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            log_line = (
                                f"{timestamp},env_{env_idx},step_{self._vlm_step_counter},"
                                f"roboreward,{task},n_frames={len(frames)},"
                                f"raw_score={result.get('raw_score')},normalized={score:.4f},"
                                f"reward={vlm_rewards[env_idx]:.4f},"
                                f"{result.get('response', '')[:120]}\n"
                            )
                            with open(self.vlm_log_file, "a") as f:
                                f.write(log_line)
                        except Exception:
                            pass

            except Exception as e:
                print(f"[VLM] Warning: roboreward failed for env {env_idx}: {e}")

        return vlm_rewards

    def _update_robometer_buffers(self, rgb_list: list) -> None:
        """Append current trigger-point frame to each env buffer."""
        total = min(self.num_envs, len(rgb_list))
        for env_idx in range(total):
            rgb = rgb_list[env_idx]
            if rgb is None or rgb.size == 0:
                continue
            if env_idx not in self._vlm_frame_buffers:
                self._vlm_frame_buffers[env_idx] = []
            self._vlm_frame_buffers[env_idx].append(rgb.copy())
            buf = self._vlm_frame_buffers[env_idx]
            if len(buf) > self.vlm_robometer_max_frames:
                keep = np.linspace(
                    0, len(buf) - 1, self.vlm_robometer_max_frames, dtype=int
                )
                self._vlm_frame_buffers[env_idx] = [buf[i] for i in keep]

    def _compute_vlm_robometer_rewards(
        self,
        rgb_list: list,
        task_descs: list,
        vlm_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute video-mode rewards using buffered frames and /compute_robometer."""
        total = min(self.num_envs, len(rgb_list))
        selected = self._select_env_indices(total)

        # Keep previous values for envs not sampled this round.
        if self.vlm_sample_envs > 0 and self.vlm_sample_envs < total:
            vlm_rewards[:] = self._vlm_last_rewards

        for env_idx in selected:
            frames = self._vlm_frame_buffers.get(env_idx, [])
            if not frames:
                continue

            task_desc = task_descs[env_idx] if env_idx < len(task_descs) else task_descs[0]
            task_prompt = self._format_vlm_prompt(task_desc)

            try:
                result = self._vlm_client.compute_robometer(frames, task_prompt)
                score = float(result.get("score", 0.0))
                if self.vlm_use_rel_reward and self.vlm_reward_scale != 0.0:
                    prev_score = float(self._vlm_last_rewards[env_idx].item()) / self.vlm_reward_scale
                    score = score - prev_score

                vlm_rewards[env_idx] = score * self.vlm_reward_scale

                if self._vlm_log_enabled:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    response = result.get("response", "")
                    raw_score = result.get("raw_score", None)
                    log_line = (
                        f"{timestamp},env_{env_idx},step_{self._vlm_step_counter},robometer_video,"
                        f"{task_desc},{score:.3f},raw={raw_score},frames={len(frames)},"
                        f"{response[:120]}\n"
                    )
                    with open(self.vlm_log_file, "a") as f:
                        f.write(log_line)
            except Exception as e:
                print(f"[VLM] Warning: robometer reward failed for env {env_idx}: {e}")

        return vlm_rewards

    def _compute_vlm_rewards(self, extracted_obs) -> torch.Tensor:
        """
        Compute VLM rewards for all environments.

        Returns tensor of shape (num_envs,) on self.device.
        """
        vlm_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        if not self._vlm_connected or self._vlm_client is None:
            return vlm_rewards

        rgb_list = self._extract_rgb_for_vlm(extracted_obs)

        # Get task descriptions
        task_descs = self._get_task_descriptions()

        if self.vlm_use_comparison:
            return self._compute_vlm_comparison_rewards(rgb_list, task_descs, vlm_rewards)

        if self.vlm_use_roboreward:
            self._update_roboreward_buffers(rgb_list)
            return self._compute_vlm_roboreward_rewards(rgb_list, task_descs, vlm_rewards)

        if self.vlm_use_robometer:
            self._update_robometer_buffers(rgb_list)
            return self._compute_vlm_robometer_rewards(rgb_list, task_descs, vlm_rewards)

        # Single-frame modes: completion / progress / quality
        for i in range(self.num_envs):
            try:
                rgb = rgb_list[i]
                if rgb is None or rgb.size == 0:
                    continue

                task_desc = task_descs[i] if i < len(task_descs) else task_descs[0]
                task_prompt = (
                    self._format_vlm_prompt(task_desc)
                    if self.vlm_reward_type == "completion"
                    else task_desc
                )

                if self.vlm_reward_type == "completion":
                    result = self._vlm_client.compute_completion(rgb, task_prompt)
                    if isinstance(result, dict):
                        raw_score = float(result.get("score", 0.0))
                        completed = bool(result.get("completed", False))
                        if self.vlm_completion_binary_reward:
                            score = 1.0 if completed else 0.0
                            is_success = completed
                        else:
                            score = raw_score
                            is_success = completed or (
                                raw_score >= self.vlm_success_threshold
                            )
                        response = result.get("response", "")
                    else:
                        score = 0.0
                        response = ""
                        is_success = False
                else:
                    initial_rgb = None
                    if self._vlm_include_initial_image and i < len(self._vlm_initial_images):
                        initial_rgb = self._vlm_initial_images[i]
                    result = self._vlm_client.compute_reward(
                        rgb,
                        task_prompt,
                        reward_type=self.vlm_reward_type,
                        initial_image=initial_rgb,
                    )
                    if isinstance(result, dict):
                        score = result.get("score", 0.0)
                        response = result.get("response", "")
                        is_success = float(score) >= self.vlm_success_threshold
                    else:
                        score = 0.0
                        response = ""
                        is_success = False

                vlm_rewards[i] = float(score) * self.vlm_reward_scale
                if self._vlm_success_source == "vlm":
                    self._maybe_save_image(rgb, task_desc, i, is_success)

                # Log detailed VLM call
                if self._vlm_log_enabled:
                    try:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        log_line = f"{timestamp},env_{i},step_{self._vlm_step_counter},{self.vlm_reward_type},{task_desc},{float(score):.3f},{response[:100]}\n"
                        with open(self.vlm_log_file, "a") as f:
                            f.write(log_line)
                    except Exception as log_err:
                        pass  # Don't crash training for logging errors

            except Exception as e:
                print(f"[VLM] Warning: reward failed for env {i}: {e}")

        return vlm_rewards

    def _maybe_save_image(self, rgb, task_desc, env_idx, is_success):
        if not self._vlm_save_images:
            return
        if self._vlm_image_count >= self._vlm_image_max:
            return
        if self._vlm_step_counter % self._vlm_image_interval != 0:
            return
        if self._vlm_save_success_only and not is_success:
            return
        try:
            safe_task = re.sub(r"[^A-Za-z0-9_-]+", "_", task_desc.strip())
            safe_task = safe_task.strip("_")[:64] or "task"
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            img_path = os.path.join(
                self._vlm_image_dir,
                f"{ts}_env{env_idx}_step{self._vlm_step_counter}_{safe_task}.png",
            )
            Image.fromarray(rgb).save(img_path)
            self._vlm_image_count += 1
        except Exception:
            pass

    def _safe_index(self, value, idx, default=float("nan")):
        """Best-effort scalar extraction from tensors/arrays/lists for logging."""
        if value is None:
            return default
        try:
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    return value.item()
                if idx < value.shape[0]:
                    return value[idx].item()
                return default
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    return value.item()
                if idx < value.shape[0]:
                    return value[idx].item()
                return default
            if isinstance(value, (list, tuple)):
                if idx < len(value):
                    return value[idx]
                return default
            return value
        except Exception:
            return default

    def _log_openloop_step(self, infos, step_reward, vlm_rewards, terminations, truncations):
        """Append per-step VLM/env signals for open-loop metric computation."""
        if not self._vlm_openloop_log_enabled:
            return
        try:
            episode_info = infos.get("episode", {}) if isinstance(infos, dict) else {}
            success_tensor = infos.get("success", None) if isinstance(infos, dict) else None
            done_tensor = torch.logical_or(terminations, truncations)
            task_descs = self._get_task_descriptions()
            worker_rank = getattr(self.worker_info, "rank", -1)
            timestamp = datetime.now().strftime("%H:%M:%S")

            lines = []
            for i in range(self.num_envs):
                task_desc = task_descs[i] if i < len(task_descs) else task_descs[0]
                vlm_score = float(self._safe_index(vlm_rewards, i, 0.0))
                env_success = int(bool(self._safe_index(success_tensor, i, False)))
                env_step_reward = float(self._safe_index(step_reward, i, 0.0))
                env_done = int(bool(self._safe_index(done_tensor, i, False)))
                env_return = float(
                    self._safe_index(episode_info.get("return", None), i, float("nan"))
                )
                env_episode_len = float(
                    self._safe_index(
                        episode_info.get("episode_len", None), i, float("nan")
                    )
                )

                lines.append(
                    f"{timestamp}	{worker_rank}	{i}	{self._vlm_step_counter}	"
                    f"{self.vlm_reward_type}	{task_desc}	{vlm_score:.6f}	"
                    f"{env_success}	{env_step_reward:.6f}	{env_done}	"
                    f"{env_return:.6f}	{env_episode_len:.1f}\n"
                )

            with open(self._vlm_openloop_log_file, "a") as f:
                f.writelines(lines)
        except Exception:
            # Open-loop logging should never interrupt stepping.
            pass

    def _save_env_success_images(self, extracted_obs, infos):
        """Save images when env success is reported."""
        if not isinstance(infos, dict):
            return

        success = None
        obs_for_save = extracted_obs

        if "final_info" in infos and isinstance(infos["final_info"], dict):
            final_info = infos["final_info"]
            if self._vlm_success_metric == "success":
                if "success" in final_info:
                    success = final_info["success"]
            elif (
                "episode" in final_info
                and isinstance(final_info["episode"], dict)
                and "success_once" in final_info["episode"]
            ):
                success = final_info["episode"]["success_once"]
            if "final_observation" in infos:
                obs_for_save = infos["final_observation"]

        if success is None:
            if self._vlm_success_metric == "success":
                if "success" in infos:
                    success = infos["success"]
            elif (
                "episode" in infos
                and isinstance(infos["episode"], dict)
                and "success_once" in infos["episode"]
            ):
                success = infos["episode"]["success_once"]

        if success is None:
            return

        if isinstance(success, torch.Tensor):
            success_mask = success.detach().cpu().numpy().astype(bool)
        else:
            success_mask = np.asarray(success).astype(bool)

        if not success_mask.any():
            return

        rgb_list = self._extract_rgb_from_obs(obs_for_save)
        task_descs = self._get_task_descriptions()
        for i, ok in enumerate(success_mask):
            if not ok:
                continue
            task_desc = task_descs[i] if i < len(task_descs) else task_descs[0]
            rgb = rgb_list[i]
            if rgb is None or rgb.size == 0:
                continue
            self._maybe_save_image(rgb, task_desc, i, True)

    def _extract_rgb_from_obs(self, extracted_obs) -> list:
        """Extract RGB directly from obs without render fallback."""
        images = extracted_obs.get("main_images", None)
        if images is None:
            return [None] * self.num_envs
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        if images.dtype != np.uint8:
            if images.max() <= 1.0:
                images = (images * 255).astype(np.uint8)
            else:
                images = images.astype(np.uint8)
        return [images[i] for i in range(min(self.num_envs, len(images)))]

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Reset and refresh cached initial frames / robometer buffers."""
        extracted_obs, infos = super().reset(seed=seed, options=options)
        env_indices = None
        if options is not None and "env_idx" in options:
            env_idx = options["env_idx"]
            if isinstance(env_idx, torch.Tensor):
                env_indices = env_idx.detach().cpu().tolist()
            else:
                env_indices = np.asarray(env_idx).tolist()
        self._update_initial_images_from_obs(extracted_obs, env_indices=env_indices)

        # Reset video buffers and trigger schedule.
        if env_indices is None:
            self._vlm_prev_images = {}
            self._vlm_frame_buffers = {}
            self._vlm_roboreward_frame_buffers = {}
            self._vlm_active_steps = set()
            if self.vlm_calls_per_episode > 0 and self.vlm_call_interval > 0:
                all_triggers = list(
                    range(
                        self.vlm_call_interval,
                        self._max_episode_steps + 1,
                        self.vlm_call_interval,
                    )
                )
                if all_triggers:
                    envs_per_trigger = max(
                        1,
                        self.vlm_sample_envs if self.vlm_sample_envs > 0 else self.num_envs,
                    )
                    num_triggers = max(1, self.vlm_calls_per_episode // envs_per_trigger)
                    if num_triggers < len(all_triggers):
                        chosen = np.random.choice(
                            all_triggers, size=num_triggers, replace=False
                        )
                        self._vlm_active_steps = set(chosen.tolist())
                    else:
                        self._vlm_active_steps = set(all_triggers)
        else:
            for idx in env_indices:
                idx = int(idx)
                self._vlm_prev_images.pop(idx, None)
                self._vlm_frame_buffers.pop(idx, None)
                self._vlm_roboreward_frame_buffers.pop(idx, None)
        return extracted_obs, infos

    def _calc_step_reward(self, reward, info):
        """
        Compute combined env + VLM reward.

        Called from ManiskillEnv.step().
        """
        # Get original env reward
        env_reward = super()._calc_step_reward(reward, info)

        if not self.use_vlm_reward or not self._vlm_connected:
            return env_reward

        # Check call interval
        self._vlm_step_counter += 1
        if self.vlm_call_interval > 1 and (
            self._vlm_step_counter % self.vlm_call_interval != 0
        ):
            return env_reward

        # VLM rewards need observations - store for later
        # _calc_step_reward is called inside step() before we have extracted_obs
        # So we compute VLM reward in step() override instead
        return env_reward

    def step(
        self, actions: Union[Array, dict] = None, auto_reset=True
    ) -> tuple[Array, Array, Array, Array, dict]:
        """Step with VLM reward blending."""
        # Call parent step (which calls _calc_step_reward internally)
        extracted_obs, step_reward, terminations, truncations, infos = super().step(
            actions, auto_reset=auto_reset
        )

        # DEBUG: Log success info structure
        if self._vlm_log_enabled and self._vlm_step_counter % 10 == 0:
            try:
                has_success = "success" in infos
                has_final_info = "final_info" in infos
                has_final_obs = "final_observation" in infos
                debug_line = f"[VLM_DEBUG] step={self._vlm_step_counter}, auto_reset={auto_reset}, has_success={has_success}, has_final_info={has_final_info}, has_final_obs={has_final_obs}\n"
                if has_success and isinstance(infos.get("success"), torch.Tensor):
                    success_vals = infos["success"].cpu().numpy()
                    debug_line += f"  success values: {success_vals}\n"
                with open(self.vlm_log_file, "a") as f:
                    f.write(debug_line)
            except:
                pass

        if not self.use_vlm_reward or not self._vlm_connected:
            return extracted_obs, step_reward, terminations, truncations, infos

        is_trigger_step = (
            self.vlm_call_interval <= 1
            or self._vlm_step_counter % self.vlm_call_interval == 0
        )
        if self._vlm_active_steps:
            is_trigger_step = self._vlm_step_counter in self._vlm_active_steps

        if not is_trigger_step:
            if self.vlm_non_call_reward_mode == "zero":
                non_call_reward = torch.zeros_like(step_reward)
            elif self.vlm_non_call_reward_mode == "hold":
                non_call_reward = self._vlm_last_rewards.clone()
            else:
                non_call_reward = step_reward

            if self.record_metrics:
                self.returns += (non_call_reward - step_reward)
            infos["vlm_rewards"] = non_call_reward.clone()
            return extracted_obs, non_call_reward, terminations, truncations, infos

        # Compute VLM rewards using current observations
        vlm_rewards = self._compute_vlm_rewards(extracted_obs)
        self._vlm_last_rewards = vlm_rewards.clone()

        # Blend: (1 - w) * env_reward + w * vlm_reward
        if self.vlm_pure_reward:
            combined_reward = vlm_rewards
        else:
            combined_reward = (
                (1.0 - self.vlm_reward_weight) * step_reward
                + self.vlm_reward_weight * vlm_rewards
            )

        # Update metrics with combined reward
        if self.record_metrics:
            # Adjust returns to use combined reward instead of env-only reward
            self.returns += (combined_reward - step_reward)

        # Add VLM info
        infos["vlm_rewards"] = vlm_rewards.clone()
        self._log_openloop_step(
            infos=infos,
            step_reward=step_reward,
            vlm_rewards=vlm_rewards,
            terminations=terminations,
            truncations=truncations,
        )
        if self._vlm_success_source == "env":
            self._save_env_success_images(extracted_obs, infos)

        return extracted_obs, combined_reward, terminations, truncations, infos

    def _reset_metrics(self, env_idx=None):
        """Reset metrics including VLM state."""
        super()._reset_metrics(env_idx)
        self._vlm_step_counter = 0
        if env_idx is None:
            self._vlm_last_rewards[:] = 0.0
            self._vlm_prev_images = {}
            self._vlm_frame_buffers = {}
            self._vlm_roboreward_frame_buffers = {}
            self._vlm_active_steps = set()
            return
        if isinstance(env_idx, torch.Tensor):
            env_idx = env_idx.to(self.device)
            self._vlm_last_rewards[env_idx] = 0.0
            for idx in env_idx.detach().cpu().tolist():
                idx = int(idx)
                self._vlm_prev_images.pop(idx, None)
                self._vlm_frame_buffers.pop(idx, None)
                self._vlm_roboreward_frame_buffers.pop(idx, None)
        else:
            env_idx = np.asarray(env_idx).tolist()
            self._vlm_last_rewards[env_idx] = 0.0
            for idx in env_idx:
                idx = int(idx)
                self._vlm_prev_images.pop(idx, None)
                self._vlm_frame_buffers.pop(idx, None)
                self._vlm_roboreward_frame_buffers.pop(idx, None)
