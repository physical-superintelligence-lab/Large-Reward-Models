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

"""ManiskillEnv subclass that records trajectories for open-loop VLM evaluation.

Recorded data layout (compatible with eval/score_with_vlm.py):

  <record_output_dir>/worker_<rank>/trajectories/traj_<env>_<episode>/
      metadata.jsonl   - one JSON line per step: {t, oracle_reward, cumulative_reward, success}
      summary.json     - {success_once, total_oracle_reward}
      frame_000.png    - RGB frame at t=0 (state after action)
      frame_001.png
      ...
"""

import json
import os
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
from mani_skill.utils.structs.types import Array

__all__ = ["RecordingManiskillEnv"]


class RecordingManiskillEnv(ManiskillEnv):
    """ManiskillEnv with per-trajectory recording for offline VLM scoring."""

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info,
        record_metrics: bool = True,
    ):
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics=False)

        self.record_trajectories: bool = getattr(cfg, "record_trajectories", True)
        record_output_dir: str = getattr(cfg, "record_output_dir", "./logs/openloop_data/default")

        # Worker rank -> per-worker sub-directory
        rank: int = worker_info.rank if worker_info is not None else 0
        self._worker_rank = rank
        self._worker_output_dir = os.path.join(record_output_dir, f"worker_{rank}")

        # How many of this worker's envs to record
        total_record_count: int = getattr(cfg, "record_env_count", num_envs)
        per_worker_count = max(1, total_record_count // max(1, total_num_processes))
        self._record_num_envs = min(per_worker_count, num_envs)

        # Per-env recording state (only for the first _record_num_envs envs)
        self._step_counters = [0] * self._record_num_envs
        self._step_data: list[list[dict]] = [[] for _ in range(self._record_num_envs)]
        self._cumulative_rewards = [0.0] * self._record_num_envs
        self._ep_success_once = [False] * self._record_num_envs
        self._traj_counters = [0] * self._record_num_envs  # episode index per env

    def _record_metrics(self, step_reward, infos):
        infos["episode"] = {}
        return infos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_frames(self, obs: dict) -> Optional[np.ndarray]:
        """Return [B, H, W, C] uint8 numpy array from obs, or None."""
        if "main_images" in obs:
            imgs = obs["main_images"]  # [B, H, W, C] uint8 tensor
            return imgs.cpu().numpy()
        return None

    def _traj_dir(self, env_idx: int) -> str:
        episode = self._traj_counters[env_idx]
        global_env = self._worker_rank * self.num_envs + env_idx
        return os.path.join(
            self._worker_output_dir,
            "trajectories",
            f"traj_{global_env:04d}_{episode:02d}",
        )

    def _write_frame(self, env_idx: int, t: int, frame: np.ndarray) -> None:
        traj_dir = self._traj_dir(env_idx)
        os.makedirs(traj_dir, exist_ok=True)
        Image.fromarray(frame).save(os.path.join(traj_dir, f"frame_{t:03d}.png"))

    def _flush_trajectory(self, env_idx: int) -> None:
        """Persist buffered step data to disk, then reset buffers for this env."""
        steps = self._step_data[env_idx]
        if not steps:
            return

        traj_dir = self._traj_dir(env_idx)
        os.makedirs(traj_dir, exist_ok=True)

        # metadata.jsonl
        with open(os.path.join(traj_dir, "metadata.jsonl"), "w") as f:
            for step in steps:
                f.write(json.dumps(step) + "\n")

        # summary.json
        total_oracle_reward = steps[-1]["cumulative_reward"]
        summary = {
            "success_once": self._ep_success_once[env_idx],
            "total_oracle_reward": total_oracle_reward,
        }
        with open(os.path.join(traj_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Reset buffers
        self._step_data[env_idx] = []
        self._cumulative_rewards[env_idx] = 0.0
        self._ep_success_once[env_idx] = False
        self._step_counters[env_idx] = 0
        self._traj_counters[env_idx] += 1

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def step(
        self, actions: Union[Array, dict] = None, auto_reset: bool = True
    ) -> tuple[Array, Array, Array, Array, dict]:
        # Call parent with auto_reset=False so we control auto-reset ourselves
        obs, step_reward, terminations, truncations, infos = super().step(
            actions, auto_reset=False
        )

        if self.record_trajectories:
            frames = self._get_frames(obs)
            rew_np = step_reward.cpu().numpy()
            success_tensor = infos.get(
                "success",
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            )
            success_np = success_tensor.cpu().numpy()

            for i in range(self._record_num_envs):
                t = self._step_counters[i]
                self._cumulative_rewards[i] += float(rew_np[i])
                self._ep_success_once[i] = self._ep_success_once[i] or bool(success_np[i])

                self._step_data[i].append(
                    {
                        "t": t,
                        "oracle_reward": float(rew_np[i]),
                        "cumulative_reward": float(self._cumulative_rewards[i]),
                        "success": bool(success_np[i]),
                    }
                )

                if frames is not None:
                    self._write_frame(i, t, frames[i])

                self._step_counters[i] += 1

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return obs, step_reward, terminations, truncations, infos

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        """Flush completed trajectories before resetting done envs."""
        if self.record_trajectories:
            done_np = dones.cpu().numpy()
            for i in range(self._record_num_envs):
                if done_np[i]:
                    self._flush_trajectory(i)

        return super()._handle_auto_reset(dones, extracted_obs, infos)
