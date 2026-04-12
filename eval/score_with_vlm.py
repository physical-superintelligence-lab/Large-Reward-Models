#!/usr/bin/env python3
"""
Score recorded trajectories with VLM reward server.

Loads trajectory data saved by RecordingManiskillEnv and calls the VLM reward
server (Qwen3-VL) to compute completion and comparison scores for each frame.

Usage:
    python scripts/score_with_vlm.py \
        --data_dir logs/openloop_data/sft_baseline/worker_0 \
        --vlm_url http://localhost:5001 \
        --output vlm_scores.json \
        --mode both
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add RLinf root to path for VLMRewardClient
# install.sh copies this file to <rlinf>/examples/embodiment/eval/
# so we resolve: eval/ -> embodiment/ -> examples/ -> rlinf_root -> rlinf/envs/maniskill/
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_EMBODIED_DIR = os.path.dirname(_EVAL_DIR)
_RLINF_ROOT = os.path.dirname(os.path.dirname(_EMBODIED_DIR))
sys.path.insert(0, os.path.join(_RLINF_ROOT, "rlinf", "envs", "maniskill"))
from vlm_reward_client import VLMRewardClient


def load_trajectory(traj_dir: str) -> dict:
    """Load a single trajectory from disk."""
    traj_dir = Path(traj_dir)

    # Load metadata
    metadata_path = traj_dir / "metadata.jsonl"
    steps = []
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))

    # Load summary
    summary_path = traj_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    # List available frames
    frame_paths = {}
    for step in steps:
        t = step["t"]
        frame_path = traj_dir / f"frame_{t:03d}.png"
        if frame_path.exists():
            frame_paths[t] = str(frame_path)

    return {
        "traj_dir": str(traj_dir),
        "steps": steps,
        "summary": summary,
        "frame_paths": frame_paths,
    }


def load_all_trajectories(data_dir: str) -> list:
    """Load all trajectories from the data directory."""
    data_dir = Path(data_dir)
    traj_base = data_dir / "trajectories"

    if not traj_base.exists():
        print(f"No trajectories directory found at {traj_base}")
        return []

    traj_dirs = sorted(traj_base.iterdir())
    trajectories = []
    for traj_dir in traj_dirs:
        if traj_dir.is_dir() and (traj_dir / "metadata.jsonl").exists():
            traj = load_trajectory(traj_dir)
            trajectories.append(traj)

    print(f"Loaded {len(trajectories)} trajectories from {data_dir}")
    return trajectories


def score_trajectory_completion(
    traj: dict,
    client: VLMRewardClient,
    task_description: str,
) -> list:
    """Score all frames in a trajectory using completion mode."""
    scores = []
    for step in traj["steps"]:
        t = step["t"]
        frame_path = traj["frame_paths"].get(t)
        if frame_path is None:
            scores.append({"t": t, "vlm_score": None, "response": ""})
            continue

        # Load image
        try:
            img = np.array(Image.open(frame_path).convert("RGB"))
        except (OSError, Exception) as e:
            print(f"  Warning: skipping corrupted frame t={t}: {e}")
            scores.append({"t": t, "vlm_completion_score": None, "response": "corrupted_frame"})
            continue

        result = client.compute_reward(img, task_description, reward_type="progress")
        scores.append({
            "t": t,
            "vlm_completion_score": result.get("score", 0.0),
            "response": result.get("response", ""),
            "success": result.get("success", False),
        })

    return scores


def score_trajectory_comparison(
    traj: dict,
    client: VLMRewardClient,
    task_description: str,
    comparison_interval: int = 1,
) -> list:
    """Score frame pairs using comparison mode with configurable interval.

    Matches RL training behavior where VLM is called every `comparison_interval`
    steps, comparing the current frame with the frame from `comparison_interval`
    steps ago.
    """
    scores = []
    sorted_steps = sorted(traj["steps"], key=lambda s: s["t"])

    # Build timestep -> frame_path lookup and load images lazily
    step_lookup = {s["t"]: s for s in sorted_steps}
    timesteps = [s["t"] for s in sorted_steps]

    if not timesteps:
        return scores

    # First frame: no comparison possible
    first_t = timesteps[0]
    scores.append({"t": first_t, "vlm_comparison_score": 0.0, "result": "first_frame"})

    # Compare every `comparison_interval` steps (matching RL training vlm_call_interval)
    for i in range(comparison_interval, len(timesteps), comparison_interval):
        curr_t = timesteps[i]
        prev_t = timesteps[i - comparison_interval]

        curr_path = traj["frame_paths"].get(curr_t)
        prev_path = traj["frame_paths"].get(prev_t)
        if curr_path is None or prev_path is None:
            scores.append({"t": curr_t, "vlm_comparison_score": None})
            continue

        try:
            curr_img = np.array(Image.open(curr_path).convert("RGB"))
        except (OSError, Exception) as e:
            print(f"  Warning: skipping corrupted frame t={curr_t}: {e}")
            scores.append({"t": curr_t, "vlm_comparison_score": None})
            continue

        try:
            prev_img = np.array(Image.open(prev_path).convert("RGB"))
        except (OSError, Exception) as e:
            print(f"  Warning: skipping corrupted frame t={prev_t}: {e}")
            scores.append({"t": curr_t, "vlm_comparison_score": None})
            continue

        result = client.compute_comparison(prev_img, curr_img, task_description)
        scores.append({
            "t": curr_t,
            "vlm_comparison_score": result.get("score", 0.0),
            "result": result.get("result", "N/A"),
            "response": result.get("response", ""),
        })

    return scores


def score_trajectory_progress(
    traj: dict,
    client: VLMRewardClient,
    task_description: str,
    call_interval: int = 1,
) -> list:
    """Score frames every call_interval steps using single-frame progress mode (0.X or 1.0)."""
    scores = []
    sorted_steps = sorted(traj["steps"], key=lambda s: s["t"])
    timesteps = [s["t"] for s in sorted_steps]

    if not timesteps:
        return scores

    for i, t in enumerate(timesteps):
        is_last = (i == len(timesteps) - 1)
        if call_interval > 1 and (i % call_interval != 0) and not is_last:
            continue

        frame_path = traj["frame_paths"].get(t)
        if frame_path is None:
            scores.append({"t": t, "vlm_completion_score": None, "response": ""})
            continue

        try:
            img = np.array(Image.open(frame_path).convert("RGB"))
        except (OSError, Exception) as e:
            print(f"  Warning: skipping corrupted frame t={t}: {e}")
            scores.append({"t": t, "vlm_completion_score": None, "response": "corrupted_frame"})
            continue

        result = client.compute_reward(img, task_description, reward_type="progress")
        scores.append({
            "t": t,
            "vlm_completion_score": result.get("score", 0.0),
            "response": result.get("response", ""),
            "success": result.get("success", False),
        })

    return scores


def score_single_trajectory(
    traj: dict,
    client: VLMRewardClient,
    task_description: str,
    mode: str,
    comparison_interval: int = 1,
    call_interval: int = 1,
) -> dict:
    """Score a single trajectory with completion and/or comparison."""
    result = {
        "traj_dir": traj["traj_dir"],
        "summary": traj["summary"],
        "oracle_steps": traj["steps"],
    }

    if mode in ("completion", "both"):
        result["completion_scores"] = score_trajectory_completion(
            traj, client, task_description
        )

    if mode in ("comparison", "both"):
        result["comparison_scores"] = score_trajectory_comparison(
            traj, client, task_description, comparison_interval
        )

    if mode == "progress":
        result["completion_scores"] = score_trajectory_progress(
            traj, client, task_description, call_interval
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="Score trajectories with VLM reward server")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to recorded trajectory data (worker_X directory)")
    parser.add_argument("--vlm_url", type=str, default="http://localhost:5002",
                        help="VLM reward server URL")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <data_dir>/vlm_scores.json)")
    parser.add_argument("--mode", type=str, default="completion",
                        choices=["completion", "comparison", "both", "progress"],
                        help="Scoring mode")
    parser.add_argument("--task_description", type=str,
                        default="Pick up the object and place it on the plate",
                        help="Task description for VLM")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Number of parallel workers (1 = sequential)")
    parser.add_argument("--max_trajs", type=int, default=0,
                        help="Max trajectories to score (0 = all)")
    parser.add_argument("--comparison_interval", type=int, default=1,
                        help="Frame interval for comparison mode (default: 1, score every step)")
    parser.add_argument("--call_interval", type=int, default=1,
                        help="Frame interval for progress mode: score every N steps (default: 1)")
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = os.path.join(args.data_dir, "vlm_scores.json")

    # Load trajectories
    trajectories = load_all_trajectories(args.data_dir)
    if not trajectories:
        print("No trajectories found. Exiting.")
        return

    if args.max_trajs > 0:
        trajectories = trajectories[:args.max_trajs]
        print(f"Limiting to {len(trajectories)} trajectories")

    # Initialize VLM client
    client = VLMRewardClient(server_url=args.vlm_url, timeout=60.0)
    print(f"Checking VLM server at {args.vlm_url}...")
    if not client.health_check():
        print("ERROR: VLM server not available. Start it first.")
        return
    print("VLM server is healthy.")

    # Score trajectories
    all_results = []
    total_frames = sum(len(t["steps"]) for t in trajectories)
    if args.mode == "comparison":
        total_vlm_calls = sum(max(1, len(t["steps"]) // args.comparison_interval) for t in trajectories)
    elif args.mode == "progress":
        total_vlm_calls = sum(max(1, len(t["steps"]) // args.call_interval) for t in trajectories)
    elif args.mode == "completion":
        total_vlm_calls = total_frames
    else:
        total_vlm_calls = total_frames + sum(max(0, len(t["steps"]) // args.comparison_interval) for t in trajectories)
    print(f"\nScoring {len(trajectories)} trajectories ({total_frames} total frames)")
    print(f"Mode: {args.mode}, comparison_interval: {args.comparison_interval}, call_interval: {args.call_interval}")
    print(f"Total VLM calls: ~{total_vlm_calls}")
    print(f"Estimated time: ~{total_vlm_calls * 3.0 / 60:.1f} min")

    start_time = time.time()

    if args.max_workers <= 1:
        # Sequential scoring
        for i, traj in enumerate(tqdm(trajectories, desc="Scoring trajectories")):
            result = score_single_trajectory(traj, client, args.task_description, args.mode, args.comparison_interval, args.call_interval)
            all_results.append(result)

            # Save intermediate results every 10 trajectories
            if (i + 1) % 10 == 0:
                _save_results(all_results, args.output + ".partial", args)
    else:
        # Parallel scoring
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    score_single_trajectory, traj, client, args.task_description, args.mode, args.comparison_interval, args.call_interval
                ): i
                for i, traj in enumerate(trajectories)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
                result = future.result()
                all_results.append(result)

    elapsed = time.time() - start_time
    print(f"\nScoring complete in {elapsed:.1f}s ({elapsed / len(trajectories):.2f}s per trajectory)")

    # Save results
    _save_results(all_results, args.output, args)
    print(f"Results saved to {args.output}")

    # Clean up partial file
    partial_path = args.output + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)


def _save_results(results: list, output_path: str, args):
    """Save scoring results to JSON."""
    output = {
        "vlm_url": args.vlm_url,
        "mode": args.mode,
        "task_description": args.task_description,
        "num_trajectories": len(results),
        "trajectories": results,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
