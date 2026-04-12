#!/usr/bin/env python3
"""
Compute open-loop VLM reward metrics.

Auto-detects the scoring mode from vlm_scores.json and outputs mode-specific metrics:
  - progress / completion: ROC-AUC, Pairwise Acc (%), Global Pearson, Per-traj Pearson
  - comparison (contrastive): Direction Acc (%), Progress Recall (%), Monotonicity (success)

Usage:
    python compute_openloop_metrics.py \
        --scores_path vlm_scores.json \
        --output_dir metrics/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available. Correlation metrics will be skipped.")

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: sklearn not available. ROC-AUC will be skipped.")


def load_scores(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_trajectory_data(traj: dict) -> dict:
    oracle_steps = traj.get("oracle_steps", [])
    completion_scores = traj.get("completion_scores", [])
    comparison_scores = traj.get("comparison_scores", [])
    summary = traj.get("summary", {})

    oracle_rewards = [s.get("oracle_reward", 0.0) for s in oracle_steps]
    cumulative_oracle = [s.get("cumulative_reward", 0.0) for s in oracle_steps]
    if not cumulative_oracle or all(c == 0 for c in cumulative_oracle):
        cumulative_oracle = list(np.cumsum(oracle_rewards))

    success_flags = [s.get("success", False) for s in oracle_steps]
    episode_success = summary.get("success_once", any(success_flags))

    vlm_completion = []
    if completion_scores:
        for s in completion_scores:
            v = s.get("vlm_completion_score")
            if v is not None:
                vlm_completion.append(v)

    vlm_comparison = []
    if comparison_scores:
        for s in comparison_scores:
            v = s.get("vlm_comparison_score")
            if v is not None:
                vlm_comparison.append(v)

    return {
        "oracle_rewards": oracle_rewards,
        "cumulative_oracle": cumulative_oracle,
        "success_flags": success_flags,
        "episode_success": episode_success,
        "total_oracle_reward": summary.get("total_oracle_reward", cumulative_oracle[-1] if cumulative_oracle else 0),
        "vlm_completion": vlm_completion,
        "vlm_comparison": vlm_comparison,
        "num_steps": len(oracle_steps),
    }


# ============================================================
# Progress / Completion metrics
# ============================================================

def compute_progress_completion_metrics(trajs_data: list) -> dict:
    """ROC-AUC, Pairwise Acc (%), Global Pearson, Per-traj Pearson."""
    metrics = {}

    labels = [int(t["episode_success"]) for t in trajs_data]
    n_success = sum(labels)
    n_fail = len(labels) - n_success
    metrics["n_success"] = n_success
    metrics["n_failure"] = n_fail

    has_scores = all(len(t["vlm_completion"]) > 0 for t in trajs_data)
    if not has_scores:
        print("WARNING: No VLM completion/progress scores found.")
        return metrics

    final_scores = [t["vlm_completion"][-1] if t["vlm_completion"] else 0.0
                    for t in trajs_data]

    # ROC-AUC
    if HAS_SKLEARN and n_success > 0 and n_fail > 0:
        metrics["roc_auc"] = float(roc_auc_score(labels, final_scores))

    # Pairwise Acc (%)
    if n_success > 0 and n_fail > 0:
        correct = 0
        total = 0
        for i in range(len(trajs_data)):
            for j in range(i + 1, len(trajs_data)):
                if labels[i] != labels[j]:
                    total += 1
                    if labels[i] > labels[j]:
                        correct += int(final_scores[i] > final_scores[j])
                    else:
                        correct += int(final_scores[j] > final_scores[i])
        if total > 0:
            metrics["pairwise_acc_pct"] = float(correct / total * 100)

    # Global Pearson
    if HAS_SCIPY:
        total_vlm = [sum(t["vlm_completion"]) for t in trajs_data if t["vlm_completion"]]
        total_oracle = [t["total_oracle_reward"] for t in trajs_data if t["vlm_completion"]]
        if len(total_vlm) >= 3 and np.std(total_vlm) > 1e-8 and np.std(total_oracle) > 1e-8:
            r, _ = pearsonr(total_vlm, total_oracle)
            metrics["global_pearson"] = float(r)

    # Per-traj Pearson
    if HAS_SCIPY:
        pearson_list = []
        for t in trajs_data:
            if len(t["vlm_completion"]) < 3 or len(t["cumulative_oracle"]) < 3:
                continue
            vlm = np.array(t["vlm_completion"])
            oracle = np.array(t["cumulative_oracle"][:len(vlm)])
            if len(vlm) != len(oracle):
                min_len = min(len(vlm), len(oracle))
                vlm = vlm[:min_len]
                oracle = oracle[:min_len]
            if np.std(vlm) < 1e-8 or np.std(oracle) < 1e-8:
                continue
            r, _ = pearsonr(vlm, oracle)
            if not np.isnan(r):
                pearson_list.append(r)
        if pearson_list:
            metrics["per_traj_pearson_mean"] = float(np.mean(pearson_list))
            metrics["per_traj_pearson_std"] = float(np.std(pearson_list))

    return metrics


# ============================================================
# Contrastive (comparison) metrics
# ============================================================

def compute_contrastive_metrics(trajs_data: list) -> dict:
    """Direction Acc (%), Progress Recall (%), Monotonicity (success)."""
    metrics = {}

    labels = [int(t["episode_success"]) for t in trajs_data]
    n_success = sum(labels)
    n_fail = len(labels) - n_success
    metrics["n_success"] = n_success
    metrics["n_failure"] = n_fail

    has_scores = all(len(t["vlm_comparison"]) > 0 for t in trajs_data)
    if not has_scores:
        print("WARNING: No VLM comparison scores found.")
        return metrics

    # Direction Acc (%): fraction of comparison calls where VLM direction
    # matches oracle direction (both positive or both negative/zero)
    correct_dir = 0
    total_dir = 0
    for t in trajs_data:
        vlm = np.array(t["vlm_comparison"])
        oracle = np.array(t["oracle_rewards"][:len(vlm)])
        if len(vlm) != len(oracle):
            min_len = min(len(vlm), len(oracle))
            vlm = vlm[:min_len]
            oracle = oracle[:min_len]
        for v, o in zip(vlm, oracle):
            total_dir += 1
            if (v > 0 and o > 0) or (v < 0 and o < 0) or (v == 0 and o == 0):
                correct_dir += 1
    if total_dir > 0:
        metrics["direction_acc_pct"] = float(correct_dir / total_dir * 100)

    # Progress Recall (%): when oracle reward is positive (progress happened),
    # how often does VLM predict positive (+1)
    true_positive = 0
    total_positive = 0
    for t in trajs_data:
        vlm = np.array(t["vlm_comparison"])
        oracle = np.array(t["oracle_rewards"][:len(vlm)])
        if len(vlm) != len(oracle):
            min_len = min(len(vlm), len(oracle))
            vlm = vlm[:min_len]
            oracle = oracle[:min_len]
        for v, o in zip(vlm, oracle):
            if o > 0:
                total_positive += 1
                if v > 0:
                    true_positive += 1
    if total_positive > 0:
        metrics["progress_recall_pct"] = float(true_positive / total_positive * 100)

    # Monotonicity (success): for successful episodes, check if cumulative
    # VLM comparison score is monotonically increasing (measuring temporal consistency)
    mono_scores = []
    for t in trajs_data:
        if not t["episode_success"]:
            continue
        vlm = np.array(t["vlm_comparison"])
        if len(vlm) < 2:
            continue
        cum_vlm = np.cumsum(vlm)
        diffs = np.diff(cum_vlm)
        # Monotonicity = fraction of non-negative increments
        mono = float(np.sum(diffs >= 0) / len(diffs))
        mono_scores.append(mono)
    if mono_scores:
        metrics["monotonicity_success"] = float(np.mean(mono_scores))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute open-loop VLM reward metrics")
    parser.add_argument("--scores_path", type=str, nargs="+", required=True,
                        help="Path(s) to vlm_scores.json (multiple paths are merged)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for metrics")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.scores_path[0]), "metrics")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and merge trajectories
    all_trajectories = []
    detected_mode = None
    for path in args.scores_path:
        print(f"Loading scores from {path}")
        data = load_scores(path)
        trajs = data.get("trajectories", [])
        print(f"  {len(trajs)} trajectories")
        all_trajectories.extend(trajs)
        if detected_mode is None:
            detected_mode = data.get("mode", "")
    print(f"Total trajectories: {len(all_trajectories)}")
    print(f"Detected mode: {detected_mode}")

    trajs_data = [extract_trajectory_data(t) for t in all_trajectories]

    n_success = sum(1 for t in trajs_data if t["episode_success"])
    n_failure = len(trajs_data) - n_success
    print(f"Success: {n_success}, Failure: {n_failure}")

    # Compute mode-specific metrics
    if detected_mode == "comparison":
        print("\n--- Contrastive (Comparison) Metrics ---")
        mode_metrics = compute_contrastive_metrics(trajs_data)
    else:
        print(f"\n--- Progress/Completion Metrics (mode={detected_mode}) ---")
        mode_metrics = compute_progress_completion_metrics(trajs_data)

    for k, v in mode_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    all_metrics = {
        "mode": detected_mode,
        "metrics": mode_metrics,
        "meta": {
            "scores_paths": args.scores_path,
            "n_trajectories": len(trajs_data),
            "n_success": n_success,
            "n_failure": n_failure,
        },
    }

    metrics_path = os.path.join(args.output_dir, "openloop_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
