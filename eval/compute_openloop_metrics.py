#!/usr/bin/env python3
"""
Compute open-loop VLM reward metrics.

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
    from scipy.stats import pearsonr, spearmanr, kendalltau
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available. Correlation metrics will be skipped.")

try:
    from sklearn.metrics import roc_auc_score, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: sklearn not available. ROC-AUC metrics will be skipped.")


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


def compute_discriminative_metrics(trajs_data: list) -> dict:
    """Success/failure classification metrics."""
    metrics = {}

    labels = [int(t["episode_success"]) for t in trajs_data]
    n_success = sum(labels)
    n_fail = len(labels) - n_success
    metrics["n_success"] = n_success
    metrics["n_failure"] = n_fail

    if n_success == 0 or n_fail == 0:
        print(f"WARNING: Cannot compute discriminative metrics - "
              f"{n_success} success, {n_fail} failure episodes")
        return metrics

    has_comparison = all(len(t["vlm_comparison"]) > 0 for t in trajs_data)
    if has_comparison:
        total_comp_scores = [sum(t["vlm_comparison"]) for t in trajs_data]

        if HAS_SKLEARN:
            metrics["comparison_roc_auc"] = roc_auc_score(labels, total_comp_scores)

            best_acc = 0.0
            score_range = np.linspace(min(total_comp_scores) - 1, max(total_comp_scores) + 1, 100)
            for thresh in score_range:
                preds = [int(s >= thresh) for s in total_comp_scores]
                acc = accuracy_score(labels, preds)
                if acc > best_acc:
                    best_acc = acc
            metrics["comparison_best_accuracy"] = best_acc

        correct = 0
        total = 0
        for i in range(len(trajs_data)):
            for j in range(i + 1, len(trajs_data)):
                if labels[i] != labels[j]:
                    total += 1
                    if labels[i] > labels[j]:
                        correct += int(total_comp_scores[i] > total_comp_scores[j])
                    else:
                        correct += int(total_comp_scores[j] > total_comp_scores[i])
        if total > 0:
            metrics["comparison_pairwise_accuracy"] = correct / total

        success_scores = [s for s, l in zip(total_comp_scores, labels) if l == 1]
        failure_scores = [s for s, l in zip(total_comp_scores, labels) if l == 0]
        metrics["comparison_mean_success"] = float(np.mean(success_scores))
        metrics["comparison_mean_failure"] = float(np.mean(failure_scores))

    has_completion = all(len(t["vlm_completion"]) > 0 for t in trajs_data)
    if has_completion:
        final_scores = [t["vlm_completion"][-1] if t["vlm_completion"] else 0.0
                        for t in trajs_data]

        if HAS_SKLEARN:
            metrics["completion_roc_auc"] = roc_auc_score(labels, final_scores)

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
            metrics["completion_pairwise_accuracy"] = correct / total

    return metrics


def compute_correlation_metrics(trajs_data: list) -> dict:
    """Correlation between VLM scores and oracle rewards."""
    metrics = {}
    if not HAS_SCIPY:
        return metrics

    # Comparison mode: per-trajectory correlation
    pearson_comp_list = []
    spearman_comp_list = []
    kendall_comp_list = []
    for t in trajs_data:
        if len(t["vlm_comparison"]) < 3:
            continue
        vlm_delta = np.array(t["vlm_comparison"])
        oracle = np.array(t["oracle_rewards"][:len(vlm_delta)])

        if len(vlm_delta) != len(oracle):
            min_len = min(len(vlm_delta), len(oracle))
            vlm_delta = vlm_delta[:min_len]
            oracle = oracle[:min_len]

        if np.std(vlm_delta) < 1e-8 or np.std(oracle) < 1e-8:
            continue

        r_p, _ = pearsonr(vlm_delta, oracle)
        r_s, _ = spearmanr(vlm_delta, oracle)
        r_k, _ = kendalltau(vlm_delta, oracle)
        if not np.isnan(r_p):
            pearson_comp_list.append(r_p)
        if not np.isnan(r_s):
            spearman_comp_list.append(r_s)
        if not np.isnan(r_k):
            kendall_comp_list.append(r_k)

    if pearson_comp_list:
        metrics["comparison_pearson_mean"] = float(np.mean(pearson_comp_list))
        metrics["comparison_pearson_std"] = float(np.std(pearson_comp_list))
    if spearman_comp_list:
        metrics["comparison_spearman_mean"] = float(np.mean(spearman_comp_list))
        metrics["comparison_spearman_std"] = float(np.std(spearman_comp_list))
    if kendall_comp_list:
        metrics["comparison_kendall_mean"] = float(np.mean(kendall_comp_list))
        metrics["comparison_kendall_std"] = float(np.std(kendall_comp_list))

    # Global correlation (comparison)
    total_comp = [sum(t["vlm_comparison"]) for t in trajs_data if t["vlm_comparison"]]
    total_oracle_comp = [t["total_oracle_reward"] for t in trajs_data if t["vlm_comparison"]]
    if len(total_comp) >= 3 and np.std(total_comp) > 1e-8 and np.std(total_oracle_comp) > 1e-8:
        r_global_comp, _ = pearsonr(total_comp, total_oracle_comp)
        metrics["comparison_global_pearson"] = float(r_global_comp)

    # Completion mode: per-trajectory correlation
    pearson_list = []
    spearman_list = []
    kendall_list = []
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

        r_p, _ = pearsonr(vlm, oracle)
        r_s, _ = spearmanr(vlm, oracle)
        r_k, _ = kendalltau(vlm, oracle)
        if not np.isnan(r_p):
            pearson_list.append(r_p)
        if not np.isnan(r_s):
            spearman_list.append(r_s)
        if not np.isnan(r_k):
            kendall_list.append(r_k)

    if pearson_list:
        metrics["completion_pearson_mean"] = float(np.mean(pearson_list))
        metrics["completion_pearson_std"] = float(np.std(pearson_list))
    if spearman_list:
        metrics["completion_spearman_mean"] = float(np.mean(spearman_list))
        metrics["completion_spearman_std"] = float(np.std(spearman_list))
    if kendall_list:
        metrics["completion_kendall_mean"] = float(np.mean(kendall_list))
        metrics["completion_kendall_std"] = float(np.std(kendall_list))

    # Global correlation (completion)
    total_vlm = [sum(t["vlm_completion"]) for t in trajs_data if t["vlm_completion"]]
    total_oracle = [t["total_oracle_reward"] for t in trajs_data if t["vlm_completion"]]
    if len(total_vlm) >= 3 and np.std(total_vlm) > 1e-8 and np.std(total_oracle) > 1e-8:
        r_global, _ = pearsonr(total_vlm, total_oracle)
        metrics["global_pearson"] = float(r_global)

    return metrics


def compute_temporal_metrics(trajs_data: list) -> dict:
    """Temporal consistency metrics."""
    metrics = {}

    comp_variances = []
    comp_jump_freqs = []
    comp_positive_fracs = []
    comp_jump_threshold = 1.0

    for t in trajs_data:
        vlm = np.array(t["vlm_comparison"])
        if len(vlm) < 2:
            continue

        comp_variances.append(float(np.var(vlm)))
        diffs = np.abs(np.diff(vlm))
        jumps = np.sum(diffs > comp_jump_threshold)
        comp_jump_freqs.append(float(jumps / max(1, len(diffs))))

        n_positive = np.sum(vlm > 0)
        comp_positive_fracs.append(float(n_positive / len(vlm)))

    if comp_variances:
        metrics["comparison_variance_mean"] = float(np.mean(comp_variances))
    if comp_jump_freqs:
        metrics["comparison_jump_freq_mean"] = float(np.mean(comp_jump_freqs))
    if comp_positive_fracs:
        metrics["comparison_positive_frac_mean"] = float(np.mean(comp_positive_fracs))

    # Positive fraction for success vs failure
    success_pos_frac = []
    failure_pos_frac = []
    for t in trajs_data:
        vlm = np.array(t["vlm_comparison"])
        if len(vlm) < 2:
            continue
        frac = float(np.sum(vlm > 0) / len(vlm))
        if t["episode_success"]:
            success_pos_frac.append(frac)
        else:
            failure_pos_frac.append(frac)
    if success_pos_frac:
        metrics["comparison_positive_frac_success"] = float(np.mean(success_pos_frac))
    if failure_pos_frac:
        metrics["comparison_positive_frac_failure"] = float(np.mean(failure_pos_frac))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute open-loop VLM reward metrics")
    parser.add_argument("--scores_path", type=str, required=True,
                        help="Path to vlm_scores.json from score_with_vlm.py")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for metrics")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.scores_path), "metrics")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading scores from {args.scores_path}")
    data = load_scores(args.scores_path)
    trajectories = data.get("trajectories", [])
    print(f"Loaded {len(trajectories)} trajectories")

    trajs_data = [extract_trajectory_data(t) for t in trajectories]

    n_success = sum(1 for t in trajs_data if t["episode_success"])
    n_failure = len(trajs_data) - n_success
    print(f"Success: {n_success}, Failure: {n_failure}")

    print("\n--- Discriminative Accuracy ---")
    disc_metrics = compute_discriminative_metrics(trajs_data)
    for k, v in disc_metrics.items():
        print(f"  {k}: {v}")

    print("\n--- Correlation ---")
    corr_metrics = compute_correlation_metrics(trajs_data)
    for k, v in corr_metrics.items():
        print(f"  {k}: {v}")

    print("\n--- Temporal Consistency ---")
    temp_metrics = compute_temporal_metrics(trajs_data)
    for k, v in temp_metrics.items():
        print(f"  {k}: {v}")

    all_metrics = {
        "discriminative": disc_metrics,
        "correlation": corr_metrics,
        "temporal": temp_metrics,
        "meta": {
            "scores_path": args.scores_path,
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
