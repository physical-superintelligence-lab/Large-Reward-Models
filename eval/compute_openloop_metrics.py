#!/usr/bin/env python3
"""
Compute open-loop VLM reward metrics and generate plots for paper.

Reads VLM scoring results (from score_with_vlm.py) and computes:
1. Reward Discriminative Accuracy (ROC-AUC, pairwise ranking, binary accuracy)
2. Correlation with Ground Truth (Pearson, Spearman - per-trajectory and global)
3. Temporal Consistency (variance, jump frequency, monotonicity)

Usage:
    python scripts/compute_openloop_metrics.py \
        --scores_path logs/openloop_data/sft_baseline/worker_0/vlm_scores.json \
        --output_dir logs/openloop_data/sft_baseline/metrics
"""

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr, kendalltau
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available. Correlation metrics will be skipped.")

try:
    from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: sklearn not available. ROC-AUC metrics will be skipped.")


# ============================================================
# Data Loading
# ============================================================

def load_scores(path: str) -> dict:
    """Load VLM scoring results."""
    with open(path) as f:
        return json.load(f)


def extract_trajectory_data(traj: dict) -> dict:
    """Extract aligned oracle and VLM data from a trajectory."""
    oracle_steps = traj.get("oracle_steps", [])
    completion_scores = traj.get("completion_scores", [])
    comparison_scores = traj.get("comparison_scores", [])
    summary = traj.get("summary", {})

    # Oracle data
    oracle_rewards = [s.get("oracle_reward", 0.0) for s in oracle_steps]
    cumulative_oracle = [s.get("cumulative_reward", 0.0) for s in oracle_steps]
    if not cumulative_oracle or all(c == 0 for c in cumulative_oracle):
        cumulative_oracle = list(np.cumsum(oracle_rewards))

    is_grasped = [s.get("is_grasped", False) for s in oracle_steps]
    success_flags = [s.get("success", False) for s in oracle_steps]

    # Episode-level success
    episode_success = summary.get("success_once", any(success_flags))

    # VLM completion scores (filter out None from corrupted/failed frames)
    vlm_completion = []
    if completion_scores:
        for s in completion_scores:
            v = s.get("vlm_completion_score")
            if v is not None:
                vlm_completion.append(v)

    # VLM comparison scores (filter out None from corrupted/failed frames)
    vlm_comparison = []
    if comparison_scores:
        for s in comparison_scores:
            v = s.get("vlm_comparison_score")
            if v is not None:
                vlm_comparison.append(v)

    return {
        "oracle_rewards": oracle_rewards,
        "cumulative_oracle": cumulative_oracle,
        "is_grasped": is_grasped,
        "success_flags": success_flags,
        "episode_success": episode_success,
        "total_oracle_reward": summary.get("total_oracle_reward", cumulative_oracle[-1] if cumulative_oracle else 0),
        "vlm_completion": vlm_completion,
        "vlm_comparison": vlm_comparison,
        "num_steps": len(oracle_steps),
    }


# ============================================================
# Metric 1: Reward Discriminative Accuracy
# ============================================================

def compute_discriminative_metrics(trajs_data: list) -> dict:
    """Compute success/failure classification metrics."""
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

    # Use total comparison score as classifier (primary metric for comparison-trained model)
    has_comparison = all(len(t["vlm_comparison"]) > 0 for t in trajs_data)
    if has_comparison:
        total_comp_scores = [sum(t["vlm_comparison"]) for t in trajs_data]

        if HAS_SKLEARN:
            metrics["comparison_roc_auc"] = roc_auc_score(labels, total_comp_scores)

            # Best threshold accuracy
            best_acc = 0.0
            best_thresh = 0.0
            score_range = np.linspace(min(total_comp_scores) - 1, max(total_comp_scores) + 1, 100)
            for thresh in score_range:
                preds = [int(s >= thresh) for s in total_comp_scores]
                acc = accuracy_score(labels, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = thresh
            metrics["comparison_best_accuracy"] = best_acc
            metrics["comparison_best_threshold"] = best_thresh

        # Pairwise ranking
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

        # Mean total comparison score for success vs failure
        success_scores = [s for s, l in zip(total_comp_scores, labels) if l == 1]
        failure_scores = [s for s, l in zip(total_comp_scores, labels) if l == 0]
        metrics["comparison_mean_success"] = float(np.mean(success_scores))
        metrics["comparison_mean_failure"] = float(np.mean(failure_scores))

    # Also compute completion metrics if available
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


# ============================================================
# Metric 2: Correlation with Ground Truth
# ============================================================

def compute_correlation_metrics(trajs_data: list) -> dict:
    """Compute correlation between VLM scores and oracle rewards."""
    metrics = {}
    if not HAS_SCIPY:
        return metrics

    # Per-trajectory correlation (comparison mode vs oracle delta) - PRIMARY
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

    # Global correlation: total comparison score vs total oracle reward
    total_comp = [sum(t["vlm_comparison"]) for t in trajs_data if t["vlm_comparison"]]
    total_oracle_comp = [t["total_oracle_reward"] for t in trajs_data if t["vlm_comparison"]]
    if len(total_comp) >= 3 and np.std(total_comp) > 1e-8 and np.std(total_oracle_comp) > 1e-8:
        r_global_comp, _ = pearsonr(total_comp, total_oracle_comp)
        metrics["comparison_global_pearson"] = float(r_global_comp)
        r_global_comp_k, _ = kendalltau(total_comp, total_oracle_comp)
        metrics["comparison_global_kendall"] = float(r_global_comp_k)

    # Per-trajectory correlation (completion mode, if available)
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
        metrics["completion_pearson_median"] = float(np.median(pearson_list))
    if spearman_list:
        metrics["completion_spearman_mean"] = float(np.mean(spearman_list))
        metrics["completion_spearman_std"] = float(np.std(spearman_list))
    if kendall_list:
        metrics["completion_kendall_mean"] = float(np.mean(kendall_list))
        metrics["completion_kendall_std"] = float(np.std(kendall_list))
        metrics["completion_kendall_median"] = float(np.median(kendall_list))

    # Global correlation (completion, if available)
    total_vlm = [sum(t["vlm_completion"]) for t in trajs_data if t["vlm_completion"]]
    total_oracle = [t["total_oracle_reward"] for t in trajs_data if t["vlm_completion"]]
    if len(total_vlm) >= 3 and np.std(total_vlm) > 1e-8 and np.std(total_oracle) > 1e-8:
        r_global, _ = pearsonr(total_vlm, total_oracle)
        metrics["global_pearson"] = float(r_global)
        r_global_k, _ = kendalltau(total_vlm, total_oracle)
        metrics["global_kendall"] = float(r_global_k)

    return metrics


# ============================================================
# Metric 3: Temporal Consistency
# ============================================================

def compute_temporal_metrics(trajs_data: list) -> dict:
    """Compute temporal consistency metrics for VLM rewards."""
    metrics = {}

    # Comparison mode metrics (PRIMARY for comparison-trained model)
    comp_variances = []
    comp_jump_freqs = []
    comp_positive_fracs = []
    comp_jump_threshold = 1.0  # comparison scores are +1/-1

    for t in trajs_data:
        vlm = np.array(t["vlm_comparison"])
        if len(vlm) < 2:
            continue

        comp_variances.append(float(np.var(vlm)))
        diffs = np.abs(np.diff(vlm))
        jumps = np.sum(diffs > comp_jump_threshold)
        comp_jump_freqs.append(float(jumps / max(1, len(diffs))))

        # Fraction of positive comparisons (progress indicator)
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


# ============================================================
# Visualization
# ============================================================

def plot_roc_curve(trajs_data: list, output_dir: str):
    """Plot ROC curve for success/failure classification."""
    if not HAS_SKLEARN:
        return

    labels = [int(t["episode_success"]) for t in trajs_data]
    if sum(labels) == 0 or sum(labels) == len(labels):
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    has_completion = all(len(t["vlm_completion"]) > 0 for t in trajs_data)
    if has_completion:
        final_scores = [t["vlm_completion"][-1] if t["vlm_completion"] else 0.0
                        for t in trajs_data]
        fpr, tpr, _ = roc_curve(labels, final_scores)
        auc = roc_auc_score(labels, final_scores)
        ax.plot(fpr, tpr, label=f"Completion (AUC={auc:.3f})", linewidth=2)

    has_comparison = all(len(t["vlm_comparison"]) > 0 for t in trajs_data)
    if has_comparison:
        total_scores = [sum(t["vlm_comparison"]) for t in trajs_data]
        fpr, tpr, _ = roc_curve(labels, total_scores)
        auc = roc_auc_score(labels, total_scores)
        ax.plot(fpr, tpr, label=f"Comparison (AUC={auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("VLM Reward: Success vs Failure ROC", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)


def plot_trajectory_examples(trajs_data: list, output_dir: str, n_examples: int = 4):
    """Plot VLM score vs oracle reward for example trajectories."""
    has_completion = all(len(t["vlm_completion"]) > 0 for t in trajs_data)
    has_comparison = all(len(t["vlm_comparison"]) > 0 for t in trajs_data)

    if not has_completion and not has_comparison:
        return

    # Pick examples: 2 success, 2 failure (if available)
    if has_comparison:
        success_trajs = [t for t in trajs_data if t["episode_success"] and len(t["vlm_comparison"]) > 0]
        failure_trajs = [t for t in trajs_data if not t["episode_success"] and len(t["vlm_comparison"]) > 0]
    else:
        success_trajs = [t for t in trajs_data if t["episode_success"] and len(t["vlm_completion"]) > 0]
        failure_trajs = [t for t in trajs_data if not t["episode_success"] and len(t["vlm_completion"]) > 0]

    examples = []
    for t in success_trajs[:2]:
        examples.append(("Success", t))
    for t in failure_trajs[:2]:
        examples.append(("Failure", t))

    if not examples:
        return

    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for idx, (label, traj) in enumerate(examples):
        ax = axes[0, idx]

        if has_comparison:
            # Plot cumulative comparison score vs cumulative oracle reward
            vlm_comp = np.array(traj["vlm_comparison"])
            cum_comp = np.cumsum(vlm_comp)
            steps = range(len(cum_comp))
            ax.plot(steps, cum_comp, "b-", label="VLM Cum. Comparison", linewidth=2)

            # Oracle cumulative reward
            oracle = np.array(traj["cumulative_oracle"][:len(cum_comp)])
            if oracle.max() > 0:
                # Scale oracle to match comparison range for visualization
                oracle_scaled = oracle / oracle.max() * max(abs(cum_comp.max()), abs(cum_comp.min()), 1)
            else:
                oracle_scaled = oracle
            ax.plot(range(len(oracle_scaled)), oracle_scaled, "r--", label="Oracle (scaled)", linewidth=2)

        elif has_completion:
            steps = range(len(traj["vlm_completion"]))
            ax.plot(steps, traj["vlm_completion"], "b-", label="VLM Score", linewidth=2)
            oracle = np.array(traj["cumulative_oracle"][:len(traj["vlm_completion"])])
            if oracle.max() > 0:
                oracle_norm = oracle / oracle.max()
            else:
                oracle_norm = oracle
            ax.plot(range(len(oracle_norm)), oracle_norm, "r--", label="Oracle (norm)", linewidth=2)
            ax.set_ylim(-0.1, 1.1)

        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"{label} Episode", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("VLM Reward vs Oracle Reward Over Time", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "trajectory_examples.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, "trajectory_examples.png"), dpi=150)
    plt.close(fig)


def plot_score_scatter(trajs_data: list, output_dir: str):
    """Scatter plot: VLM score vs total oracle reward."""
    has_completion = all(len(t["vlm_completion"]) > 0 for t in trajs_data)
    has_comparison = all(len(t["vlm_comparison"]) > 0 for t in trajs_data)

    if not has_completion and not has_comparison:
        return

    labels = [t["episode_success"] for t in trajs_data]
    total_oracle = [t["total_oracle_reward"] for t in trajs_data]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    if has_comparison:
        vlm_scores = [sum(t["vlm_comparison"]) for t in trajs_data]
        ylabel = "Total VLM Comparison Score"
    else:
        vlm_scores = [t["vlm_completion"][-1] for t in trajs_data]
        ylabel = "Final VLM Completion Score"

    success_x = [o for o, l in zip(total_oracle, labels) if l]
    success_y = [v for v, l in zip(vlm_scores, labels) if l]
    failure_x = [o for o, l in zip(total_oracle, labels) if not l]
    failure_y = [v for v, l in zip(vlm_scores, labels) if not l]

    ax.scatter(failure_x, failure_y, c="red", alpha=0.6, label="Failure", s=30)
    ax.scatter(success_x, success_y, c="green", alpha=0.6, label="Success", s=30)

    ax.set_xlabel("Total Oracle Reward", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("VLM Score vs Oracle Reward", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "score_scatter.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, "score_scatter.png"), dpi=150)
    plt.close(fig)


def plot_score_distributions(trajs_data: list, output_dir: str):
    """Histograms of VLM scores for success vs failure."""
    has_completion = all(len(t["vlm_completion"]) > 0 for t in trajs_data)
    has_comparison = all(len(t["vlm_comparison"]) > 0 for t in trajs_data)

    if not has_completion and not has_comparison:
        return

    labels = [t["episode_success"] for t in trajs_data]

    if has_comparison:
        vlm_scores = [sum(t["vlm_comparison"]) for t in trajs_data]
        xlabel = "Total VLM Comparison Score"
        title = "VLM Comparison Score Distribution: Success vs Failure"
        # Adaptive bins for comparison scores (range can vary)
        lo = min(vlm_scores) - 1
        hi = max(vlm_scores) + 1
        bins = np.linspace(lo, hi, 21)
    else:
        vlm_scores = [t["vlm_completion"][-1] for t in trajs_data]
        xlabel = "Final VLM Completion Score"
        title = "VLM Score Distribution: Success vs Failure"
        bins = np.linspace(0, 1, 21)

    success_scores = [s for s, l in zip(vlm_scores, labels) if l]
    failure_scores = [s for s, l in zip(vlm_scores, labels) if not l]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if failure_scores:
        ax.hist(failure_scores, bins=bins, alpha=0.6, color="red", label="Failure", density=True)
    if success_scores:
        ax.hist(success_scores, bins=bins, alpha=0.6, color="green", label="Success", density=True)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "score_distribution.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, "score_distribution.png"), dpi=150)
    plt.close(fig)


def generate_latex_table(all_metrics: dict, output_dir: str):
    """Generate LaTeX table for paper."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Open-Loop VLM Reward Model Metrics (Comparison Mode)}")
    lines.append(r"\label{tab:openloop_metrics}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"Metric & Score \\")
    lines.append(r"\midrule")

    # Discriminative
    disc = all_metrics.get("discriminative", {})
    _add_row_single(lines, "ROC-AUC", disc.get("comparison_roc_auc"))
    _add_row_single(lines, "Pairwise Ranking Acc (\\%)",
                    _pct(disc.get("comparison_pairwise_accuracy")))
    _add_row_single(lines, "Best Binary Acc (\\%)",
                    _pct(disc.get("comparison_best_accuracy")))
    _add_row_single(lines, "Mean Score (Success)",
                    disc.get("comparison_mean_success"))
    _add_row_single(lines, "Mean Score (Failure)",
                    disc.get("comparison_mean_failure"))

    lines.append(r"\midrule")

    # Correlation
    corr = all_metrics.get("correlation", {})
    _add_row_single(lines, "Pearson (per-traj)", corr.get("comparison_pearson_mean"))
    _add_row_single(lines, "Spearman (per-traj)", corr.get("comparison_spearman_mean"))
    _add_row_single(lines, "Global Pearson", corr.get("comparison_global_pearson"))

    lines.append(r"\midrule")

    # Temporal
    temp = all_metrics.get("temporal", {})
    _add_row_single(lines, "Reward Variance", temp.get("comparison_variance_mean"))
    _add_row_single(lines, "Jump Frequency", temp.get("comparison_jump_freq_mean"))
    _add_row_single(lines, "Positive Frac (Success, \\%)",
                    _pct(temp.get("comparison_positive_frac_success")))
    _add_row_single(lines, "Positive Frac (Failure, \\%)",
                    _pct(temp.get("comparison_positive_frac_failure")))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    with open(os.path.join(output_dir, "metrics_table.tex"), "w") as f:
        f.write(latex)

    return latex


def _add_row_single(lines, name, val):
    """Helper to add a single-column LaTeX table row."""
    def fmt(v):
        if v is None:
            return "--"
        if isinstance(v, str):
            return v
        return f"{v:.3f}"
    lines.append(f"{name} & {fmt(val)} \\\\")


def _pct(v):
    """Convert fraction to percentage string."""
    if v is None:
        return None
    return f"{v * 100:.1f}"


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compute open-loop VLM reward metrics")
    parser.add_argument("--scores_path", type=str, required=True,
                        help="Path to vlm_scores.json from score_with_vlm.py")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for metrics and plots")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.scores_path), "metrics")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading scores from {args.scores_path}")
    data = load_scores(args.scores_path)
    trajectories = data.get("trajectories", [])
    print(f"Loaded {len(trajectories)} trajectories")

    # Extract structured data
    trajs_data = [extract_trajectory_data(t) for t in trajectories]

    # Count success/failure
    n_success = sum(1 for t in trajs_data if t["episode_success"])
    n_failure = len(trajs_data) - n_success
    print(f"Success: {n_success}, Failure: {n_failure}")

    # ---- Compute metrics ----
    print("\n--- Computing Discriminative Accuracy ---")
    disc_metrics = compute_discriminative_metrics(trajs_data)
    for k, v in disc_metrics.items():
        print(f"  {k}: {v}")

    print("\n--- Computing Correlation ---")
    corr_metrics = compute_correlation_metrics(trajs_data)
    for k, v in corr_metrics.items():
        print(f"  {k}: {v}")

    print("\n--- Computing Temporal Consistency ---")
    temp_metrics = compute_temporal_metrics(trajs_data)
    for k, v in temp_metrics.items():
        print(f"  {k}: {v}")

    # Aggregate all metrics
    all_metrics = {
        "discriminative": disc_metrics,
        "correlation": corr_metrics,
        "temporal": temp_metrics,
        "meta": {
            "scores_path": args.scores_path,
            "n_trajectories": len(trajs_data),
            "n_success": n_success,
            "n_failure": n_failure,
            "task_description": data.get("task_description", ""),
            "vlm_url": data.get("vlm_url", ""),
            "mode": data.get("mode", ""),
        },
    }

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, "openloop_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # ---- Generate plots ----
    print("\nGenerating plots...")
    plot_roc_curve(trajs_data, args.output_dir)
    plot_trajectory_examples(trajs_data, args.output_dir)
    plot_score_scatter(trajs_data, args.output_dir)
    plot_score_distributions(trajs_data, args.output_dir)
    print(f"Plots saved to {args.output_dir}")

    # ---- Generate LaTeX table ----
    latex = generate_latex_table(all_metrics, args.output_dir)
    print(f"\nLaTeX table saved to {os.path.join(args.output_dir, 'metrics_table.tex')}")
    print("\n" + latex)

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("OPEN-LOOP VLM REWARD METRICS SUMMARY (Comparison Mode)")
    print("=" * 60)
    print(f"Trajectories: {len(trajs_data)} ({n_success} success, {n_failure} failure)")
    print()
    print("Discriminative Accuracy:")
    if "comparison_roc_auc" in disc_metrics:
        print(f"  ROC-AUC:                  {disc_metrics['comparison_roc_auc']:.3f}")
    if "comparison_pairwise_accuracy" in disc_metrics:
        print(f"  Pairwise Ranking Acc:     {disc_metrics['comparison_pairwise_accuracy'] * 100:.1f}%")
    if "comparison_best_accuracy" in disc_metrics:
        print(f"  Best Binary Acc:          {disc_metrics['comparison_best_accuracy'] * 100:.1f}%")
    if "comparison_mean_success" in disc_metrics:
        print(f"  Mean Score (Success):     {disc_metrics['comparison_mean_success']:.2f}")
    if "comparison_mean_failure" in disc_metrics:
        print(f"  Mean Score (Failure):     {disc_metrics['comparison_mean_failure']:.2f}")
    print()
    print("Correlation with Ground Truth:")
    if "comparison_pearson_mean" in corr_metrics:
        print(f"  Pearson (per-traj):       {corr_metrics['comparison_pearson_mean']:.3f} +/- {corr_metrics.get('comparison_pearson_std', 0):.3f}")
    if "comparison_spearman_mean" in corr_metrics:
        print(f"  Spearman (per-traj):      {corr_metrics['comparison_spearman_mean']:.3f} +/- {corr_metrics.get('comparison_spearman_std', 0):.3f}")
    if "comparison_global_pearson" in corr_metrics:
        print(f"  Global Pearson:           {corr_metrics['comparison_global_pearson']:.3f}")
    print()
    print("Temporal Consistency:")
    if "comparison_variance_mean" in temp_metrics:
        print(f"  Reward Variance:          {temp_metrics['comparison_variance_mean']:.4f}")
    if "comparison_jump_freq_mean" in temp_metrics:
        print(f"  Jump Frequency:           {temp_metrics['comparison_jump_freq_mean']:.3f}")
    if "comparison_positive_frac_success" in temp_metrics:
        print(f"  Positive Frac (Success):  {temp_metrics['comparison_positive_frac_success'] * 100:.1f}%")
    if "comparison_positive_frac_failure" in temp_metrics:
        print(f"  Positive Frac (Failure):  {temp_metrics['comparison_positive_frac_failure'] * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
