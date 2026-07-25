#!/usr/bin/env python3
"""Extract final closed-loop results and report seed-level mean +/- std.

Paired comparisons against a reference method still use a 95% CI and a
paired t-test, since that section is testing significance rather than
just describing one method's spread.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def extract_tensorboard(run_dir: Path, success_tag: str, trials_tag: str):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    successes = []
    trials = []
    formal_eval_dir = run_dir / "closed_loop_320"
    scan_dir = formal_eval_dir if formal_eval_dir.is_dir() else run_dir
    for event_file in scan_dir.rglob("events.out.tfevents.*"):
        accumulator = EventAccumulator(
            str(event_file), size_guidance={"scalars": 0}
        )
        accumulator.Reload()
        tags = set(accumulator.Tags().get("scalars", []))
        if success_tag not in tags or trials_tag not in tags:
            continue
        successes.extend(accumulator.Scalars(success_tag))
        trials.extend(accumulator.Scalars(trials_tag))
    if not successes or not trials:
        return None
    success_event = max(successes, key=lambda event: (event.step, event.wall_time))
    trial_candidates = [event for event in trials if event.step == success_event.step]
    trial_event = max(
        trial_candidates or trials, key=lambda event: (event.step, event.wall_time)
    )
    n = int(round(trial_event.value))
    rate = float(success_event.value)
    return {
        "step": success_event.step,
        "successes": int(round(rate * n)),
        "trials": n,
        "rate": rate,
    }


def mean_std(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else math.nan
    return mean, std


def t_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, math.nan, math.nan
    sem = stats.sem(values)
    half = stats.t.ppf((1.0 + confidence) / 2.0, len(values) - 1) * sem
    return mean, mean - half, mean + half


def paired_interval(a, b, confidence=0.95):
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean, low, high = t_interval(diffs, confidence)
    if len(diffs) < 2 or np.allclose(diffs, diffs[0]):
        pvalue = math.nan if len(diffs) < 2 else (0.0 if diffs[0] != 0 else 1.0)
    else:
        pvalue = float(stats.ttest_1samp(diffs, 0.0).pvalue)
    return mean, low, high, pvalue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--reference", default="roboreward")
    parser.add_argument("--success-tag", default="eval/success_once")
    parser.add_argument("--trials-tag", default="eval/num_trajectories")
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    methods = args.methods or sorted(
        path.name for path in args.results_dir.iterdir() if path.is_dir()
    )
    rows = []
    for method in methods:
        method_dir = args.results_dir / method
        for seed_dir in sorted(method_dir.glob("seed_*")):
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            result = extract_tensorboard(seed_dir, args.success_tag, args.trials_tag)
            if result is not None:
                rows.append({"method": method, "seed": seed, **result})

    if not rows:
        raise SystemExit("No TensorBoard closed-loop results found")
    csv_out = args.csv_out or args.results_dir / "closed_loop_seed_results.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["method"]][row["seed"]] = row["rate"]
    print("| Method | Seeds | Mean ± Std across training seeds |")
    print("|---|---:|---:|")
    for method in methods:
        seed_rates = grouped.get(method, {})
        if not seed_rates:
            continue
        mean, std = mean_std(list(seed_rates.values()))
        std_text = "NA" if math.isnan(std) else f"{100*std:.2f}%"
        print(
            f"| {method} | {len(seed_rates)} | {100*mean:.2f}% ± {std_text} |"
        )

    reference = grouped.get(args.reference, {})
    if reference:
        print(f"\nPaired differences versus {args.reference}:")
        print("| Method | Paired seeds | Mean difference | 95% CI | paired t-test p |")
        print("|---|---:|---:|---:|---:|")
        for method in methods:
            if method == args.reference or method not in grouped:
                continue
            common = sorted(set(grouped[method]) & set(reference))
            if not common:
                continue
            mean, low, high, pvalue = paired_interval(
                [grouped[method][seed] for seed in common],
                [reference[seed] for seed in common],
            )
            ptext = "NA" if math.isnan(pvalue) else f"{pvalue:.4f}"
            print(
                f"| {method} | {len(common)} | {100*mean:+.2f} pp | "
                f"[{100*low:+.2f}, {100*high:+.2f}] pp | {ptext} |"
            )
    print(f"\nRaw seed results: {csv_out}")


if __name__ == "__main__":
    main()
