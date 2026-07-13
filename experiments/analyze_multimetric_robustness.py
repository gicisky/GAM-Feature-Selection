#!/usr/bin/env python3
"""Analyze supplementary multimetric robustness results."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXP_DIR = Path("/Users/gicisky/Desktop/论文研读分析/crow/crow_search1/experiments/gam第三次实验汇总")
IN_DEFAULT = EXP_DIR / "results" / "bench_multimetric_robustness.jsonl"
OUT_DIR_DEFAULT = EXP_DIR / "results" / "multimetric_robustness_summary"


METRICS = [
    "accuracy_mean",
    "balanced_accuracy_mean",
    "macro_f1_mean",
    "weighted_f1_mean",
    "redundancy_mean",
    "selected_count_mean",
    "runtime_sec",
]


def p_wilcoxon(x, y):
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def load_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "error" not in row:
                rows.append(row)
    if not rows:
        raise SystemExit(f"No successful rows found in {path}")
    return pd.DataFrame(rows)


def format_float(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(IN_DEFAULT))
    parser.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(args.input)
    df.to_csv(out_dir / "all_successful_runs.csv", index=False)

    by_dataset = (
        df.groupby(["dataset", "mode"], as_index=False)[METRICS]
        .mean()
        .sort_values(["dataset", "mode"])
    )
    by_dataset.to_csv(out_dir / "dataset_mode_means.csv", index=False)

    rows = []
    for dataset, sub in by_dataset.groupby("dataset"):
        modes = set(sub["mode"])
        if {"Bare-GA", "GAM-GA"}.issubset(modes):
            bare = sub[sub["mode"] == "Bare-GA"].iloc[0]
            gam = sub[sub["mode"] == "GAM-GA"].iloc[0]
            row = {"dataset": dataset}
            for metric in METRICS:
                row[f"Bare_{metric}"] = bare[metric]
                row[f"GAM_{metric}"] = gam[metric]
                row[f"delta_{metric}"] = gam[metric] - bare[metric]
            rows.append(row)
    paired = pd.DataFrame(rows)
    if not paired.empty:
        paired = paired.sort_values("dataset")
    paired.to_csv(out_dir / "paired_dataset_deltas.csv", index=False)

    stat_rows = []
    if not paired.empty:
        for metric in METRICS:
            delta_col = f"delta_{metric}"
            if delta_col not in paired:
                continue
            x = paired[f"Bare_{metric}"].to_numpy()
            y = paired[f"GAM_{metric}"].to_numpy()
            stat_rows.append(
                {
                    "metric": metric.replace("_mean", ""),
                    "Bare_macro": float(np.mean(x)),
                    "GAM_macro": float(np.mean(y)),
                    "delta": float(np.mean(y - x)),
                    "wins": int(np.sum((y - x) > 0)),
                    "losses": int(np.sum((y - x) < 0)),
                    "ties": int(np.sum(np.isclose(y - x, 0))),
                    "wilcoxon_p": p_wilcoxon(x, y),
                }
            )
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(out_dir / "macro_summary_and_wilcoxon.csv", index=False)

    md = []
    md.append("# Supplementary Multimetric Robustness Summary\n")
    md.append(
        "This analysis summarizes a supplementary fold-safe robustness check. "
        "The wrapper fitness remains inner-CV KNN accuracy; balanced accuracy and F1 are computed only on held-out outer folds.\n"
    )
    md.append("## Macro Summary\n")
    md.append("| Metric | Bare-GA | GAM-GA | Delta | W/L/T | Wilcoxon p |")
    md.append("|---|---:|---:|---:|---:|---:|")
    if stats.empty:
        md.append("| No paired Bare-GA/GAM-GA datasets were available. | NA | NA | NA | NA | NA |")
    else:
        for _, r in stats.iterrows():
            md.append(
                f"| {r['metric']} | {format_float(r['Bare_macro'])} | "
                f"{format_float(r['GAM_macro'])} | {format_float(r['delta'])} | "
                f"{int(r['wins'])}/{int(r['losses'])}/{int(r['ties'])} | "
                f"{format_float(r['wilcoxon_p'])} |"
            )
    md.append("\n## Dataset-Level Deltas\n")
    keep = [
        "dataset",
        "delta_accuracy_mean",
        "delta_balanced_accuracy_mean",
        "delta_macro_f1_mean",
        "delta_weighted_f1_mean",
        "delta_redundancy_mean",
        "delta_runtime_sec",
    ]
    md.append("| Dataset | ΔAcc | ΔBalanced Acc | ΔMacro-F1 | ΔWeighted-F1 | ΔRedundancy | ΔRuntime(s) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    if paired.empty:
        md.append("| No paired datasets | NA | NA | NA | NA | NA | NA |")
    else:
        for _, r in paired[keep].iterrows():
            md.append(
                f"| {r['dataset']} | {format_float(r['delta_accuracy_mean'])} | "
                f"{format_float(r['delta_balanced_accuracy_mean'])} | "
                f"{format_float(r['delta_macro_f1_mean'])} | "
                f"{format_float(r['delta_weighted_f1_mean'])} | "
                f"{format_float(r['delta_redundancy_mean'])} | "
                f"{format_float(r['delta_runtime_sec'])} |"
            )
    md.append(
        "\nRecommended manuscript positioning: report this as a supplementary robustness endpoint analysis, not as a new primary benchmark."
    )
    (out_dir / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {out_dir}")
    if stats.empty:
        print("No paired Bare-GA/GAM-GA rows available for Wilcoxon summary.")
    else:
        print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
