#!/usr/bin/env python3
"""HMRF alpha sensitivity for GAM-GA.

This supplementary experiment tests whether the fixed HMRF weighting

    HMRF_alpha(f) = alpha * MI_norm(f) + (1 - alpha) * ReliefF_norm(f)

changes the fold-safe wrapper conclusion. It is intentionally small and
targeted: GA backbone, three representative datasets, five alpha values, three
runs, five outer folds, and inner 3-fold KNN wrapper evaluation.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_classif as mi_c
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


ROOT = Path("/Users/gicisky/Desktop/论文研读分析/crow/crow_search1")
CODE_DIR = ROOT / "code"
FILTER_DIR = ROOT / "filter"
DATA_DIR = ROOT / "dataCSV_high" / "all"
EXP_DIR = ROOT / "experiments" / "gam第三次实验汇总"
RES_DIR = EXP_DIR / "results"
OUT_DEFAULT = RES_DIR / "bench_alpha_sensitivity.jsonl"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(FILTER_DIR))
os.chdir(CODE_DIR)

import bench_gam  # noqa: E402
from bench_gam import GAM, ga_gam_safe  # noqa: E402
from Filter import load_csv, reliefFScore  # noqa: E402


DATASETS = [
    "colon(62,2000).csv",
    "DLBCL(77,5469).csv",
    "9_Tumors(60,5726).csv",
]
ALPHAS = [0.2, 0.4, 0.5, 0.6, 0.8]


def dataset_name(ds_file):
    return ds_file.split("(")[0]


def safe_redundancy(X, idx):
    idx = np.asarray(idx, dtype=int)
    if len(idx) <= 1:
        return 0.0
    Xp = X[:, idx]
    Xc = (Xp - Xp.mean(axis=0)) / (Xp.std(axis=0) + 1e-12)
    C = np.abs(np.corrcoef(Xc.T))
    iu = np.triu_indices_from(C, k=1)
    value = float(np.mean(C[iu]))
    return 0.0 if np.isnan(value) else value


def compute_hmrf_alpha(X, y, alpha, seed):
    mi = mi_c(X, y, random_state=seed)
    rf = reliefFScore(X, y)
    mi_n = (mi - mi.min()) / (mi.max() - mi.min() + 1e-12)
    rf_n = (rf - rf.min()) / (rf.max() - rf.min() + 1e-12)
    return alpha * mi_n + (1.0 - alpha) * rf_n


def patch_inner_cv_evaluator(X_tr, y_tr, inner_folds, knn_k, seed):
    def inner_eval(_X, _y, idx):
        idx = np.asarray(idx, dtype=int)
        if len(idx) == 0:
            return 0.0
        clf = KNeighborsClassifier(knn_k, metric="manhattan", n_jobs=1)
        cv = KFold(inner_folds, shuffle=True, random_state=seed)
        return float(np.mean(cross_val_score(clf, X_tr[:, idx], y_tr, cv=cv)))

    return inner_eval


def outer_metrics(X_tr, y_tr, X_te, y_te, selected, knn_k):
    if len(selected) == 0:
        y_pred = np.full_like(y_te, fill_value=np.bincount(y_tr).argmax())
    else:
        clf = KNeighborsClassifier(knn_k, metric="manhattan", n_jobs=1)
        clf.fit(X_tr[:, selected], y_tr)
        y_pred = clf.predict(X_te[:, selected])
    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
        "macro_f1": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
    }


def run_task(ds_file, alpha, run_idx, args):
    ds_name = dataset_name(ds_file)
    X_raw, y_raw = load_csv(str(DATA_DIR / ds_file))
    if y_raw.dtype.kind in ("U", "S", "O"):
        y_raw = LabelEncoder().fit_transform(y_raw)
    y_raw = y_raw.astype(int)

    D = int(X_raw.shape[1])
    k = max(1, int(D * args.k_ratio))
    run_seed = args.seed + run_idx * 17 + int(alpha * 1000)
    outer = StratifiedKFold(args.outer_folds, shuffle=True, random_state=run_seed)
    fold_records = []
    start = time.time()

    for fold_idx, (tr, te) in enumerate(outer.split(X_raw, y_raw)):
        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X_raw[tr].copy())
        X_te = scaler.transform(X_raw[te])
        y_tr, y_te = y_raw[tr], y_raw[te]

        orig_evaluate = bench_gam.evaluate
        bench_gam.evaluate = patch_inner_cv_evaluator(
            X_tr, y_tr, args.inner_folds, args.knn_k, run_seed + fold_idx
        )
        np.random.seed(run_seed + fold_idx)
        try:
            scores = compute_hmrf_alpha(X_tr, y_tr, alpha, run_seed + fold_idx)
            gam = GAM(X_tr, scores)
            original_crossover = gam.safe_crossover

            def crossover_with_lambda(p1, p2, kk, gen=0, max_gen=100, **kwargs):
                return original_crossover(
                    p1, p2, kk, gen=gen, max_gen=max_gen, penalty=args.lam
                )

            gam.safe_crossover = crossover_with_lambda
            selected = ga_gam_safe(X_tr, y_tr, k, gam, args.N, args.T)
        finally:
            bench_gam.evaluate = orig_evaluate

        metrics = outer_metrics(X_tr, y_tr, X_te, y_te, selected, args.knn_k)
        metrics.update(
            {
                "fold": int(fold_idx),
                "redundancy": safe_redundancy(X_tr, selected),
                "selected_count": int(len(selected)),
                "selected_features": [int(x) for x in np.asarray(selected, dtype=int)],
            }
        )
        fold_records.append(metrics)

    elapsed = time.time() - start
    return {
        "dataset": ds_name,
        "dataset_file": ds_file,
        "base": "GA",
        "mode": "GAM-GA-alpha",
        "alpha": float(alpha),
        "lambda": float(args.lam),
        "run": int(run_idx),
        "protocol": "fold-safe alpha sensitivity",
        "D": D,
        "k": int(k),
        "N": int(args.N),
        "T": int(args.T),
        "outer_folds": int(args.outer_folds),
        "inner_folds": int(args.inner_folds),
        "knn_k": int(args.knn_k),
        "runtime_sec": float(elapsed),
        "accuracy_mean": float(np.mean([x["accuracy"] for x in fold_records])),
        "accuracy_std": float(np.std([x["accuracy"] for x in fold_records])),
        "balanced_accuracy_mean": float(np.mean([x["balanced_accuracy"] for x in fold_records])),
        "macro_f1_mean": float(np.mean([x["macro_f1"] for x in fold_records])),
        "weighted_f1_mean": float(np.mean([x["weighted_f1"] for x in fold_records])),
        "redundancy_mean": float(np.mean([x["redundancy"] for x in fold_records])),
        "selected_count_mean": float(np.mean([x["selected_count"] for x in fold_records])),
        "fold_records": fold_records,
    }


def load_completed(path):
    completed = set()
    if not path.exists():
        return completed
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "error" not in row:
                completed.add((row["dataset_file"], float(row["alpha"]), int(row["run"])))
    return completed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT_DEFAULT))
    p.add_argument("--datasets", nargs="*", default=DATASETS)
    p.add_argument("--alphas", nargs="*", type=float, default=ALPHAS)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--k-ratio", type=float, default=0.05)
    p.add_argument("--lam", type=float, default=0.6)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260713)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.N < 4:
        raise SystemExit("N must be at least 4.")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("Invalid shard settings.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and out.exists():
        out.unlink()

    tasks = [(ds, float(a), run) for ds in args.datasets for a in args.alphas for run in range(args.runs)]
    tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard_index]
    completed = load_completed(out)
    tasks = [t for t in tasks if (t[0], float(t[1]), int(t[2])) not in completed]

    print("HMRF alpha sensitivity experiment")
    print(f"Output: {out}")
    print(f"Remaining tasks: {len(tasks)}")
    print(
        f"Protocol: datasets={len(args.datasets)}, alphas={args.alphas}, runs={args.runs}, "
        f"outer={args.outer_folds}, inner={args.inner_folds}, N={args.N}, T={args.T}, lambda={args.lam}"
    )
    print(f"Shard: {args.shard_index + 1}/{args.num_shards}")

    t0 = time.time()
    for i, (ds, alpha, run) in enumerate(tasks, start=1):
        try:
            row = run_task(ds, alpha, run, args)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            row = {
                "dataset": dataset_name(ds),
                "dataset_file": ds,
                "alpha": float(alpha),
                "run": int(run),
                "error": str(exc)[:500],
            }
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if "error" in row:
            status = f"ERROR {row['error'][:80]}"
        else:
            status = (
                f"acc={row['accuracy_mean']:.4f} "
                f"bacc={row['balanced_accuracy_mean']:.4f} "
                f"mf1={row['macro_f1_mean']:.4f}"
            )
        print(
            f"[{i}/{len(tasks)}] {dataset_name(ds):14s} alpha={alpha:.1f} run={run} "
            f"{status} elapsed={(time.time()-t0)/60:.1f}m",
            flush=True,
        )


if __name__ == "__main__":
    main()
