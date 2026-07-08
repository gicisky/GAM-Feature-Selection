#!/usr/bin/env python3
"""
Jaccard selected-subset stability analysis.

Purpose
-------
This script evaluates selected-feature stability for the 3-tier GAM ablation:

    Bare-GA vs GAM-λ0 vs GAM λ=0.6

The stability metric is the pairwise Jaccard similarity among selected
feature subsets across folds and runs:

    Jaccard(S_i, S_j) = |S_i ∩ S_j| / |S_i ∪ S_j|

Recommended use
---------------
This is a supplementary/stability analysis, not the primary benchmark.

Default protocol:
    - 5 representative datasets
    - GA base optimizer
    - Bare, GAM-λ0, GAM λ=0.6
    - 5 runs × 5 outer folds = 25 selected subsets per dataset/mode
    - N=50, T=50
    - fold-safe preprocessing and HMRF scoring

Important:
    For datasets with MI pre-filtering, selected indices are mapped back to
    the original feature space before computing Jaccard. This is required
    because each fold may have a different pre-filtered feature pool.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

# Keep BLAS thread usage controlled when multiprocessing is used.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import multiprocessing as mp
import numpy as np
from sklearn.feature_selection import mutual_info_classif as mi_c
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ---------------------------------------------------------------------
# Repository-relative paths
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"experiments", "scripts"} else THIS_FILE.parent

CODE_DIR = REPO_ROOT / "code"
UTILS_DIR = REPO_ROOT / "utils"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(UTILS_DIR))

try:
    import bench_gam
    import bench_bare
    from bench_gam import GAM, ga_gam_safe
    from bench_bare import ga_bare
    from Filter import load_csv, reliefFScore
except Exception as exc:
    raise RuntimeError(
        "Failed to import project modules. Please run this script from the repository "
        "root or keep it under experiments/. Expected directories: code/, utils/, data/."
    ) from exc


DATASETS = [
    "colon(62,2000).csv",
    "DLBCL(77,5469).csv",
    "9_Tumors(60,5726).csv",
    "Brain_Tumor_2(49,10368).csv",
    "arcene(200,10000).csv",
]

MODES = ["Bare", "GAM-lambda0", "GAM"]
MODE_LABEL = {
    "Bare": "Bare-GA",
    "GAM-lambda0": "GAM-λ0",
    "GAM": "GAM λ=0.6",
}


def make_cv(y, n_splits, seed):
    """Use stratified CV when class counts permit it; otherwise fall back to KFold."""
    _, counts = np.unique(y, return_counts=True)
    if counts.min() >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def parse_solution(sol, n_features):
    """Convert returned solution to a clean sorted index array.

    Supports:
    - boolean masks
    - 0/1 integer or float masks
    - explicit index lists
    """
    sol = np.asarray(sol)

    if sol.ndim == 1 and sol.size == n_features and sol.dtype == bool:
        idx = np.where(sol)[0]
    elif sol.ndim == 1 and sol.size == n_features and np.all(np.isin(np.unique(sol), [0, 1])):
        idx = np.where(sol > 0)[0]
    else:
        idx = sol.astype(int).ravel()

    idx = np.unique(idx)
    idx = idx[(idx >= 0) & (idx < n_features)]
    return np.sort(idx)


def jaccard(s1, s2):
    s1, s2 = set(s1), set(s2)
    union = s1 | s2
    if not union:
        return 0.0
    return len(s1 & s2) / len(union)


def dataset_name_from_file(ds_file):
    return ds_file.split("(")[0]


def run_one_task(args):
    ds_file, mode, run_idx, cfg = args
    ds_name = dataset_name_from_file(ds_file)

    try:
        x_path = DATA_DIR / ds_file
        X_raw, y_raw = load_csv(str(x_path))
        if y_raw.dtype.kind in ("U", "S", "O"):
            y_raw = LabelEncoder().fit_transform(y_raw)
        y_raw = y_raw.astype(int)

        d_orig = X_raw.shape[1]
        k = min(cfg["k_max"], int(d_orig * cfg["k_ratio"]))
        run_seed = cfg["base_seed"] + run_idx * 7

        outer = make_cv(y_raw, cfg["outer_folds"], run_seed)
        folds = list(outer.split(X_raw, y_raw))

        subsets = []
        fold_accs = []
        t0 = time.time()

        for fold_id, (tr, te) in enumerate(folds):
            fold_seed = run_seed * 100 + fold_id
            np.random.seed(fold_seed)
            random.seed(fold_seed)

            scaler = MinMaxScaler()
            X_tr_full = scaler.fit_transform(X_raw[tr].copy())
            X_te_full = scaler.transform(X_raw[te].copy())
            y_tr, y_te = y_raw[tr], y_raw[te]

            # Fold-safe optional MI pre-filter for large-D datasets.
            if d_orig >= cfg["pre_threshold"]:
                mi_pre = mi_c(X_tr_full, y_tr, random_state=fold_seed)
                pre_idx = np.argsort(-mi_pre)[:min(cfg["pre_n"], d_orig)]
                X_tr = X_tr_full[:, pre_idx]
                X_te = X_te_full[:, pre_idx]
            else:
                pre_idx = None
                X_tr = X_tr_full
                X_te = X_te_full

            def evalfn(X, y, idx):
                idx = parse_solution(idx, X.shape[1])
                if len(idx) == 0:
                    return 0.0
                clf = KNeighborsClassifier(
                    n_neighbors=cfg["knn_k"],
                    metric="manhattan",
                    n_jobs=1,
                )
                return float(
                    np.mean(
                        cross_val_score(
                            clf,
                            X[:, idx],
                            y,
                            cv=make_cv(y, cfg["inner_folds"], fold_seed),
                        )
                    )
                )

            if mode == "Bare":
                old_bare_eval = getattr(bench_bare, "evaluate", None)
                old_gam_eval = getattr(bench_gam, "evaluate", None)
                try:
                    if old_bare_eval is not None:
                        bench_bare.evaluate = evalfn
                    if old_gam_eval is not None:
                        bench_gam.evaluate = evalfn
                    sol = ga_bare(X_tr, y_tr, k, cfg["N"], cfg["T"])
                finally:
                    if old_bare_eval is not None:
                        bench_bare.evaluate = old_bare_eval
                    if old_gam_eval is not None:
                        bench_gam.evaluate = old_gam_eval

            else:
                # Fold-safe HMRF scoring: train fold only.
                relief = np.asarray(reliefFScore(X_tr, y_tr), dtype=float)
                mi = np.asarray(mi_c(X_tr, y_tr, random_state=fold_seed), dtype=float)

                relief_n = (relief - relief.min()) / (relief.max() - relief.min() + 1e-12)
                mi_n = (mi - mi.min()) / (mi.max() - mi.min() + 1e-12)
                scores = 0.5 * mi_n + 0.5 * relief_n

                gam = GAM(X_tr, scores)
                original_safe_crossover = gam.safe_crossover
                penalty = 0.0 if mode == "GAM-lambda0" else cfg["lambda_penalty"]

                def patched_safe_crossover(p1, p2, k_, g=0, mg=100, **kwargs):
                    return original_safe_crossover(
                        p1, p2, k_, gen=g, max_gen=mg, penalty=penalty
                    )

                gam.safe_crossover = patched_safe_crossover

                old_gam_eval = getattr(bench_gam, "evaluate", None)
                try:
                    if old_gam_eval is not None:
                        bench_gam.evaluate = evalfn
                    sol = ga_gam_safe(X_tr, y_tr, k, gam, cfg["N"], cfg["T"])
                finally:
                    if old_gam_eval is not None:
                        bench_gam.evaluate = old_gam_eval

            sol_reduced = parse_solution(sol, X_tr.shape[1])

            # Critical: map selected indices back to original feature space before
            # comparing Jaccard across folds. Otherwise fold-specific pre-filtered
            # indices are not comparable.
            if pre_idx is not None:
                sol_original = np.sort(pre_idx[sol_reduced]).astype(int)
            else:
                sol_original = sol_reduced.astype(int)

            subsets.append(sol_original.tolist())

            if len(sol_reduced) > 0:
                clf = KNeighborsClassifier(n_neighbors=cfg["knn_k"], metric="manhattan")
                clf.fit(X_tr[:, sol_reduced], y_tr)
                fold_accs.append(float(clf.score(X_te[:, sol_reduced], y_te)))
            else:
                fold_accs.append(0.0)

        elapsed = time.time() - t0
        return {
            "script_version": "jaccard_stability_v2_original_index_mapping",
            "dataset": ds_name,
            "dataset_file": ds_file,
            "mode": mode,
            "mode_label": MODE_LABEL[mode],
            "run": run_idx,
            "D_orig": int(d_orig),
            "k": int(k),
            "N": int(cfg["N"]),
            "T": int(cfg["T"]),
            "outer_folds": int(cfg["outer_folds"]),
            "inner_folds": int(cfg["inner_folds"]),
            "subsets_original_index": subsets,
            "acc_mean": float(np.mean(fold_accs)),
            "acc_std": float(np.std(fold_accs)),
            "elapsed_s": float(elapsed),
        }

    except Exception as exc:
        import traceback

        traceback.print_exc()
        return {
            "dataset": ds_name,
            "dataset_file": ds_file,
            "mode": mode,
            "mode_label": MODE_LABEL.get(mode, mode),
            "run": run_idx,
            "error": str(exc)[:500],
        }


def compute_summary(records, out_summary_path=None):
    by_ds_mode = defaultdict(list)
    acc_by_ds_mode = defaultdict(list)

    for rec in records:
        if "error" in rec:
            continue
        key = (rec["dataset"], rec["mode"])
        subsets = rec.get("subsets_original_index", rec.get("subsets", []))
        by_ds_mode[key].extend(subsets)
        acc_by_ds_mode[key].append(rec.get("acc_mean", np.nan))

    datasets = sorted({k[0] for k in by_ds_mode.keys()})
    summary_rows = []

    print("\n=== Jaccard Stability ===")
    print(f"{'Dataset':20s} {'Bare-GA':>16s} {'GAM-λ0':>16s} {'GAM λ=0.6':>16s} {'Best':>12s}")
    print("-" * 88)

    for ds in datasets:
        vals = {}
        for mode in MODES:
            subsets = by_ds_mode.get((ds, mode), [])
            if len(subsets) >= 2:
                pairs = [jaccard(subsets[i], subsets[j]) for i, j in combinations(range(len(subsets)), 2)]
                vals[mode] = (float(np.mean(pairs)), float(np.std(pairs)), len(subsets), len(pairs))
            else:
                vals[mode] = (np.nan, np.nan, len(subsets), 0)

        valid_modes = [m for m in MODES if np.isfinite(vals[m][0])]
        best = max(valid_modes, key=lambda m: vals[m][0]) if valid_modes else "NA"

        print(
            f"{ds:20s} "
            f"{vals['Bare'][0]:>7.4f}±{vals['Bare'][1]:<7.4f} "
            f"{vals['GAM-lambda0'][0]:>7.4f}±{vals['GAM-lambda0'][1]:<7.4f} "
            f"{vals['GAM'][0]:>7.4f}±{vals['GAM'][1]:<7.4f} "
            f"{MODE_LABEL.get(best, best):>12s}"
        )

        row = {"dataset": ds, "best": MODE_LABEL.get(best, best)}
        for mode in MODES:
            row[f"{mode}_jaccard_mean"] = vals[mode][0]
            row[f"{mode}_jaccard_std"] = vals[mode][1]
            row[f"{mode}_n_subsets"] = vals[mode][2]
            row[f"{mode}_n_pairs"] = vals[mode][3]
            acc_vals = [x for x in acc_by_ds_mode.get((ds, mode), []) if np.isfinite(x)]
            row[f"{mode}_acc_mean"] = float(np.mean(acc_vals)) if acc_vals else np.nan
        summary_rows.append(row)

    print("-" * 88)
    for mode in MODES:
        means = [
            r[f"{mode}_jaccard_mean"]
            for r in summary_rows
            if np.isfinite(r[f"{mode}_jaccard_mean"])
        ]
        if means:
            print(f"{MODE_LABEL[mode]:14s} macro Jaccard: {np.mean(means):.4f}")

    if out_summary_path is not None:
        with open(out_summary_path, "w", encoding="utf-8") as f:
            for row in summary_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return summary_rows


def load_existing_records(out_path):
    records = []
    if not out_path.exists():
        return records
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def main():
    parser = argparse.ArgumentParser(description="Jaccard selected-subset stability analysis.")
    parser.add_argument("--quick", action="store_true", help="Quick smoke-test mode: 2 datasets, 2 runs, N=30, T=30.")
    parser.add_argument("--workers", type=int, default=None, help="Number of multiprocessing workers.")
    parser.add_argument("--overwrite", action="store_true", help="Delete previous output before running.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "bench_jaccard_stability.jsonl"
    summary_path = RESULTS_DIR / "bench_jaccard_stability_summary.jsonl"

    if args.quick:
        datasets = DATASETS[:2]
        nr = 2
        N = 30
        T = 30
    else:
        datasets = DATASETS
        nr = 5
        N = 50
        T = 50

    cfg = {
        "N": N,
        "T": T,
        "base_seed": 42,
        "knn_k": 5,
        "inner_folds": 3,
        "outer_folds": 5,
        "k_ratio": 0.05,
        "k_max": 200,
        "pre_threshold": 10000,
        "pre_n": 5000,
        "lambda_penalty": 0.6,
    }

    if args.overwrite and out_path.exists():
        out_path.unlink()
    if args.overwrite and summary_path.exists():
        summary_path.unlink()

    existing = load_existing_records(out_path)
    completed = {
        (rec.get("dataset"), rec.get("mode"), rec.get("run"))
        for rec in existing
        if "error" not in rec
    }

    tasks = [(ds, mode, r, cfg) for ds in datasets for mode in MODES for r in range(nr)]
    pending = [
        t for t in tasks
        if (dataset_name_from_file(t[0]), t[1], t[2]) not in completed
    ]

    print(
        f"{len(datasets)} datasets × {len(MODES)} modes × {nr} runs "
        f"= {len(tasks)} tasks ({len(pending)} remaining)"
    )
    print(
        f"N={N}, T={T}, outer={cfg['outer_folds']}-fold, inner={cfg['inner_folds']}-fold, "
        f"GA base, λ={cfg['lambda_penalty']}"
    )
    print(f"Output : {out_path}")
    print(f"Summary: {summary_path}")

    if not pending:
        print("No pending tasks. Recomputing summary from existing output.")
        compute_summary(existing, summary_path)
        return

    t0 = time.time()
    workers = args.workers
    if workers is None:
        workers = max(1, min(4, mp.cpu_count() - 2))

    with mp.Pool(workers, maxtasksperchild=1) as pool:
        for i, rec in enumerate(pool.imap_unordered(run_one_task, pending), start=1):
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if "error" in rec:
                status = f"ERROR: {rec['error'][:80]}"
            else:
                status = (
                    f"folds={len(rec['subsets_original_index'])}, "
                    f"acc={rec['acc_mean']:.4f}, {rec['elapsed_s']:.0f}s"
                )

            elapsed = time.time() - t0
            remaining = len(pending) - i
            eta = elapsed / i * remaining if i > 0 else 0.0

            print(
                f"[{i}/{len(pending)}] "
                f"{rec.get('dataset', '?'):14s} {rec.get('mode_label', rec.get('mode', '?')):12s} "
                f"{status} | elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m",
                flush=True,
            )

    all_records = load_existing_records(out_path)
    print(f"\nDone: {time.time() - t0:.0f}s → {out_path}")
    compute_summary(all_records, summary_path)


if __name__ == "__main__":
    main()
