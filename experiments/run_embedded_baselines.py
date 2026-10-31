#!/usr/bin/env python3
"""
Fold-safe embedded feature-selection baselines.

Baselines:
    - LASSO logistic regression
    - Elastic Net logistic regression
    - Random Forest feature importance

Protocol:
    - 7 representative datasets
    - 5-fold outer evaluation
    - Each embedded selector is trained only on the outer training fold
    - Top-k features are selected from model coefficients/importances
    - Final evaluation uses KNN with Manhattan distance, matching the manuscript
    - Optional comparison with existing GAM λ=0.6 results in results/bench_foldsafe_full.jsonl

This script is intended as a supplementary boundary comparison. It does not
replace the main GAM benchmark.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Keep thread usage controlled.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
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
    from Filter import load_csv
except Exception as exc:
    raise RuntimeError(
        "Failed to import Filter.load_csv. Please run from the repository root "
        "or keep this script under experiments/. Expected directories: utils/, data/."
    ) from exc


DATASETS = [
    "colon(62,2000).csv",
    "DLBCL(77,5469).csv",
    "9_Tumors(60,5726).csv",
    "Brain_Tumor1(90,5920).csv",
    "Brain_Tumor_2(50,10368).csv",
    "arcene(200,10000).csv",
    "Lung(202,12601).csv",
]


def dataset_name(ds_file: str) -> str:
    return ds_file.split("(")[0]


def make_cv(y, n_splits: int, seed: int):
    """Use stratified CV when class counts permit; otherwise fall back to KFold."""
    _, counts = np.unique(y.astype(int), return_counts=True)
    if counts.min() >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def top_k_from_importance(importance, k: int):
    """Return top-k indices, robust to NaN/inf and all-zero scores."""
    imp = np.asarray(importance, dtype=float).ravel()
    imp[~np.isfinite(imp)] = 0.0

    if imp.size == 0:
        return np.array([], dtype=int)

    k = min(k, imp.size)
    # Stable tie-breaking by feature index.
    order = np.lexsort((np.arange(imp.size), -imp))
    return np.sort(order[:k].astype(int))


def coefficient_importance(model):
    """Convert logistic-regression coefficients to feature importance."""
    coef = np.asarray(model.coef_, dtype=float)
    if coef.ndim == 1:
        return np.abs(coef)
    # For binary and multiclass: aggregate across one-vs-rest/multinomial classes.
    return np.mean(np.abs(coef), axis=0)


def build_methods(seed: int, rf_trees: int):
    """Fresh model templates. clone() is used before each fold."""
    return {
        # liblinear is usually robust for small-n high-D L1 logistic regression.
        "LASSO": LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
        ),
        # Elastic net requires saga.
        "ElasticNet": LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=1.0,
            l1_ratio=0.5,
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=rf_trees,
            random_state=seed,
            class_weight="balanced",
            n_jobs=1,
        ),
    }


def load_gam_reference(results_dir: Path, target_datasets):
    """Load existing GAM λ=0.6 fold-safe reference from bench_foldsafe_full.jsonl if present."""
    path = results_dir / "bench_foldsafe_full.jsonl"
    gam = {}
    if not path.exists():
        return gam

    target = set(target_datasets)
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(d, dict) or "error" in d:
                continue

            ds = d.get("dataset")
            mode = d.get("mode")
            protocol = d.get("protocol")

            # Accept several common spellings used in earlier result files.
            is_gam = mode in {"GAM", "GAM λ=0.6", "GAM-lambda0.6", "GAM_lambda0.6"}
            is_foldsafe = protocol in {"fold-safe", "foldsafe", "Fold-safe", None}

            if ds in target and is_gam and is_foldsafe and "acc_mean" in d:
                gam.setdefault(ds, []).append(float(d["acc_mean"]))

    return {ds: float(np.mean(vals)) for ds, vals in gam.items() if vals}


def evaluate_embedded_dataset(ds_file, methods, outer_folds, knn_k, seed, k_ratio, k_max):
    """Run all embedded baselines on one dataset."""
    ds = dataset_name(ds_file)
    x_path = DATA_DIR / ds_file
    X_raw, y_raw = load_csv(str(x_path))

    if y_raw.dtype.kind in ("U", "S", "O"):
        y_raw = LabelEncoder().fit_transform(y_raw)
    y_raw = y_raw.astype(int)

    D = X_raw.shape[1]
    k = min(k_max, max(1, int(D * k_ratio)))

    folds = list(make_cv(y_raw, outer_folds, seed).split(X_raw, y_raw))
    fold_results = {m: [] for m in methods}
    fold_counts = {m: [] for m in methods}

    for fold_id, (tr, te) in enumerate(folds):
        fold_seed = seed * 100 + fold_id

        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X_raw[tr].copy())
        X_te = scaler.transform(X_raw[te].copy())
        y_tr, y_te = y_raw[tr], y_raw[te]

        for method_name, template in methods.items():
            model = clone(template)

            # Update fold-specific random_state when supported.
            if hasattr(model, "random_state"):
                model.set_params(random_state=fold_seed)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(X_tr, y_tr)

            if method_name == "RandomForest":
                importance = model.feature_importances_
            else:
                importance = coefficient_importance(model)

            top_k = top_k_from_importance(importance, k)

            if len(top_k) == 0:
                acc = 0.0
            else:
                clf = KNeighborsClassifier(n_neighbors=knn_k, metric="manhattan")
                clf.fit(X_tr[:, top_k], y_tr)
                acc = float(clf.score(X_te[:, top_k], y_te))

            fold_results[method_name].append(acc)
            fold_counts[method_name].append(len(top_k))

    rows = []
    for method_name in methods:
        rows.append(
            {
                "script_version": "embedded_baselines_v2_foldsafe_topk",
                "dataset": ds,
                "dataset_file": ds_file,
                "method": method_name,
                "protocol": "fold-safe",
                "selector": "embedded-top-k",
                "classifier": f"KNN(k={knn_k}, Manhattan)",
                "D": int(D),
                "k": int(k),
                "outer_folds": int(outer_folds),
                "acc_mean": float(np.mean(fold_results[method_name])),
                "acc_std": float(np.std(fold_results[method_name])),
                "selected_count_mean": float(np.mean(fold_counts[method_name])),
                "selected_count_std": float(np.std(fold_counts[method_name])),
            }
        )
    return rows


def print_summary(rows, gam_ref):
    datasets = DATASETS
    methods = ["LASSO", "ElasticNet", "RandomForest"]

    by_ds_method = {(r["dataset"], r["method"]): r for r in rows}

    print()
    print(
        f"{'Dataset':20s} {'LASSO':>10s} {'ElasticNet':>10s} "
        f"{'RF':>10s} {'GAM(E2)':>10s} {'Winner':>12s}"
    )
    print("-" * 78)

    for ds_file in datasets:
        ds = dataset_name(ds_file)
        vals = []
        for m in methods:
            vals.append(by_ds_method.get((ds, m), {}).get("acc_mean", np.nan))
        gam = gam_ref.get(ds, np.nan)

        candidates = {methods[i]: vals[i] for i in range(len(methods)) if np.isfinite(vals[i])}
        if np.isfinite(gam):
            candidates["GAM(E2)"] = gam
        winner = max(candidates, key=candidates.get) if candidates else "NA"

        print(
            f"{ds:20s} "
            f"{vals[0]:10.4f} {vals[1]:10.4f} {vals[2]:10.4f} "
            f"{gam:10.4f} {winner:>12s}"
        )

    print("-" * 78)
    for m in methods:
        accs = [
            by_ds_method[(dataset_name(ds_file), m)]["acc_mean"]
            for ds_file in datasets
            if (dataset_name(ds_file), m) in by_ds_method
        ]
        print(f"{m:14s} macro acc: {np.mean(accs):.4f}")

    gam_accs = [gam_ref[dataset_name(ds_file)] for ds_file in datasets if dataset_name(ds_file) in gam_ref]
    if gam_accs:
        print(f"{'GAM(E2)':14s} macro acc: {np.mean(gam_accs):.4f}")
    else:
        print("GAM(E2) reference not found in results/bench_foldsafe_full.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Fold-safe embedded feature-selection baselines.")
    parser.add_argument("--quick", action="store_true", help="Quick mode: first 2 datasets only.")
    parser.add_argument("--rf-trees", type=int, default=200, help="Number of trees for RandomForest.")
    parser.add_argument("--outer-folds", type=int, default=5, help="Outer CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--knn-k", type=int, default=5, help="KNN k.")
    parser.add_argument("--k-ratio", type=float, default=0.05, help="Feature budget ratio.")
    parser.add_argument("--k-max", type=int, default=200, help="Maximum feature budget.")
    args = parser.parse_args()

    datasets = DATASETS[:2] if args.quick else DATASETS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "bench_embedded_baselines.jsonl"
    summary_path = RESULTS_DIR / "bench_embedded_baselines_summary.jsonl"

    methods = build_methods(args.seed, args.rf_trees)
    gam_ref = load_gam_reference(RESULTS_DIR, [dataset_name(x) for x in DATASETS])

    print("Fold-safe embedded feature-selection baselines")
    print("=" * 78)
    print(f"Datasets     : {len(datasets)}")
    print(f"Methods      : {', '.join(methods.keys())}")
    print(f"Outer CV     : {args.outer_folds}-fold")
    print(f"Feature budget: k=min({args.k_max}, int({args.k_ratio}D))")
    print(f"Classifier   : KNN(k={args.knn_k}), Manhattan")
    print(f"Output       : {out_path}")
    print("=" * 78)

    all_rows = []
    t0 = time.time()

    for i, ds_file in enumerate(datasets, start=1):
        ds_t0 = time.time()
        rows = evaluate_embedded_dataset(
            ds_file=ds_file,
            methods=methods,
            outer_folds=args.outer_folds,
            knn_k=args.knn_k,
            seed=args.seed,
            k_ratio=args.k_ratio,
            k_max=args.k_max,
        )
        all_rows.extend(rows)
        msg = ", ".join([f"{r['method']}={r['acc_mean']:.4f}" for r in rows])
        print(f"[{i}/{len(datasets)}] {dataset_name(ds_file):14s} {msg} ({time.time()-ds_t0:.1f}s)", flush=True)

    # Save detailed rows.
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save one summary row per dataset.
    methods_names = list(methods.keys())
    by_ds_method = {(r["dataset"], r["method"]): r for r in all_rows}
    with open(summary_path, "w", encoding="utf-8") as f:
        for ds_file in datasets:
            ds = dataset_name(ds_file)
            row = {"dataset": ds, "GAM_E2_acc": gam_ref.get(ds)}
            for m in methods_names:
                rec = by_ds_method.get((ds, m), {})
                row[f"{m}_acc"] = rec.get("acc_mean")
                row[f"{m}_acc_std"] = rec.get("acc_std")
            vals = {m: row.get(f"{m}_acc") for m in methods_names}
            if row["GAM_E2_acc"] is not None:
                vals["GAM(E2)"] = row["GAM_E2_acc"]
            vals = {k: v for k, v in vals.items() if v is not None and np.isfinite(v)}
            row["winner"] = max(vals, key=vals.get) if vals else None
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print_summary(all_rows, gam_ref)
    print(f"\nSaved detailed results: {out_path}")
    print(f"Saved summary results : {summary_path}")
    print(f"Total runtime         : {time.time()-t0:.1f}s")
    print("\nNote: These embedded methods are supplementary boundary baselines under the same fold-safe top-k KNN evaluation protocol.")


if __name__ == "__main__":
    main()
