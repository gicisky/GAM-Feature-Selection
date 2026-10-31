"""Supplementary robustness endpoints for GAM.

Purpose
-------
This script runs a lightweight fold-safe robustness check that reports
Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1, redundancy, selected count,
and runtime. It is designed as a supplementary endpoint analysis, not as a
replacement for the main 15-dataset fold-safe benchmark.

Recommended ASOC-ready protocol
-------------------------------
Datasets: colon, DLBCL, 9_Tumors, Brain_Tumor_2, SRBCT, arcene, Lung
Methods: Bare-GA and GAM-GA(lambda=0.6)
Runs: 3
Outer CV: 5-fold
Inner CV: 3-fold
N=50, T=50

The wrapper fitness remains inner-CV KNN accuracy. Balanced accuracy and F1 are
computed only on the held-out outer fold after feature selection.
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
ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / 'code'
UTILS_DIR = ROOT / 'utils'
DATA_DIR = ROOT / 'data'
EXP_DIR = ROOT
RES_DIR = EXP_DIR / 'results'
OUT_DEFAULT = RES_DIR / 'bench_multimetric_robustness_clean.jsonl'
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(UTILS_DIR))
os.chdir(CODE_DIR)
import bench_gam
from bench_bare import ga_bare
from bench_gam import GAM, ga_gam_safe
from Filter import load_csv, reliefFScore
DATASETS = ['colon(62,2000).csv', 'DLBCL(77,5469).csv', '9_Tumors(60,5726).csv', 'Brain_Tumor_2(49,10368).csv', 'SRBCT(83,2308).csv', 'arcene(200,10000).csv', 'Lung(202,12601).csv']

def dataset_name(ds_file: str) -> str:
    return ds_file.split('(')[0]

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

def compute_hmrf(X, y, seed):
    mi = mi_c(X, y, random_state=seed)
    rf = reliefFScore(X, y)
    mi_n = (mi - mi.min()) / (mi.max() - mi.min() + 1e-12)
    rf_n = (rf - rf.min()) / (rf.max() - rf.min() + 1e-12)
    return 0.5 * mi_n + 0.5 * rf_n

def evaluate_outer_fold(X_tr, y_tr, X_te, y_te, selected, knn_k):
    selected = np.asarray(selected, dtype=int)
    if len(selected) == 0:
        y_pred = np.full_like(y_te, fill_value=np.bincount(y_tr).argmax())
    else:
        clf = KNeighborsClassifier(knn_k, metric='manhattan', n_jobs=1)
        clf.fit(X_tr[:, selected], y_tr)
        y_pred = clf.predict(X_te[:, selected])
    return {'accuracy': float(accuracy_score(y_te, y_pred)), 'balanced_accuracy': float(balanced_accuracy_score(y_te, y_pred)), 'macro_f1': float(f1_score(y_te, y_pred, average='macro', zero_division=0)), 'weighted_f1': float(f1_score(y_te, y_pred, average='weighted', zero_division=0))}

def patch_inner_cv_evaluator(X_tr, y_tr, inner_folds, knn_k, seed):

    def inner_eval(_X, _y, idx):
        idx = np.asarray(idx, dtype=int)
        if len(idx) == 0:
            return 0.0
        clf = KNeighborsClassifier(knn_k, metric='manhattan', n_jobs=1)
        cv = KFold(inner_folds, shuffle=True, random_state=seed)
        return float(np.mean(cross_val_score(clf, X_tr[:, idx], y_tr, cv=cv)))
    return inner_eval

def run_fold_safe_task(ds_file, mode, run_idx, args):
    ds_name = dataset_name(ds_file)
    X_raw, y_raw = load_csv(str(DATA_DIR / ds_file))
    if y_raw.dtype.kind in ('U', 'S', 'O'):
        y_raw = LabelEncoder().fit_transform(y_raw)
    y_raw = y_raw.astype(int)
    D = int(X_raw.shape[1])
    k = max(1, int(D * args.k_ratio))
    run_seed = args.seed + run_idx * 17
    outer = StratifiedKFold(args.outer_folds, shuffle=True, random_state=run_seed)
    fold_records = []
    selected_all = []
    start = time.time()
    for fold_idx, (tr, te) in enumerate(outer.split(X_raw, y_raw)):
        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X_raw[tr].copy())
        X_te = scaler.transform(X_raw[te])
        y_tr, y_te = (y_raw[tr], y_raw[te])
        orig_evaluate = bench_gam.evaluate
        bench_gam.evaluate = patch_inner_cv_evaluator(X_tr, y_tr, args.inner_folds, args.knn_k, run_seed + fold_idx)
        np.random.seed(run_seed + fold_idx)
        try:
            if mode == 'Bare-GA':
                selected = ga_bare(X_tr, y_tr, k, args.N, args.T)
            elif mode == 'GAM-GA':
                scores = compute_hmrf(X_tr, y_tr, run_seed + fold_idx)
                gam = GAM(X_tr, scores)
                original_crossover = gam.safe_crossover

                def crossover_with_lambda(p1, p2, kk, gen=0, max_gen=100, **kwargs):
                    return original_crossover(p1, p2, kk, gen=gen, max_gen=max_gen, penalty=args.lam)
                gam.safe_crossover = crossover_with_lambda
                selected = ga_gam_safe(X_tr, y_tr, k, gam, args.N, args.T)
            else:
                raise ValueError(f'Unsupported mode: {mode}')
        finally:
            bench_gam.evaluate = orig_evaluate
        metrics = evaluate_outer_fold(X_tr, y_tr, X_te, y_te, selected, args.knn_k)
        metrics.update({'fold': fold_idx, 'redundancy': safe_redundancy(X_tr, selected), 'selected_count': int(len(selected)), 'selected_features': [int(x) for x in np.asarray(selected, dtype=int)]})
        fold_records.append(metrics)
        selected_all.append(metrics['selected_features'])
    elapsed = time.time() - start
    result = {'dataset': ds_name, 'dataset_file': ds_file, 'mode': mode, 'base': 'GA', 'run': int(run_idx), 'protocol': 'fold-safe multimetric robustness', 'D': D, 'k': int(k), 'N': int(args.N), 'T': int(args.T), 'lambda': None if mode == 'Bare-GA' else float(args.lam), 'outer_folds': int(args.outer_folds), 'inner_folds': int(args.inner_folds), 'knn_k': int(args.knn_k), 'runtime_sec': float(elapsed), 'accuracy_mean': float(np.mean([x['accuracy'] for x in fold_records])), 'accuracy_std': float(np.std([x['accuracy'] for x in fold_records])), 'balanced_accuracy_mean': float(np.mean([x['balanced_accuracy'] for x in fold_records])), 'balanced_accuracy_std': float(np.std([x['balanced_accuracy'] for x in fold_records])), 'macro_f1_mean': float(np.mean([x['macro_f1'] for x in fold_records])), 'macro_f1_std': float(np.std([x['macro_f1'] for x in fold_records])), 'weighted_f1_mean': float(np.mean([x['weighted_f1'] for x in fold_records])), 'weighted_f1_std': float(np.std([x['weighted_f1'] for x in fold_records])), 'redundancy_mean': float(np.mean([x['redundancy'] for x in fold_records])), 'redundancy_std': float(np.std([x['redundancy'] for x in fold_records])), 'selected_count_mean': float(np.mean([x['selected_count'] for x in fold_records])), 'fold_records': fold_records, 'selected_features_by_fold': selected_all}
    return result

def load_completed(out_path):
    completed = set()
    if not out_path.exists():
        return completed
    with out_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if 'error' not in row:
                completed.add((row['dataset_file'], row['mode'], int(row['run'])))
    return completed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(OUT_DEFAULT))
    parser.add_argument('--datasets', nargs='*', default=DATASETS)
    parser.add_argument('--modes', nargs='*', default=['Bare-GA', 'GAM-GA'])
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--outer-folds', type=int, default=5)
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--N', type=int, default=50)
    parser.add_argument('--T', type=int, default=50)
    parser.add_argument('--k-ratio', type=float, default=0.05)
    parser.add_argument('--lam', type=float, default=0.6)
    parser.add_argument('--knn-k', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20260713)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.N < 4:
        raise SystemExit('N must be at least 4 for the GA tournament/update operators.')
    if args.T < 1:
        raise SystemExit('T must be at least 1.')
    if args.num_shards < 1:
        raise SystemExit('--num-shards must be at least 1.')
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit('--shard-index must satisfy 0 <= shard-index < num-shards.')
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and out_path.exists():
        out_path.unlink()
    tasks = [(ds, mode, run) for ds in args.datasets for mode in args.modes for run in range(args.runs)]
    tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard_index]
    completed = load_completed(out_path)
    tasks = [t for t in tasks if (t[0], t[1], t[2]) not in completed]
    print('Supplementary multimetric robustness experiment')
    print(f'Output: {out_path}')
    print(f'Remaining tasks: {len(tasks)}')
    print(f'Protocol: {len(args.datasets)} datasets, modes={args.modes}, runs={args.runs}, outer={args.outer_folds}, inner={args.inner_folds}, N={args.N}, T={args.T}, lambda={args.lam}')
    print(f'Shard: {args.shard_index + 1}/{args.num_shards}')
    t0 = time.time()
    for i, (ds, mode, run) in enumerate(tasks, start=1):
        try:
            row = run_fold_safe_task(ds, mode, run, args)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            row = {'dataset': dataset_name(ds), 'dataset_file': ds, 'mode': mode, 'run': int(run), 'error': str(exc)[:500]}
        with out_path.open('a') as f:
            f.write(json.dumps(row) + '\n')
        if 'error' in row:
            status = f"ERROR {row['error'][:80]}"
        else:
            status = f"acc={row['accuracy_mean']:.4f} bacc={row['balanced_accuracy_mean']:.4f} mf1={row['macro_f1_mean']:.4f}"
        elapsed = time.time() - t0
        print(f'[{i}/{len(tasks)}] {dataset_name(ds):14s} {mode:7s} run={run} {status} elapsed={elapsed / 60:.1f}m', flush=True)
if __name__ == '__main__':
    main()
