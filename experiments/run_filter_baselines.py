#!/usr/bin/env python3
"""
Filter Baselines — MI, ReliefF, mRMR, F-score
Fold-safe: filter on outer_train, evaluate on outer_test.
k = dataset-specific average number of features selected by GAM-λ0.6.
For fair comparison, each filter selects the same number of features as GAM.
Output: bench_foldsafe_filter.jsonl — ~10 min with 2 workers.
"""
import sys, os, json, time, numpy as np
import multiprocessing as mp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_ROOT, 'code')
UTILS_DIR = os.path.join(REPO_ROOT, 'utils')
DATA_DIR = os.path.join(REPO_ROOT, 'data')
sys.path.insert(0, CODE_DIR); sys.path.insert(0, UTILS_DIR); os.chdir(CODE_DIR)

from Filter import load_csv, reliefFScore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif as mi_c, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

RES_DIR = os.path.join(REPO_ROOT, 'results')
OUT = f'{RES_DIR}/bench_foldsafe_filter.jsonl'

DATASETS = [
    'colon(62,2000).csv', 'DLBCL(77,5469).csv', '9_Tumors(60,5726).csv',
    'Brain_Tumor1(90,5920).csv', 'Brain_Tumor_2(50,10368).csv',
    'arcene(200,10000).csv', 'Lung(202,12601).csv',
]
# k = GAM λ=0.6 average selected features per dataset (fair comparison)
FILTER_K = {
    'colon': 100, 'DLBCL': 273, '9_Tumors': 286,
    'Brain_Tumor1': 296, 'Brain_Tumor_2': 518, 'arcene': 500, 'Lung': 630,
}
FILTERS = ['MI', 'ReliefF', 'mRMR', 'F-score']
OUTER_FOLDS, KNN_K = 5, 5


def mrmr_select(X, Y, k):
    """Windowed mRMR with batch correlation (optimized for high-D)."""
    p = X.shape[1]; k = min(k, p)
    if k <= 0: return np.array([], dtype=int)
    mi = mi_c(X, Y, random_state=42)
    mi_norm = (mi - mi.min()) / (mi.max() - mi.min() + 1e-12)
    X_std = (X - X.mean(0)) / (X.std(0) + 1e-12)
    selected = [int(np.argmax(mi_norm))]
    remaining = np.array([i for i in range(p) if i != selected[0]])

    while len(selected) < k:
        recent = selected[-20:]
        best_val = -np.inf; best_idx_in_rem = 0

        # Batch: compute correlation of recent features vs all remaining in chunks
        chunk = 500
        for start in range(0, len(remaining), chunk):
            rem_chunk = remaining[start:start + chunk]
            # corrcoef of (|recent|+|chunk|) × n samples
            combined = np.vstack([X_std[:, recent].T, X_std[:, rem_chunk].T])
            corr = np.abs(np.corrcoef(combined))
            red_block = corr[:len(recent), len(recent):].mean(axis=0)  # avg redundancy per candidate
            scores = mi_norm[rem_chunk] - red_block
            local_best = int(np.argmax(scores))
            if scores[local_best] > best_val:
                best_val = scores[local_best]
                best_idx_in_rem = start + local_best

        selected.append(int(remaining[best_idx_in_rem]))
        remaining = np.delete(remaining, best_idx_in_rem)

    return np.array(selected)


def run_one(args):
    ds_file, filter_name = args
    ds_name = ds_file.split('(')[0]
    try:
        X_raw, Y_raw = load_csv(f'{DATA_DIR}/{ds_file}')
        if Y_raw.dtype.kind in ('U', 'S', 'O'): Y_raw = LabelEncoder().fit_transform(Y_raw)
        Y_raw = Y_raw.astype(int)
        D = X_raw.shape[1]
        k = min(FILTER_K[ds_name], D)

        accs = []
        skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42)
        for tr, te in skf.split(X_raw, Y_raw):
            sf = MinMaxScaler()
            X_tr = sf.fit_transform(X_raw[tr].copy())
            X_te = sf.transform(X_raw[te])
            Y_tr, Y_te = Y_raw[tr], Y_raw[te]

            # Filter on outer_train only (fold-safe)
            if filter_name == 'MI':
                scores = mi_c(X_tr, Y_tr, random_state=42)
                scores = np.nan_to_num(scores, nan=-np.inf, posinf=0.0, neginf=-np.inf)
                sel = np.argsort(-scores)[:k]
            elif filter_name == 'ReliefF':
                scores = reliefFScore(X_tr, Y_tr)
                scores = np.nan_to_num(scores, nan=-np.inf, posinf=0.0, neginf=-np.inf)
                sel = np.argsort(-scores)[:k]
            elif filter_name == 'mRMR':
                sel = mrmr_select(X_tr, Y_tr, k)
            elif filter_name == 'F-score':
                scores, _ = f_classif(X_tr, Y_tr)
                scores = np.nan_to_num(scores, nan=-np.inf, posinf=0.0, neginf=-np.inf)
                sel = np.argsort(-scores)[:k]

            knn = KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1)
            acc = float(knn.fit(X_tr[:, sel], Y_tr).score(X_te[:, sel], Y_te)) if len(sel) > 0 else 0.0
            accs.append(acc)

        return [{'dataset': ds_name, 'filter': filter_name, 'D': D, 'k': k,
                 'k_source': 'GAM-lambda0.6-average-selected-features',
                 'protocol': 'fold-safe', 'outer_cv': '5-fold stratified',
                 'classifier': 'KNN(k=5, Manhattan)',
                 'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
                 'per_fold_acc': accs}]
    except Exception as e:
        import traceback; traceback.print_exc()
        return [{'dataset': ds_name, 'filter': filter_name, 'error': str(e)[:200]}]


if __name__ == '__main__':
    tasks = [(ds, f) for ds in DATASETS for f in FILTERS]
    total = len(tasks)  # 28
    print(f'{len(DATASETS)}DS × {len(FILTERS)}filters = {total} tasks')
    print(f'5-fold CV, k=5%×D, KNN Manhattan, Output: {OUT}')

    t0 = time.time()
    with mp.Pool(4) as pool:
        for i, results in enumerate(pool.imap_unordered(run_one, tasks)):
            with open(OUT, 'a') as f:
                for r in results: f.write(json.dumps(r) + '\n')
            r0 = results[0]
            status = f'ERROR: {r0["error"][:50]}' if 'error' in r0 else f'acc={r0.get("acc_mean",0):.3f}'
            print(f'[{i+1}/{total}] {r0.get("dataset","?"):14s} {r0.get("filter","?"):8s} {status} ({time.time()-t0:.0f}s)', flush=True)

    print(f'\nDone: {time.time()-t0:.0f}s → {OUT}')
