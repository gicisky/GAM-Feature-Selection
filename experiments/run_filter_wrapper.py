#!/usr/bin/env python3
"""
Filter-guided GAM variants: mRMR/ReliefF/HMRF scoring under same GAM operators.
7DS × 2 bases (GA,DE) × 3 methods × 3 runs. N=50,T=50, single split, inner-3fold.
Output: bench_filter_guided_gam.jsonl
"""
import os, sys, json, time, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing as mp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_ROOT, 'code')
UTILS_DIR = os.path.join(REPO_ROOT, 'utils')
DATA_DIR = os.path.join(REPO_ROOT, 'data')
sys.path.insert(0, CODE_DIR); sys.path.insert(0, UTILS_DIR); os.chdir(CODE_DIR)

import bench_gam
from bench_gam import GAM, ga_gam_safe
from Filter import load_csv, reliefFScore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif as mi_c
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

RES_DIR = os.path.join(REPO_ROOT, 'results')
OUT = f'{RES_DIR}/bench_filter_guided_gam.jsonl'

DATASETS = [
    'colon(62,2000).csv', 'DLBCL(77,5469).csv',
    '9_Tumors(60,5726).csv', 'Brain_Tumor1(90,5920).csv',
    'Brain_Tumor_2(49,10368).csv', 'arcene(200,10000).csv',
    'Lung(202,12601).csv',
]
BASES = ['GA', 'DE']
METHODS = ['mRMR-guided-GAM', 'ReliefF-guided-GAM', 'HMRF-GAM']
NR, KNN_K, INNER_FOLDS = 3, 5, 3
N, T = 50, 50
PRE_N = 5000  # only for D > 10000


def safe_redundancy(X, idx):
    if len(idx) <= 1:
        return 0.0
    Xp = X[:, idx]
    Xc = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-12)
    C = np.nan_to_num(np.abs(np.corrcoef(Xc.T)), nan=0.0, posinf=0.0, neginf=0.0)
    iu = np.triu_indices_from(C, k=1)
    return float(np.mean(C[iu]))


def make_inner_cv(y, n_splits=3):
    _, counts = np.unique(y.astype(int), return_counts=True)
    if counts.min() >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def mrmr_ranking(X, Y, max_features=None):
    """
    Approximate mRMR feature ranking using batch correlation.
    Redundancy estimated against last min(50,|S|) selected features.
    Returns per-feature scores in [0,1]. Unranked features receive 0.
    """
    p = X.shape[1]
    limit = min(max_features, p) if max_features else p
    X_std = (X - X.mean(0)) / (X.std(0) + 1e-12)

    mi = mi_c(X, Y, random_state=42)
    mi_norm = (mi - mi.min()) / (mi.max() - mi.min() + 1e-12)

    selected = [int(np.argmax(mi_norm))]
    remaining = np.array([i for i in range(p) if i != selected[0]])

    while len(selected) < limit and len(remaining) > 0:
        recent = selected[-min(50, len(selected)):]
        best_val = -np.inf
        best_idx_in_rem = 0
        chunk = 500
        for start in range(0, len(remaining), chunk):
            rem_chunk = remaining[start:start + chunk]
            combined = np.vstack([X_std[:, recent].T, X_std[:, rem_chunk].T])
            corr = np.abs(np.corrcoef(combined))
            red_block = corr[:len(recent), len(recent):].mean(axis=0)
            scores_chunk = mi_norm[rem_chunk] - red_block
            local_best = int(np.argmax(scores_chunk))
            if scores_chunk[local_best] > best_val:
                best_val = scores_chunk[local_best]
                best_idx_in_rem = start + local_best
        selected.append(int(remaining[best_idx_in_rem]))
        remaining = np.delete(remaining, best_idx_in_rem)

    scores = np.zeros(p)
    for rank, feat in enumerate(selected):
        scores[feat] = 1.0 - rank / limit
    return scores


def de_gam_safe(X, Y, k, gam, N=100, T=100, CR=0.9):
    p = gam.p; k_eff = min(k, p)
    pop = gam.safe_init(N, k_eff)
    fit = np.array([bench_gam.evaluate(X, Y, np.where(pop[i])[0]) for i in range(N)])
    bi = fit.argmax(); best_fit, best = fit[bi], pop[bi].copy()
    for t in range(T):
        for i in range(N):
            cands = [j for j in range(N) if j != i]
            a, b, c = np.random.choice(cands, 3, replace=False)
            diff = pop[b] != pop[c]; donor = pop[a].copy(); donor[diff] = pop[b][diff]
            jrand = np.random.randint(p); trial = pop[i].copy()
            for j in range(p):
                if np.random.rand() < CR or j == jrand: trial[j] = donor[j]
            child = gam.safe_crossover(pop[i], trial, k_eff, gen=t, max_gen=T)
            child = gam.safe_mutate(child, k_eff)
            nf = bench_gam.evaluate(X, Y, np.where(child)[0])
            if nf >= fit[i]: pop[i] = child; fit[i] = nf
            if nf > best_fit: best_fit = nf; best = child.copy()
    return np.where(best)[0]


BASE_FN = {'GA': ga_gam_safe, 'DE': de_gam_safe}


def run_one_task(args):
    ds_file, base_name, method, run_idx = args
    ds_name = ds_file.split('(')[0]
    try:
        X_raw, Y_raw = load_csv(f'{DATA_DIR}/{ds_file}')
        if Y_raw.dtype.kind in ('U', 'S', 'O'):
            Y_raw = LabelEncoder().fit_transform(Y_raw)
        Y_raw = Y_raw.astype(int)
        D_orig = X_raw.shape[1]; k = min(200, int(D_orig * 0.05))
        seed = 42 + run_idx * 7

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        tr, te = next(skf.split(X_raw, Y_raw))

        sf = MinMaxScaler()
        X_tr_full = sf.fit_transform(X_raw[tr].copy())
        X_te_full = sf.transform(X_raw[te])
        Y_tr, Y_te = Y_raw[tr], Y_raw[te]

        # MI pre-filter for D > 10000
        if D_orig > 10000:
            mi_pre = mi_c(X_tr_full, Y_tr, random_state=42)
            pre_idx = np.argsort(-mi_pre)[:min(PRE_N, D_orig)]
            X_tr = X_tr_full[:, pre_idx]
            X_te = X_te_full[:, pre_idx]
            D_reduced = len(pre_idx)
        else:
            X_tr, X_te = X_tr_full, X_te_full
            D_reduced = D_orig

        # HMRF and ReliefF scores (shared across methods)
        rf = reliefFScore(X_tr, Y_tr)
        mi_fold = mi_c(X_tr, Y_tr, random_state=42)
        hmrf = (0.5 * (mi_fold - mi_fold.min()) / (mi_fold.max() - mi_fold.min() + 1e-12)
                + 0.5 * (rf - rf.min()) / (rf.max() - rf.min() + 1e-12))
        rf_norm = (rf - rf.min()) / (rf.max() - rf.min() + 1e-12)

        # Build scores and penalty per method
        if method == 'HMRF-GAM':
            scores = hmrf
            penalty = 0.6
        elif method == 'ReliefF-guided-GAM':
            scores = rf_norm
            penalty = 0.0
        else:  # mRMR-guided-GAM
            scores = mrmr_ranking(X_tr, Y_tr, max_features=min(2000, D_reduced))
            penalty = 0.0

        # Run optimizer
        gam = GAM(X_tr, scores)
        o = gam.safe_crossover
        gam.safe_crossover = lambda p1, p2, k, g=0, mg=100, **kw: o(
            p1, p2, k, gen=g, max_gen=mg, penalty=penalty)

        orig_eval = bench_gam.evaluate
        try:
            bench_gam.evaluate = lambda X_, Y_, idx: (
                0.0 if len(idx) == 0 else float(np.mean(cross_val_score(
                    KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1),
                    X_[:, idx], Y_,
                    cv=make_inner_cv(Y_, INNER_FOLDS)))))
            np.random.seed(seed)
            t0 = time.time()
            sol = BASE_FN[base_name](X_tr, Y_tr, k, gam, N, T)
            elapsed = time.time() - t0
        finally:
            bench_gam.evaluate = orig_eval

        # Test accuracy + redundancy
        knn = KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1)
        test_acc = float(knn.fit(X_tr[:, sol], Y_tr).score(X_te[:, sol], Y_te)) if len(sol) > 0 else 0.0
        red = safe_redundancy(X_tr, sol)

        return [{
            'dataset': ds_name, 'base': base_name, 'method': method,
            'run': run_idx, 'D_orig': D_orig, 'D_reduced': D_reduced,
            'k': k, 'N': N, 'T': T, 'penalty': penalty,
            'test_acc': test_acc, 'n_sel': int(len(sol)),
            'red': red if not np.isnan(red) else 0.0,
            'elapsed_s': elapsed,
        }]
    except Exception as e:
        import traceback; traceback.print_exc()
        return [{'dataset': ds_name, 'base': base_name, 'method': method,
                 'run': run_idx, 'error': str(e)[:200]}]


if __name__ == '__main__':
    tasks = [(ds, b, m, r) for ds in DATASETS for b in BASES
             for m in METHODS for r in range(NR)]
    total = len(tasks)  # 7×2×3×3 = 126

    completed = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for l in f:
                d = json.loads(l)
                if 'error' not in d:
                    completed.add((d['dataset'], d['base'], d['method'], d['run']))
        print(f'Resuming: {len(completed)} tasks done')

    tasks_pending = [t for t in tasks if (t[0].split('(')[0], t[1], t[2], t[3]) not in completed]
    print(f'{len(DATASETS)}DS × {len(BASES)}bases × {len(METHODS)}methods × {NR}runs = {len(tasks_pending)} remaining')
    print(f'N={N} T={T} inner-{INNER_FOLDS}fold, MI-pre D>10000, Output: {OUT}')

    t0 = time.time()
    with mp.Pool(4, maxtasksperchild=1) as pool:
        for i, results in enumerate(pool.imap_unordered(run_one_task, tasks_pending)):
            with open(OUT, 'a') as f:
                for r in results:
                    f.write(json.dumps(r) + '\n')
            r0 = results[0]
            if 'error' in r0:
                status = f"ERROR: {r0['error'][:50]}"
            else:
                status = f"test_acc={r0['test_acc']:.4f} red={r0.get('red',0):.4f} n={r0['n_sel']} ({r0['elapsed_s']:.0f}s)"
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(tasks_pending) - i - 1) if i > 0 and len(tasks_pending) > i + 1 else 0
            remaining = len(tasks_pending) - i - 1
            print(f'[{i+1}/{len(tasks_pending)}] {r0.get("dataset","?"):14s} {r0.get("base","?"):6s} '
                  f'{r0.get("method","?"):20s} {status} ({elapsed/60:.0f}m, {remaining} left)', flush=True)

    print(f'\nDone: {time.time()-t0:.0f}s → {OUT}')
