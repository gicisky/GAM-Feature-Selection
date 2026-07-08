#!/usr/bin/env python3
"""
Independent wrapper baselines: BGWO + BPSO vs GAM-GA.
Standard binary metaheuristics (no GAM operators, no HMRF, no λ).
7DS × 3 methods × 3 runs. N=50,T=50, single split, inner-3fold.
Output: bench_independent_wrappers.jsonl
"""
import os, sys, json, time, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing as mp

CODE_DIR = '/Users/gicisky/Desktop/论文研读分析/crow/crow_search1/code'
FILTER_DIR = '/Users/gicisky/Desktop/论文研读分析/crow/crow_search1/filter'
DATA_DIR = '/Users/gicisky/Desktop/论文研读分析/crow/crow_search1/dataCSV_high/all'
sys.path.insert(0, CODE_DIR); sys.path.insert(0, FILTER_DIR); os.chdir(CODE_DIR)

import bench_gam
from bench_gam import GAM, ga_gam_safe
from Filter import load_csv, reliefFScore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif as mi_c
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

RES_DIR = '/Users/gicisky/Desktop/论文研读分析/crow/crow_search1/experiments/gam_20260531/results'
OUT = f'{RES_DIR}/bench_independent_wrappers.jsonl'

DATASETS = [
    'colon(62,2000).csv', 'DLBCL(77,5469).csv',
    '9_Tumors(60,5726).csv', 'Brain_Tumor1(90,5920).csv',
    'Brain_Tumor_2(49,10368).csv', 'arcene(200,10000).csv',
    'Lung(202,12601).csv',
]
METHODS = ['BGWO', 'BPSO', 'GAM-GA']
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


def evaluate_fn(X, Y, idx):
    """Inner KNN-CV fitness (shared by all methods)."""
    if len(idx) == 0:
        return 0.0
    return float(np.mean(cross_val_score(
        KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1),
        X[:, idx], Y,
        cv=make_inner_cv(Y, INNER_FOLDS))))


def topk_binary(scores, k):
    """Convert continuous scores to binary mask with exactly k ones (top-k discretization)."""
    p = len(scores)
    binary = np.zeros(p, dtype=bool)
    top_idx = np.argpartition(scores, -k)[-k:]
    binary[top_idx] = True
    return binary


def k_repair(binary, k):
    """Ensure exactly k features selected."""
    n_sel = np.sum(binary)
    if n_sel == k:
        return binary
    if n_sel > k:
        pos = np.where(binary)[0]
        np.random.shuffle(pos)
        binary[pos[k:]] = False
    else:
        neg = np.where(~binary)[0]
        binary[np.random.choice(neg, k - n_sel, replace=False)] = True
    return binary


# ============================================================
# BGWO: Binary Grey Wolf Optimizer (standard, independent)
# Mirjalili et al., 2014 (GWO) adapted for binary FS
# ============================================================
def bgwo(X, Y, k, N_pop=50, T_max=50):
    """
    Standard binary GWO for feature selection.
    - Random initialization (no HMRF guidance)
    - α/β/δ hierarchy
    - Sigmoid transfer → top-k binary discretization
    - No correlation graph, no λ penalty
    """
    p = X.shape[1]
    k_eff = min(k, p)

    # Random init: each wolf selects exactly k_eff random features
    pop = np.zeros((N_pop, p), dtype=bool)
    for i in range(N_pop):
        sel = np.random.choice(p, k_eff, replace=False)
        pop[i, sel] = True

    fit = np.array([evaluate_fn(X, Y, np.where(pop[i])[0]) for i in range(N_pop)])
    order = np.argsort(-fit)
    pop, fit = pop[order], fit[order]

    # Hierarchy
    alpha = pop[0].astype(float)
    alpha_fit = fit[0]
    beta = pop[1].astype(float) if N_pop > 1 else alpha.copy()
    delta = pop[2].astype(float) if N_pop > 2 else alpha.copy()

    best = pop[0].copy()
    best_fit = fit[0]

    for t in range(T_max):
        a = 2.0 * (1.0 - t / T_max)  # linearly decreases from 2 to 0

        for i in range(N_pop):
            # Alpha guidance
            r1, r2 = np.random.rand(p), np.random.rand(p)
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2
            D_alpha = np.abs(C1 * alpha - pop[i].astype(float))
            X1 = alpha - A1 * D_alpha

            # Beta guidance
            r1, r2 = np.random.rand(p), np.random.rand(p)
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            D_beta = np.abs(C2 * beta - pop[i].astype(float))
            X2 = beta - A2 * D_beta

            # Delta guidance
            r1, r2 = np.random.rand(p), np.random.rand(p)
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            D_delta = np.abs(C3 * delta - pop[i].astype(float))
            X3 = delta - A3 * D_delta

            new_cont = (X1 + X2 + X3) / 3.0
            # Sigmoid → probabilities → top-k binary
            probs = 1.0 / (1.0 + np.exp(-np.clip(new_cont, -50, 50)))
            new_bin = topk_binary(probs, k_eff)

            nf = evaluate_fn(X, Y, np.where(new_bin)[0])
            if nf > fit[i]:
                pop[i] = new_bin.copy()
                fit[i] = nf
            if nf > best_fit:
                best = new_bin.copy()
                best_fit = nf

        # Refresh hierarchy each generation
        order = np.argsort(-fit)
        pop, fit = pop[order], fit[order]
        alpha = pop[0].astype(float)
        beta = pop[1].astype(float) if N_pop > 1 else alpha.copy()
        delta = pop[2].astype(float) if N_pop > 2 else alpha.copy()

    return np.where(best)[0], float(best_fit)


# ============================================================
# BPSO: Binary Particle Swarm Optimization (standard, independent)
# Kennedy & Eberhart, 1997 (BPSO) adapted for FS
# ============================================================
def bpso(X, Y, k, N_pop=50, T_max=50):
    """
    Standard binary PSO for feature selection.
    - Random initialization (no HMRF guidance)
    - Inertia weight linearly decreasing 0.9→0.4
    - Sigmoid transfer → top-k binary discretization
    - No correlation graph, no λ penalty
    """
    p = X.shape[1]
    k_eff = min(k, p)

    # Random init
    pos = np.zeros((N_pop, p), dtype=bool)
    for i in range(N_pop):
        sel = np.random.choice(p, k_eff, replace=False)
        pos[i, sel] = True

    fit = np.array([evaluate_fn(X, Y, np.where(pos[i])[0]) for i in range(N_pop)])
    vel = np.random.uniform(-1, 1, (N_pop, p))

    pbest = pos.copy()
    pbest_fit = fit.copy()
    gbest_idx = np.argmax(fit)
    gbest = pos[gbest_idx].copy()
    gbest_fit = fit[gbest_idx]

    for t in range(T_max):
        w = 0.9 - 0.5 * t / T_max  # inertia: 0.9 → 0.4

        for i in range(N_pop):
            r1, r2 = np.random.rand(p), np.random.rand(p)
            # Velocity update
            vel[i] = (w * vel[i]
                      + 2.0 * r1 * (pbest[i].astype(float) - pos[i].astype(float))
                      + 2.0 * r2 * (gbest.astype(float) - pos[i].astype(float)))
            vel[i] = np.clip(vel[i], -6, 6)
            # Sigmoid → probabilities → top-k binary
            probs = 1.0 / (1.0 + np.exp(-vel[i]))
            new_bin = topk_binary(probs, k_eff)

            nf = evaluate_fn(X, Y, np.where(new_bin)[0])
            if nf > fit[i]:
                pos[i] = new_bin.copy()
                fit[i] = nf
            if nf > pbest_fit[i]:
                pbest[i] = new_bin.copy()
                pbest_fit[i] = nf
            if nf > gbest_fit:
                gbest = new_bin.copy()
                gbest_fit = nf

    return np.where(gbest)[0], float(gbest_fit)


# ============================================================
# GAM-GA: HMRF + GAM operators + λ=0.6 (representative GAM)
# ============================================================
def run_gam_ga(X, Y, k, hmrf_scores):
    """GAM-GA with HMRF scores and λ=0.6."""
    gam = GAM(X, hmrf_scores)
    o = gam.safe_crossover
    gam.safe_crossover = lambda p1, p2, k, g=0, mg=100, **kw: o(
        p1, p2, k, gen=g, max_gen=mg, penalty=0.6)

    orig_eval = bench_gam.evaluate
    try:
        bench_gam.evaluate = lambda X_, Y_, idx: evaluate_fn(X_, Y_, idx)
        sol = ga_gam_safe(X, Y, k, gam, N, T)
    finally:
        bench_gam.evaluate = orig_eval
    return sol


def run_one_task(args):
    ds_file, method, run_idx = args
    ds_name = ds_file.split('(')[0]
    try:
        X_raw, Y_raw = load_csv(f'{DATA_DIR}/{ds_file}')
        if Y_raw.dtype.kind in ('U', 'S', 'O'):
            Y_raw = LabelEncoder().fit_transform(Y_raw)
        Y_raw = Y_raw.astype(int)
        D_orig = X_raw.shape[1]
        k = min(200, int(D_orig * 0.05))
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

        # HMRF scores (for GAM-GA only)
        rf = reliefFScore(X_tr, Y_tr)
        mi_fold = mi_c(X_tr, Y_tr, random_state=42)
        hmrf = (0.5 * (mi_fold - mi_fold.min()) / (mi_fold.max() - mi_fold.min() + 1e-12)
                + 0.5 * (rf - rf.min()) / (rf.max() - rf.min() + 1e-12))

        np.random.seed(seed)
        t0 = time.time()

        if method == 'BGWO':
            sol, best_fit = bgwo(X_tr, Y_tr, k, N, T)
        elif method == 'BPSO':
            sol, best_fit = bpso(X_tr, Y_tr, k, N, T)
        else:  # GAM-GA
            sol = np.asarray(run_gam_ga(X_tr, Y_tr, k, hmrf))
            if sol.dtype == bool:
                sol = np.where(sol)[0]
            else:
                sol = sol.astype(int)

        elapsed = time.time() - t0

        # Test accuracy + redundancy
        knn = KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1)
        test_acc = float(knn.fit(X_tr[:, sol], Y_tr).score(
            X_te[:, sol], Y_te)) if len(sol) > 0 else 0.0
        red = safe_redundancy(X_tr, sol)

        return [{
            'dataset': ds_name, 'method': method, 'run': run_idx,
            'D_orig': D_orig, 'D_reduced': D_reduced,
            'k': k, 'N': N, 'T': T,
            'test_acc': test_acc, 'n_sel': int(len(sol)),
            'red': red if not np.isnan(red) else 0.0,
            'elapsed_s': elapsed,
        }]
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [{'dataset': ds_name, 'method': method,
                 'run': run_idx, 'error': str(e)[:200]}]


if __name__ == '__main__':
    tasks = [(ds, m, r) for ds in DATASETS for m in METHODS for r in range(NR)]
    total = len(tasks)  # 7×3×3 = 63

    completed = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for l in f:
                d = json.loads(l)
                if 'error' not in d:
                    completed.add((d['dataset'], d['method'], d['run']))
        print(f'Resuming: {len(completed)} tasks done')

    tasks_pending = [t for t in tasks if (t[0].split('(')[0], t[1], t[2]) not in completed]
    print(f'{len(DATASETS)}DS × {len(METHODS)}methods × {NR}runs = {len(tasks_pending)} remaining')
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
                status = (f"test_acc={r0['test_acc']:.4f} red={r0.get('red',0):.4f} "
                          f"n={r0['n_sel']} ({r0['elapsed_s']:.0f}s)")
            elapsed = time.time() - t0
            remaining = len(tasks_pending) - i - 1
            eta = elapsed / (i + 1) * remaining if i > 0 and remaining > 0 else 0
            print(f'[{i+1}/{len(tasks_pending)}] {r0.get("dataset","?"):14s} '
                  f'{r0.get("method","?"):8s} {status} '
                  f'({elapsed/60:.0f}m, {remaining} left)', flush=True)

    print(f'\nDone: {time.time()-t0:.0f}s → {OUT}')
