#!/usr/bin/env python3
"""
Recent searchers: SCSO (2022) + COA (2023) & GAM-SCSO + GAM-COA.
Continuous position → top-k binary. Fold-safe protocol.
10DS × 4 methods × 5 runs × 5-fold. N=50,T=50, λ=0.6.
Output: bench_recent_searchers.jsonl
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
from bench_gam import GAM
from Filter import load_csv, reliefFScore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif as mi_c
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

RES_DIR = '/Users/gicisky/Desktop/论文研读分析/crow/crow_search1/experiments/gam_20260531/results'
OUT = f'{RES_DIR}/bench_recent_searchers.jsonl'

DATASETS = [
    'colon(62,2000).csv', 'SRBCT(83,2308).csv',
    'DLBCL(77,5469).csv', 'Leukemia1(72,5327).csv',
    '9_Tumors(60,5726).csv', 'Brain_Tumor1(90,5920).csv',
    'Brain_Tumor_2(49,10368).csv', 'arcene(200,10000).csv',
    'Lung(202,12601).csv', 'Prostate(102,12600).csv',
]
METHODS = ['SCSO', 'GAM-SCSO', 'COA', 'GAM-COA']
NR, KNN_K, INNER_FOLDS, OUTER_FOLDS = 5, 5, 3, 5
N, T = 50, 50
PRE_N = 5000
METHOD_SEEDS = {'SCSO': 11, 'GAM-SCSO': 22, 'COA': 33, 'GAM-COA': 44}

# ============== Utilities ==============

def safe_redundancy(X, idx):
    if len(idx) <= 1: return 0.0
    Xp = X[:, idx]; Xc = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-12)
    C = np.nan_to_num(np.abs(np.corrcoef(Xc.T)), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.mean(C[np.triu_indices_from(C, k=1)]))

def make_cv(y, n_splits, seed):
    _, counts = np.unique(y.astype(int), return_counts=True)
    if counts.min() >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)

def evaluate_fn(X, Y, idx):
    if len(idx) == 0: return 0.0
    return float(np.mean(cross_val_score(
        KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1),
        X[:, idx], Y, cv=make_cv(Y, INNER_FOLDS, 42))))

def topk_binary(scores, k):
    p = len(scores); binary = np.zeros(p, dtype=bool)
    binary[np.argpartition(scores, -k)[-k:]] = True
    return binary

def compute_hmrf(X, Y):
    mi = mi_c(X, Y, random_state=42)
    rf = reliefFScore(X, Y)
    mn = (mi - mi.min()) / (mi.max() - mi.min() + 1e-12)
    rn = (rf - rf.min()) / (rf.max() - rf.min() + 1e-12)
    return 0.5 * mn + 0.5 * rn

# ============== SCSO (continuous-position binary adaptation) ==============

def scso(X, Y, k, N_pop=50, T_max=50):
    """
    Sand Cat Swarm Optimization (Seyyedabbasi & Kiani, 2022).
    Continuous position → top-k binary.
    """
    p = X.shape[1]; k_eff = min(k, p)
    # Continuous positions, init in [0,1]
    pos = np.random.rand(N_pop, p)
    pop = np.array([topk_binary(pos[i], k_eff) for i in range(N_pop)])
    fit = np.array([evaluate_fn(X, Y, np.where(pop[i])[0]) for i in range(N_pop)])
    bi = np.argmax(fit); best_pos = pos[bi].copy(); best_bin = pop[bi].copy(); best_fit = fit[bi]

    for t in range(T_max):
        r_G = 2.0 - 2.0 * t / T_max
        for i in range(N_pop):
            R = 2.0 * r_G * np.random.rand() - r_G
            theta = np.random.uniform(0, 2*np.pi, p)

            if np.abs(R) <= 1:  # exploitation
                new_pos = best_pos + np.random.rand(p) * np.abs(best_pos - pos[i]) * np.cos(theta)
            else:  # exploration
                idx_r = np.random.randint(N_pop)
                new_pos = pos[idx_r] - np.random.rand(p) * np.abs(pos[idx_r] - pos[i]) * np.cos(theta)

            new_pos = np.clip(new_pos, 0, 1)
            new_bin = topk_binary(new_pos, k_eff)
            nf = evaluate_fn(X, Y, np.where(new_bin)[0])
            if nf > fit[i]:
                pos[i] = new_pos; pop[i] = new_bin; fit[i] = nf
            if nf > best_fit:
                best_pos = new_pos.copy(); best_bin = new_bin.copy(); best_fit = nf

    return np.where(best_bin)[0], float(best_fit)

# ============== COA (continuous-position binary adaptation) ==============

def coa(X, Y, k, N_pop=50, T_max=50):
    """
    Coati Optimization Algorithm (Dehghani et al., KBS 2023).
    Continuous position → top-k binary.
    Phase 1: hunting iguana (exploration, N/2). Phase 2: escaping predators (exploitation, N/2).
    """
    p = X.shape[1]; k_eff = min(k, p)
    pos = np.random.rand(N_pop, p)
    pop = np.array([topk_binary(pos[i], k_eff) for i in range(N_pop)])
    fit = np.array([evaluate_fn(X, Y, np.where(pop[i])[0]) for i in range(N_pop)])
    bi = np.argmax(fit); best_pos = pos[bi].copy(); best_bin = pop[bi].copy(); best_fit = fit[bi]

    for t in range(T_max):
        N_half = N_pop // 2
        # Phase 1: hunting iguana
        iguana_pos = best_pos.copy()
        for i in range(N_half):
            I_factor = np.random.randint(1, 3)
            if np.random.rand() < 0.5:
                new_pos = pos[i] + np.random.rand(p) * (iguana_pos - I_factor * pos[i])
            else:
                new_pos = pos[i] + np.random.uniform(-1, 1, p) * (iguana_pos - pos[i])
            new_pos = np.clip(new_pos, 0, 1)
            new_bin = topk_binary(new_pos, k_eff)
            nf = evaluate_fn(X, Y, np.where(new_bin)[0])
            if nf > fit[i]:
                pos[i] = new_pos; pop[i] = new_bin; fit[i] = nf

        # Phase 2: escaping predators (local perturbation around current)
        for i in range(N_half, N_pop):
            local_radius = 1.0 - t / T_max
            new_pos = pos[i] + local_radius * np.random.uniform(-1, 1, p) * np.abs(best_pos - pos[i])
            new_pos = np.clip(new_pos, 0, 1)
            new_bin = topk_binary(new_pos, k_eff)
            nf = evaluate_fn(X, Y, np.where(new_bin)[0])
            if nf > fit[i]:
                pos[i] = new_pos; pop[i] = new_bin; fit[i] = nf

        # Update global best
        bi = np.argmax(fit); best_pos = pos[bi].copy(); best_bin = pop[bi].copy(); best_fit = fit[bi]

    return np.where(best_bin)[0], float(best_fit)

# ============== GAM-wrapped versions ==============

def gam_scso(X, Y, k, scores, N_pop=50, T_max=50):
    """SCSO with GAM operators: safe_init + graph-guided crossover + λ penalty."""
    p = len(scores); k_eff = min(k, p)
    gam = GAM(X, scores)
    # safe_init binary pop → back to continuous positions
    pop_bin = gam.safe_init(N_pop, k_eff)
    pos = np.zeros((N_pop, p))
    for i in range(N_pop):
        pos[i] = np.where(pop_bin[i], 0.9, 0.1) + np.random.uniform(-0.05, 0.05, p)
        pos[i] = np.clip(pos[i], 0, 1)
    fit = np.array([evaluate_fn(X, Y, np.where(pop_bin[i])[0]) for i in range(N_pop)])
    bi = np.argmax(fit); best_pos = pos[bi].copy(); best_bin = pop_bin[bi].copy(); best_fit = fit[bi]

    o = gam.safe_crossover
    for t in range(T_max):
        r_G = 2.0 - 2.0 * t / T_max
        for i in range(N_pop):
            R = 2.0 * r_G * np.random.rand() - r_G
            theta = np.random.uniform(0, 2*np.pi, p)
            if np.abs(R) <= 1:
                new_pos = best_pos + np.random.rand(p) * np.abs(best_pos - pos[i]) * np.cos(theta)
            else:
                idx_r = np.random.randint(N_pop)
                new_pos = pos[idx_r] - np.random.rand(p) * np.abs(pos[idx_r] - pos[i]) * np.cos(theta)
            new_pos = np.clip(new_pos, 0, 1)
            raw_bin = topk_binary(new_pos, k_eff)
            # GAM crossover + mutation on binary
            child = o(pop_bin[i], raw_bin, k_eff, gen=t, max_gen=T_max, penalty=0.6)
            child = gam.safe_mutate(child, k_eff)
            nf = evaluate_fn(X, Y, np.where(child)[0])
            if nf > fit[i]:
                pop_bin[i] = child; pos[i] = np.where(child, 0.9, 0.1) + np.random.uniform(-0.05, 0.05, p)
                pos[i] = np.clip(pos[i], 0, 1); fit[i] = nf
            if nf > best_fit:
                best_bin = child.copy()
                best_pos = np.where(child, 0.9, 0.1) + np.random.uniform(-0.05, 0.05, p)
                best_pos = np.clip(best_pos, 0, 1)
                best_fit = nf
    return np.where(best_bin)[0]

def gam_coa(X, Y, k, scores, N_pop=50, T_max=50):
    """COA with GAM operators: safe_init + graph-guided crossover + λ penalty."""
    p = len(scores); k_eff = min(k, p)
    gam = GAM(X, scores)
    pop_bin = gam.safe_init(N_pop, k_eff)
    pos = np.zeros((N_pop, p))
    for i in range(N_pop):
        pos[i] = np.where(pop_bin[i], 0.9, 0.1) + np.random.uniform(-0.05, 0.05, p)
        pos[i] = np.clip(pos[i], 0, 1)
    fit = np.array([evaluate_fn(X, Y, np.where(pop_bin[i])[0]) for i in range(N_pop)])
    bi = np.argmax(fit); best_pos = pos[bi].copy(); best_bin = pop_bin[bi].copy(); best_fit = fit[bi]

    o = gam.safe_crossover
    for t in range(T_max):
        N_half = N_pop // 2
        iguana_pos = best_pos.copy()
        for i in range(N_half):
            I_factor = np.random.randint(1, 3)
            if np.random.rand() < 0.5:
                new_pos = pos[i] + np.random.rand(p) * (iguana_pos - I_factor * pos[i])
            else:
                new_pos = pos[i] + np.random.uniform(-1, 1, p) * (iguana_pos - pos[i])
            new_pos = np.clip(new_pos, 0, 1)
            raw_bin = topk_binary(new_pos, k_eff)
            child = o(pop_bin[i], raw_bin, k_eff, gen=t, max_gen=T_max, penalty=0.6)
            child = gam.safe_mutate(child, k_eff)
            nf = evaluate_fn(X, Y, np.where(child)[0])
            if nf > fit[i]:
                pop_bin[i] = child; pos[i] = np.where(child, 0.9, 0.1) + np.random.uniform(-0.05, 0.05, p)
                pos[i] = np.clip(pos[i], 0, 1); fit[i] = nf

        for i in range(N_half, N_pop):
            local_radius = 1.0 - t / T_max
            new_pos = pos[i] + local_radius * np.random.uniform(-1, 1, p) * np.abs(best_pos - pos[i])
            new_pos = np.clip(new_pos, 0, 1)
            raw_bin = topk_binary(new_pos, k_eff)
            child = o(pop_bin[i], raw_bin, k_eff, gen=t, max_gen=T_max, penalty=0.6)
            child = gam.safe_mutate(child, k_eff)
            nf = evaluate_fn(X, Y, np.where(child)[0])
            if nf > fit[i]:
                pop_bin[i] = child; pos[i] = np.where(child, 0.9, 0.1) + np.random.uniform(-0.05, 0.05, p)
                pos[i] = np.clip(pos[i], 0, 1); fit[i] = nf

        bi = np.argmax(fit); best_pos = pos[bi].copy(); best_bin = pop_bin[bi].copy(); best_fit = fit[bi]

    return np.where(best_bin)[0]

# ============== Task runner ==============

def run_one_task(args):
    ds_file, method, run_idx = args
    ds_name = ds_file.split('(')[0]
    try:
        X_raw, Y_raw = load_csv(f'{DATA_DIR}/{ds_file}')
        if Y_raw.dtype.kind in ('U', 'S', 'O'): Y_raw = LabelEncoder().fit_transform(Y_raw)
        Y_raw = Y_raw.astype(int)
        D_orig = X_raw.shape[1]; k = min(200, int(D_orig * 0.05))
        run_seed = 42 + run_idx * 7

        skf = make_cv(Y_raw, OUTER_FOLDS, run_seed)
        fold_splits = list(skf.split(X_raw, Y_raw))

        fold_accs, fold_reds, fold_counts = [], [], []
        t0 = time.time()

        for fi, (tr, te) in enumerate(fold_splits):
            sf = MinMaxScaler()
            X_tr_full = sf.fit_transform(X_raw[tr].copy())
            X_te_full = sf.transform(X_raw[te])
            Y_tr, Y_te = Y_raw[tr], Y_raw[te]

            if D_orig >= 10000:
                mi_pre = mi_c(X_tr_full, Y_tr, random_state=42)
                pre_idx = np.argsort(-mi_pre)[:min(PRE_N, D_orig)]
                X_tr = X_tr_full[:, pre_idx]; X_te = X_te_full[:, pre_idx]
            else:
                X_tr, X_te = X_tr_full, X_te_full

            # HMRF only for GAM methods
            hmrf = compute_hmrf(X_tr, Y_tr) if method.startswith('GAM') else None

            # Per-fold unique seed
            fold_seed = run_seed + fi * 1000 + METHOD_SEEDS[method]
            np.random.seed(fold_seed)

            if method == 'SCSO':
                sol, _ = scso(X_tr, Y_tr, k, N, T)
            elif method == 'COA':
                sol, _ = coa(X_tr, Y_tr, k, N, T)
            elif method == 'GAM-SCSO':
                sol = gam_scso(X_tr, Y_tr, k, hmrf, N, T)
            else:  # GAM-COA
                sol = gam_coa(X_tr, Y_tr, k, hmrf, N, T)

            sol = np.asarray(sol)
            if sol.dtype == bool: sol = np.where(sol)[0]

            knn = KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1)
            acc = float(knn.fit(X_tr[:, sol], Y_tr).score(X_te[:, sol], Y_te)) if len(sol) > 0 else 0.0
            red = safe_redundancy(X_tr, sol)
            fold_accs.append(acc); fold_reds.append(red if not np.isnan(red) else 0.0)
            fold_counts.append(len(sol))

        elapsed = time.time() - t0

        return [{
            'dataset': ds_name, 'method': method, 'run': run_idx,
            'D_orig': D_orig, 'k': k, 'N': N, 'T': T,
            'acc_mean': float(np.mean(fold_accs)), 'acc_std': float(np.std(fold_accs)),
            'red_mean': float(np.nanmean(fold_reds)), 'red_std': float(np.nanstd(fold_reds)),
            'sel_count_mean': float(np.mean(fold_counts)), 'sel_count_std': float(np.std(fold_counts)),
            'elapsed_s': elapsed,
        }]
    except Exception as e:
        import traceback; traceback.print_exc()
        return [{'dataset': ds_name, 'method': method, 'run': run_idx, 'error': str(e)[:200]}]


if __name__ == '__main__':
    tasks = [(ds, m, r) for ds in DATASETS for m in METHODS for r in range(NR)]
    total = len(tasks)  # 10×4×5 = 200

    completed = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for l in f:
                d = json.loads(l)
                if 'error' not in d: completed.add((d['dataset'], d['method'], d['run']))
        print(f'Resuming: {len(completed)} tasks done')

    tasks_pending = [t for t in tasks if (t[0].split('(')[0], t[1], t[2]) not in completed]
    print(f'{len(DATASETS)}DS × {len(METHODS)}methods × {NR}runs = {len(tasks_pending)} remaining')
    print(f'N={N} T={T} {OUTER_FOLDS}-fold, inner-{INNER_FOLDS}fold, MI-pre D>=10000, λ=0.6, Output: {OUT}')

    t0 = time.time()
    with mp.Pool(4, maxtasksperchild=1) as pool:
        for i, results in enumerate(pool.imap_unordered(run_one_task, tasks_pending)):
            with open(OUT, 'a') as f:
                for r in results: f.write(json.dumps(r) + '\n')
            r0 = results[0]
            if 'error' in r0:
                status = f"ERROR: {r0['error'][:50]}"
            else:
                status = (f"acc={r0['acc_mean']:.4f} red={r0.get('red_mean',0):.4f} "
                          f"n={r0.get('sel_count_mean',0):.0f} ({r0['elapsed_s']:.0f}s)")
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(tasks_pending) - i - 1) if i > 0 and len(tasks_pending) > i + 1 else 0
            remaining = len(tasks_pending) - i - 1
            print(f'[{i+1}/{len(tasks_pending)}] {r0.get("dataset","?"):14s} {r0.get("method","?"):12s} '
                  f'{status} ({elapsed/60:.0f}m, {remaining} left)', flush=True)

    print(f'\nDone: {time.time()-t0:.0f}s → {OUT}')
