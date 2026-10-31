"""
External Baselines — fold-safe comparison with filter, filter-wrapper,
and bare-wrapper methods. Complements the internal GAM ablation (λ0 vs λ0.6).

Methods:
  1. mRMR + KNN       — classic min-redundancy filter (Ding & Peng, 2005)
  2. ReliefF + KNN    — HMRF's ReliefF component alone
  3. mRMR + GA        — two-stage: mRMR pre-filter → GA wrapper
  4. ReliefF + GA     — two-stage: ReliefF pre-filter → GA wrapper
  5. Bare GA          — pure GA without HMRF, GAM, or graph (ablation Group A)
  6. GA + GAM λ=0.6   — reference from main experiment (co-read from bench_foldsafe_full.jsonl)

All protocols: fold-safe (scaler + HMRF on outer_train only), 5 outer × 5 inner CV.

Output: bench_foldsafe_external.jsonl
"""
import sys, os, json, time, numpy as np
import multiprocessing as mp
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_ROOT, 'code')
UTILS_DIR = os.path.join(REPO_ROOT, 'utils')
DATA_DIR = os.path.join(REPO_ROOT, 'data')
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, UTILS_DIR)
os.chdir(CODE_DIR)
from Filter import load_csv, reliefFScore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif as mi_c
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
RES_DIR = os.path.join(REPO_ROOT, 'results')
OUT = f'{RES_DIR}/bench_foldsafe_external.jsonl'

def mrmr_select(X, Y, k):
    """
    Windowed mRMR greedy forward selection.
    Relevance:  mutual information with target (normalized).
    Redundancy: mean absolute Pearson correlation with last 20 selected features
                (windowed for computational tractability on high-dimensional data).
    Note: this is NOT standard full-history mRMR; the window size of 20 is a
    practical approximation. The paper should refer to it as "windowed mRMR".
    """
    p = X.shape[1]
    k = min(k, p)
    if k <= 0:
        return np.array([], dtype=int)
    mi = mi_c(X, Y, random_state=42)
    mi_norm = (mi - mi.min()) / (mi.max() - mi.min() + 1e-12)
    X_std = (X - X.mean(0)) / (X.std(0) + 1e-12)
    selected = []
    remaining = list(range(p))
    first = int(max(remaining, key=lambda i: mi_norm[i]))
    selected.append(first)
    remaining.remove(first)
    while len(selected) < k:
        recent = selected[-20:]
        best_i = max(remaining, key=lambda i: mi_norm[i] - np.mean([np.abs(np.corrcoef(X_std[:, i], X_std[:, s])[0, 1]) for s in recent]))
        selected.append(best_i)
        remaining.remove(best_i)
    return np.array(selected)

def relieff_select(X, Y, k):
    """ReliefF: select top-k by ReliefF score."""
    scores = reliefFScore(X, Y)
    return np.argsort(-scores)[:k]

def _repair_to_pool(child, pool, k_eff):
    """
    Ensure child contains exactly k_eff features, ALL drawn from pool.
    Removes any pool-external features, then fills/trims within pool.
    """
    child = child.copy()
    pool = np.asarray(pool)
    outside = np.setdiff1d(np.where(child)[0], pool)
    child[outside] = False
    selected_in_pool = pool[child[pool]]
    if len(selected_in_pool) > k_eff:
        remove = np.random.choice(selected_in_pool, len(selected_in_pool) - k_eff, replace=False)
        child[remove] = False
    elif len(selected_in_pool) < k_eff:
        available = pool[~child[pool]]
        need = k_eff - len(selected_in_pool)
        if len(available) > 0:
            add = np.random.choice(available, min(need, len(available)), replace=False)
            child[add] = True
    return child

def ga_restricted(X, Y, k, candidate_pool, N=100, T=100):
    """
    Bare GA (tournament + uniform crossover + random mutation)
    with ALL operations constrained to candidate_pool.
    No HMRF, no GAM, no graph-guided refinement.
    Repair after crossover and mutation ensures no pool-external features.
    """
    p_full = X.shape[1]
    pool = np.array(sorted(candidate_pool))
    k_eff = min(k, len(pool))
    pop = np.zeros((N, p_full), dtype=bool)
    for i in range(N):
        sel = np.random.choice(pool, k_eff, replace=False)
        pop[i, sel] = True
    import bench_gam
    evaluate = bench_gam.evaluate
    fit = np.array([evaluate(X, Y, np.where(pop[i])[0]) for i in range(N)])
    best_idx = np.argmax(fit)
    best = pop[best_idx].copy()
    best_fit = fit[best_idx]
    for t in range(T):
        for i in range(N):
            cand = np.random.choice(N, 3, replace=False)
            parent = pop[cand[fit[cand].argmax()]].copy()
            child = np.zeros(p_full, dtype=bool)
            mask = np.random.rand(p_full) < 0.5
            child[mask] = pop[i][mask]
            child[~mask] = parent[~mask]
            child = _repair_to_pool(child, pool, k_eff)
            if np.random.rand() < 0.05:
                o = pool[child[pool]]
                z = pool[~child[pool]]
                if len(o) > 0 and len(z) > 0:
                    child[np.random.choice(o)] = False
                    child[np.random.choice(z)] = True
            nf = evaluate(X, Y, np.where(child)[0])
            if nf > fit[i]:
                pop[i] = child
                fit[i] = nf
            if nf > best_fit:
                best = child.copy()
                best_fit = nf
    return np.where(best)[0]

def safe_redundancy(X, idx):
    if len(idx) <= 1:
        return 0.0
    Xp = X[:, idx]
    Xc = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-12)
    C = np.abs(np.corrcoef(Xc.T))
    iu = np.triu_indices_from(C, k=1)
    return float(np.mean(C[iu]))
DATASETS = ['colon(62,2000).csv', 'SRBCT(83,2308).csv', 'Leukemia1(72,5327).csv', 'DLBCL(77,5469).csv', '9_Tumors(60,5726).csv', 'TOX-171(171,5748).csv', 'Brain_Tumor1(90,5920).csv', 'CNS(59,7130).csv', 'arcene(200,10000).csv', 'Brain_Tumor_2(50,10368).csv', 'CLLSUB(111,11340).csv', 'MLL(71,12583).csv', 'Prostate(102,12600).csv', 'Lung(202,12601).csv', 'Eleven_Tumor(174,12534).csv']
METHODS = ['mRMR+KNN', 'ReliefF+KNN', 'mRMR+GA', 'ReliefF+GA', 'BareGA']
N, T, NR = (100, 100, 5)
OUTER_FOLDS, INNER_FOLDS, KNN_K = (5, 5, 5)

def run_one_task(args):
    ds_file, method, run_idx = args
    ds_name = ds_file.split('(')[0]
    try:
        X_raw, Y_raw = load_csv(f'{DATA_DIR}/{ds_file}')
        if Y_raw.dtype.kind in ('U', 'S', 'O'):
            Y_raw = LabelEncoder().fit_transform(Y_raw)
        Y_raw = Y_raw.astype(int)
        D = X_raw.shape[1]
        k = max(1, int(D * 0.05))
        run_seed = 42 + run_idx * 7
        skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=run_seed)
        fold_splits = list(skf.split(X_raw, Y_raw))
        import bench_gam
        from bench_bare import ga_bare
        results = []
        for fi, (tr, te) in enumerate(fold_splits):
            sf = MinMaxScaler()
            X_tr = sf.fit_transform(X_raw[tr].copy())
            X_te = sf.transform(X_raw[te])
            Y_tr, Y_te = (Y_raw[tr], Y_raw[te])
            cv_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=run_seed + fi)
            orig_e = bench_gam.evaluate
            try:
                bench_gam.evaluate = lambda X_, Y_, idx, Xtr=X_tr, Ytr=Y_tr, cv_in=cv_inner: 0.0 if len(idx) == 0 else np.mean(cross_val_score(KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1), Xtr[:, idx], Ytr, cv=cv_in))
                if method == 'mRMR+KNN':
                    sol = mrmr_select(X_tr, Y_tr, k)
                elif method == 'ReliefF+KNN':
                    sol = relieff_select(X_tr, Y_tr, k)
                elif method == 'mRMR+GA':
                    pool_k = min(D, max(200, 4 * k))
                    pool = mrmr_select(X_tr, Y_tr, pool_k)
                    np.random.seed(run_seed)
                    sol = ga_restricted(X_tr, Y_tr, k, pool, N, T)
                elif method == 'ReliefF+GA':
                    pool_k = min(D, max(200, 4 * k))
                    pool = relieff_select(X_tr, Y_tr, pool_k)
                    np.random.seed(run_seed)
                    sol = ga_restricted(X_tr, Y_tr, k, pool, N, T)
                elif method == 'BareGA':
                    np.random.seed(run_seed)
                    sol = ga_bare(X_tr, Y_tr, k, N, T)
            finally:
                bench_gam.evaluate = orig_e
            knn = KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1)
            if len(sol) > 0:
                knn.fit(X_tr[:, sol], Y_tr)
                acc = knn.score(X_te[:, sol], Y_te)
            else:
                acc = 0.0
            red_val = safe_redundancy(X_tr, sol)
            results.append({'fold': fi, 'acc': float(acc), 'red': red_val if not np.isnan(red_val) else 0.0, 'n_sel': len(sol)})
        accs = [r['acc'] for r in results]
        reds = [r['red'] for r in results]
        return [{'dataset': ds_name, 'method': method, 'run': run_idx, 'D': D, 'k': k, 'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)), 'red_mean': float(np.nanmean(reds)), 'red_std': float(np.nanstd(reds)), 'per_fold_acc': accs, 'per_fold_red': reds}]
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [{'dataset': ds_name, 'method': method, 'run': run_idx, 'error': str(e)[:200]}]
if __name__ == '__main__':
    tasks = [(ds, m, r) for ds in DATASETS for m in METHODS for r in range(NR)]
    total = len(tasks)
    completed = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for l in f:
                d = json.loads(l)
                if 'error' not in d:
                    completed.add((d['dataset'], d['method'], d['run']))
        print(f'Resuming: {len(completed)} tasks already done, skipping...')
    tasks = [t for t in tasks if (t[0].split('(')[0], t[1], t[2]) not in completed]
    print(f'Methods: {METHODS}')
    print(f'{len(DATASETS)}DS × {len(METHODS)}methods × {NR}runs = {len(tasks)} remaining / {total} total')
    print(f'Output: {OUT}')
    t0 = time.time()
    n_workers = min(6, mp.cpu_count())
    print(f'Workers: {n_workers} (reduced to avoid contention with SOTA)')
    with mp.Pool(n_workers) as pool:
        for i, task_results in enumerate(pool.imap_unordered(run_one_task, tasks)):
            with open(OUT, 'a') as f:
                for r in task_results:
                    f.write(json.dumps(r) + '\n')
            r0 = task_results[0]
            status = f"ERROR: {r0['error'][:50]}" if 'error' in r0 else f"acc={r0.get('acc_mean', 0):.3f} red={r0.get('red_mean', 0):.3f}"
            elapsed = time.time() - t0
            remaining = len(tasks) - i - 1
            eta = elapsed / (i + 1) * remaining if i > 0 else 0
            print(f"[{i + 1}/{len(tasks)}] {r0.get('dataset', '?'):14s} {r0.get('method', '?'):14s} run={r0.get('run', '?')} {status} (elapsed {elapsed / 3600:.1f}h, ETA {eta / 3600:.1f}h)", flush=True)
    print(f'\nDone: {time.time() - t0:.0f}s → {OUT}')
