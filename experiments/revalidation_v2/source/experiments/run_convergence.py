"""
Convergence curve: Bare vs GAM per-generation best fitness (cumulative, non-decreasing).
4 DS × 3 bases (GA,DE,SSA) × 2 modes × 3 runs. N=50 T=50, single split, inner-3fold.
Output: bench_convergence.jsonl
"""
import os, sys, json, time, numpy as np
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import multiprocessing as mp
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_ROOT, 'code')
UTILS_DIR = os.path.join(REPO_ROOT, 'utils')
DATA_DIR = os.path.join(REPO_ROOT, 'data')
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, UTILS_DIR)
os.chdir(CODE_DIR)
import bench_gam
from bench_gam import GAM, ga_gam_safe, ssa_gam_safe
from bench_bare import ga_bare, de_bare, ssa_bare
from Filter import load_csv, reliefFScore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif as mi_c
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
RES_DIR = os.path.join(REPO_ROOT, 'results')
OUT = f'{RES_DIR}/bench_convergence.jsonl'
DATASETS = ['colon(62,2000).csv', 'DLBCL(77,5469).csv', '9_Tumors(60,5726).csv', 'arcene(200,10000).csv']
BASES = ['GA', 'DE', 'SSA']
MODES = ['Bare', 'GAM']
NR, KNN_K, INNER_FOLDS = (3, 5, 3)
N, T = (50, 50)
PRE_N = 5000

def make_inner_cv(y, n_splits=3):
    """StratifiedKFold with fallback to KFold for rare classes. Handles non-zero labels."""
    _, counts = np.unique(y.astype(int), return_counts=True)
    if counts.min() >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)

def de_gam_safe(X, Y, k, gam, N=100, T=100, CR=0.9):
    p = gam.p
    k_eff = min(k, p)
    pop = gam.safe_init(N, k_eff)
    fit = np.array([bench_gam.evaluate(X, Y, np.where(pop[i])[0]) for i in range(N)])
    bi = fit.argmax()
    best_fit, best = (fit[bi], pop[bi].copy())
    for t in range(T):
        for i in range(N):
            cands = [j for j in range(N) if j != i]
            a, b, c = np.random.choice(cands, 3, replace=False)
            diff = pop[b] != pop[c]
            donor = pop[a].copy()
            donor[diff] = pop[b][diff]
            jrand = np.random.randint(p)
            trial = pop[i].copy()
            for j in range(p):
                if np.random.rand() < CR or j == jrand:
                    trial[j] = donor[j]
            child = gam.safe_crossover(pop[i], trial, k_eff, gen=t, max_gen=T)
            child = gam.safe_mutate(child, k_eff)
            nf = bench_gam.evaluate(X, Y, np.where(child)[0])
            if nf >= fit[i]:
                pop[i] = child
                fit[i] = nf
            if nf > best_fit:
                best_fit = nf
                best = child.copy()
    return np.where(best)[0]
BASE_FN_BARE = {'GA': ga_bare, 'DE': de_bare, 'SSA': ssa_bare}
BASE_FN_GAM = {'GA': ga_gam_safe, 'DE': de_gam_safe, 'SSA': ssa_gam_safe}

def run_with_convergence_and_sol(X, Y, k, base_name, mode, seed, gam=None):
    """Run optimizer, record per-generation cumulative-best and mean fitness, return (conv, sol)."""
    eval_count = [0]
    window_fits = []
    global_best = [-np.inf]
    convergence_records = []

    def recording_eval(X_, Y_, idx):
        eval_count[0] += 1
        cv = make_inner_cv(Y_, INNER_FOLDS)
        if len(idx) == 0:
            fit = 0.0
        else:
            fit = float(np.mean(cross_val_score(KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1), X_[:, idx], Y_, cv=cv)))
        global_best[0] = max(global_best[0], fit)
        window_fits.append(fit)
        if eval_count[0] % N == 0:
            convergence_records.append({'gen': len(convergence_records), 'best': float(global_best[0]), 'mean': float(np.mean(window_fits)), 'eval_count': eval_count[0]})
            window_fits.clear()
        return fit
    orig_eval = bench_gam.evaluate
    try:
        bench_gam.evaluate = recording_eval
        np.random.seed(seed)
        if mode == 'Bare':
            sol = BASE_FN_BARE[base_name](X, Y, k, N, T)
        else:
            o = gam.safe_crossover
            gam.safe_crossover = lambda p1, p2, k, gen=0, max_gen=100, **kw: o(p1, p2, k, gen=gen, max_gen=max_gen, penalty=0.6)
            sol = BASE_FN_GAM[base_name](X, Y, k, gam, N, T)
    finally:
        bench_gam.evaluate = orig_eval
    return (convergence_records[:T], sol)

def run_one_task(args):
    ds_file, base_name, mode, run_idx = args
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
        Y_tr, Y_te = (Y_raw[tr], Y_raw[te])
        if D_orig > 10000:
            mi = mi_c(X_tr_full, Y_tr, random_state=42)
            pre_idx = np.argsort(-mi)[:min(PRE_N, D_orig)]
            X_tr = X_tr_full[:, pre_idx]
            X_te = X_te_full[:, pre_idx]
            D_reduced = len(pre_idx)
        else:
            X_tr, X_te = (X_tr_full, X_te_full)
            D_reduced = D_orig
        rf = reliefFScore(X_tr, Y_tr)
        mi_fold = mi_c(X_tr, Y_tr, random_state=42)
        scores = 0.5 * (mi_fold - mi_fold.min()) / (mi_fold.max() - mi_fold.min() + 1e-12) + 0.5 * (rf - rf.min()) / (rf.max() - rf.min() + 1e-12)
        gam = GAM(X_tr, scores) if mode == 'GAM' else None
        t0 = time.time()
        conv, sol_raw = run_with_convergence_and_sol(X_tr, Y_tr, k, base_name, mode, seed, gam)
        sol = np.where(sol_raw)[0] if sol_raw.dtype == bool else sol_raw
        elapsed = time.time() - t0
        knn = KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1)
        test_acc = float(knn.fit(X_tr[:, sol], Y_tr).score(X_te[:, sol], Y_te)) if len(sol) > 0 else 0.0
        return [{'dataset': ds_name, 'base': base_name, 'mode': mode, 'run': run_idx, 'D_orig': D_orig, 'D_reduced': D_reduced, 'k': k, 'N': N, 'T': T, 'test_acc': test_acc, 'n_sel': int(len(sol)), 'convergence': conv, 'conv_len': len(conv), 'final_best': conv[-1]['best'] if conv else None, 'elapsed_s': elapsed}]
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [{'dataset': ds_name, 'base': base_name, 'mode': mode, 'run': run_idx, 'error': str(e)[:200]}]
if __name__ == '__main__':
    tasks = [(ds, b, m, r) for ds in DATASETS for b in BASES for m in MODES for r in range(NR)]
    completed = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for l in f:
                d = json.loads(l)
                if 'error' not in d and 'convergence' in d:
                    completed.add((d['dataset'], d['base'], d['mode'], d['run']))
        print(f'Resuming: {len(completed)} tasks done')
    tasks_pending = [t for t in tasks if (t[0].split('(')[0], t[1], t[2], t[3]) not in completed]
    print(f'{len(DATASETS)}DS × {len(BASES)}bases × {len(MODES)}modes × {NR}runs = {len(tasks_pending)} remaining')
    print(f'N={N} T={T} inner-{INNER_FOLDS}fold, MI-pre only D>10000, single split, Output: {OUT}')
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
                best_str = f"best={r0['final_best']:.4f}" if r0.get('final_best') is not None else ''
                status = f"test_acc={r0['test_acc']:.4f} {best_str} conv_len={r0.get('conv_len', 0)} ({r0.get('elapsed_s', 0) / 60:.1f}m)"
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(tasks_pending) - i - 1) if i > 0 and len(tasks_pending) > i + 1 else 0
            remaining = len(tasks_pending) - i - 1
            print(f"[{i + 1}/{len(tasks_pending)}] {r0.get('dataset', '?'):14s} {r0.get('base', '?'):6s} {r0.get('mode', '?'):6s} {status} ({elapsed / 60:.0f}m, {remaining} left)", flush=True)
    print(f'\nDone: {time.time() - t0:.0f}s → {OUT}')
