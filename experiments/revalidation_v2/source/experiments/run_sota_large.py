"""
SOTA补完: Brain_Tumor_2 + arcene × AO/GWO, MI top-5000.
GAM λ=0.6 only. Protocol: N=100,T=100,outer-5,inner-5, fold-safe.
Append to bench_foldsafe_sota.jsonl.
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
import bench_gam
from bench_gam import GAM, gwo_gam_safe
from bench_gam_sota import ao_gam_safe
from Filter import load_csv, reliefFScore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif as mi_c
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
RES_DIR = os.path.join(REPO_ROOT, 'results')
OUT = f'{RES_DIR}/bench_foldsafe_sota.jsonl'
PROGRESS = f'{RES_DIR}/.sota_large_progress.log'
DATASETS = ['Brain_Tumor_2(49,10368).csv', 'arcene(200,10000).csv']
BASES = ['AO', 'GWO']
BASE_FN = {'AO': ao_gam_safe, 'GWO': gwo_gam_safe}
MODES = [('GAM', 0.6)]
NR, OUTER_FOLDS, INNER_FOLDS, KNN_K = (5, 5, 5, 5)
N, T = (100, 100)
PRE_N = 5000

def safe_redundancy(X, idx):
    if len(idx) <= 1:
        return 0.0
    Xp = X[:, idx]
    Xc = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-12)
    C = np.nan_to_num(np.abs(np.corrcoef(Xc.T)), nan=0.0, posinf=0.0, neginf=0.0)
    iu = np.triu_indices_from(C, k=1)
    return float(np.mean(C[iu]))

def run_one_task(args):
    ds_file, base_name, run_idx = args
    ds_name = ds_file.split('(')[0]
    try:
        X_raw, Y_raw = load_csv(f'{DATA_DIR}/{ds_file}')
        if Y_raw.dtype.kind in ('U', 'S', 'O'):
            Y_raw = LabelEncoder().fit_transform(Y_raw)
        Y_raw = Y_raw.astype(int)
        D_orig = X_raw.shape[1]
        k = min(200, int(D_orig * 0.05))
        D_reduced = min(PRE_N, D_orig)
        run_seed = 42 + run_idx * 7
        base_fn = BASE_FN[base_name]
        skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=run_seed)
        fold_splits = list(skf.split(X_raw, Y_raw))
        proto = {m[0]: {'accs': [], 'reds': []} for m in MODES}
        t0 = time.time()
        with open(PROGRESS, 'a') as pf:
            pf.write(f"[{time.strftime('%H:%M:%S')}] START {ds_name} {base_name} run={run_idx} D={D_orig} k={k} N={N} T={T} inner_folds={INNER_FOLDS}\n")
        for fi, (tr, te) in enumerate(fold_splits):
            sf = MinMaxScaler()
            X_tr_full = sf.fit_transform(X_raw[tr].copy())
            X_te_full = sf.transform(X_raw[te])
            Y_tr, Y_te = (Y_raw[tr], Y_raw[te])
            mi = mi_c(X_tr_full, Y_tr, random_state=42)
            pre_idx = np.argsort(-mi)[:D_reduced]
            X_tr = X_tr_full[:, pre_idx]
            X_te = X_te_full[:, pre_idx]
            rf = reliefFScore(X_tr, Y_tr)
            mi_fold = mi_c(X_tr, Y_tr, random_state=42)
            scores = 0.5 * (mi_fold - mi_fold.min()) / (mi_fold.max() - mi_fold.min() + 1e-12) + 0.5 * (rf - rf.min()) / (rf.max() - rf.min() + 1e-12)
            for mode_name, lam in MODES:
                gam = GAM(X_tr, scores)
                o_gam = gam.safe_crossover
                gam.safe_crossover = lambda p1, p2, k, gen=0, max_gen=100, **kw: o_gam(p1, p2, k, gen=gen, max_gen=max_gen, penalty=lam)
                orig_e = bench_gam.evaluate
                try:
                    bench_gam.evaluate = lambda X_, Y_, idx, Xtr=X_tr, Ytr=Y_tr: 0.0 if len(idx) == 0 else np.mean(cross_val_score(KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1), Xtr[:, idx], Ytr, cv=KFold(INNER_FOLDS, shuffle=True, random_state=42)))
                    np.random.seed(run_seed)
                    sol = base_fn(X_tr, Y_tr, k, gam, N, T)
                finally:
                    bench_gam.evaluate = orig_e
                knn = KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1)
                acc = float(knn.fit(X_tr[:, sol], Y_tr).score(X_te[:, sol], Y_te)) if len(sol) > 0 else 0.0
                red = safe_redundancy(X_tr, sol)
                proto[mode_name]['accs'].append(acc)
                proto[mode_name]['reds'].append(red if not np.isnan(red) else 0.0)
                elapsed = time.time() - t0
                with open(PROGRESS, 'a') as pf:
                    pf.write(f"[{time.strftime('%H:%M:%S')}] {ds_name} {base_name} run={run_idx} fold={fi + 1}/{OUTER_FOLDS} {mode_name} λ={lam} acc={acc:.4f} red={red:.4f} n={len(sol)} ({elapsed / 60:.1f}m)\n")
        final_results = []
        for mode_name, lam in MODES:
            final_results.append({'dataset': ds_name, 'base': base_name, 'mode': mode_name, 'lambda': lam, 'protocol': 'fold-safe', 'run': run_idx, 'D_orig': D_orig, 'D_reduced': D_reduced, 'k': k, 'N': N, 'T': T, 'acc_mean': float(np.mean(proto[mode_name]['accs'])), 'acc_std': float(np.std(proto[mode_name]['accs'])), 'red_mean': float(np.nanmean(proto[mode_name]['reds'])), 'red_std': float(np.nanstd(proto[mode_name]['reds']))})
        return final_results
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [{'dataset': ds_name, 'base': base_name, 'run': run_idx, 'error': str(e)[:200]}]
if __name__ == '__main__':
    tasks = []
    for ds in DATASETS:
        for base in BASES:
            for r in range(NR):
                ds_name = ds.split('(')[0]
                tasks.append((ds, base, r))
    completed = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for l in f:
                d = json.loads(l)
                if 'error' not in d:
                    completed.add((d['dataset'], d['base'], d['run']))
        print(f'Resuming: {len(completed)} tasks done')
    tasks = [t for t in tasks if (t[0].split('(')[0], t[1], t[2]) not in completed]
    total = len(tasks)
    print(f'{len(DATASETS)}DS × {len(BASES)}bases × {NR}runs × 1mode(GAM) = {total} remaining')
    print(f'MI-top-{PRE_N}, fold-safe only, N={N} T={T} inner-CV={INNER_FOLDS}, Output: {OUT}')
    t0 = time.time()
    with mp.Pool(8) as pool:
        for i, results in enumerate(pool.imap_unordered(run_one_task, tasks)):
            with open(OUT, 'a') as f:
                for r in results:
                    f.write(json.dumps(r) + '\n')
            r0 = results[0]
            status = f"ERROR: {r0['error'][:50]}" if 'error' in r0 else f"acc={r0.get('acc_mean', 0):.3f}"
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
            print(f"[{i + 1}/{total}] {r0.get('dataset', '?'):14s} {r0.get('base', '?'):6s} {status} ({elapsed / 60:.0f}m, ETA {eta / 60:.0f}m)", flush=True)
    print(f'\nDone: {time.time() - t0:.0f}s → {OUT}')
