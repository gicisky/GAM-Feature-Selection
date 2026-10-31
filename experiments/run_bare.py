#!/usr/bin/env python3
"""
E1: Bare Baseline — 15DS × 5 bases × 5 runs × 5-fold outer CV
Bare = No HMRF, no safe_init, no safe_crossover, no λ penalty.
Random init from all D features, native crossover/mutation operators.
Output: bench_foldsafe_bare.jsonl
"""
import sys, os, json, time, numpy as np
import multiprocessing as mp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_ROOT, 'code')
UTILS_DIR = os.path.join(REPO_ROOT, 'utils')
DATA_DIR = os.path.join(REPO_ROOT, 'data')
sys.path.insert(0, CODE_DIR); sys.path.insert(0, UTILS_DIR); os.chdir(CODE_DIR)

from bench_bare import ga_bare, csa_bare, de_bare, pso_bare, ssa_bare
from Filter import load_csv
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

RES_DIR = os.path.join(REPO_ROOT, 'results')
OUT = f'{RES_DIR}/bench_foldsafe_bare.jsonl'

DATASETS = [
    'colon(62,2000).csv', 'SRBCT(83,2308).csv', 'Leukemia1(72,5327).csv',
    'DLBCL(77,5469).csv', '9_Tumors(60,5726).csv', 'TOX-171(171,5748).csv',
    'Brain_Tumor1(90,5920).csv', 'CNS(59,7130).csv', 'arcene(200,10000).csv',
    'Brain_Tumor_2(50,10368).csv', 'CLLSUB(111,11340).csv', 'MLL(71,12583).csv',
    'Prostate(102,12600).csv', 'Lung(202,12601).csv', 'Eleven_Tumor(174,12534).csv',
]
BASES = ['GA', 'CSA', 'DE', 'PSO', 'SSA']
BASE_FN = {'GA': ga_bare, 'CSA': csa_bare, 'DE': de_bare, 'PSO': pso_bare, 'SSA': ssa_bare}

NR, OUTER_FOLDS, INNER_FOLDS, KNN_K = 5, 5, 5, 5
N, T = 100, 100


def safe_redundancy(X, idx):
    if len(idx) <= 1:
        return 0.0
    Xp = X[:, idx]
    Xc = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-12)
    C = np.abs(np.corrcoef(Xc.T))
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
        D = X_raw.shape[1]; k = max(1, int(D * 0.05))
        run_seed = 42 + run_idx * 7
        base_fn = BASE_FN[base_name]

        skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=run_seed)
        fold_splits = list(skf.split(X_raw, Y_raw))

        import bench_gam
        accs, reds = [], []

        for fi, (tr, te) in enumerate(fold_splits):
            # Fold-safe scaler
            sf = MinMaxScaler()
            X_tr = sf.fit_transform(X_raw[tr].copy())
            X_te = sf.transform(X_raw[te])
            Y_tr, Y_te = Y_raw[tr], Y_raw[te]

            # Inner CV fitness (no HMRF, just KNN CV)
            orig_e = bench_gam.evaluate
            try:
                bench_gam.evaluate = lambda X_, Y_, idx: (
                    0.0 if len(idx) == 0 else np.mean(cross_val_score(
                        KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1),
                        X_tr[:, idx], Y_tr,
                        cv=KFold(INNER_FOLDS, shuffle=True, random_state=42))))

                np.random.seed(run_seed)
                sol = base_fn(X_tr, Y_tr, k, N, T)
            finally:
                bench_gam.evaluate = orig_e

            knn = KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1)
            acc = float(knn.fit(X_tr[:, sol], Y_tr).score(X_te[:, sol], Y_te)) if len(sol) > 0 else 0.0
            accs.append(acc)
            red = safe_redundancy(X_tr, sol)
            reds.append(red if not np.isnan(red) else 0.0)

        return [{
            'dataset': ds_name, 'base': base_name, 'run': run_idx,
            'D': D, 'k': k, 'N': N, 'T': T,
            'mode': 'Bare',
            'protocol': 'fold-safe',
            'acc_mean': float(np.mean(accs)),
            'acc_std': float(np.std(accs)),
            'red_mean': float(np.nanmean(reds)),
            'red_std': float(np.nanstd(reds)),
            'per_fold_acc': accs,
            'per_fold_red': reds,
        }]
    except Exception as e:
        import traceback; traceback.print_exc()
        return [{'dataset': ds_name, 'base': base_name, 'run': run_idx, 'error': str(e)[:200]}]


if __name__ == '__main__':
    tasks = [(ds, b, r) for ds in DATASETS for b in BASES for r in range(NR)]
    # Sort by D smallest first
    tasks.sort(key=lambda x: int(x[0].split('(')[-1].split(',')[1].split(')')[0]))

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
    print(f'{len(DATASETS)}DS × {len(BASES)}bases × {NR}runs × {OUTER_FOLDS}folds')
    print(f'{total} remaining tasks (sorted by D), N={N} T={T}')
    print(f'Output: {OUT}')

    t0 = time.time()
    with mp.Pool(10) as pool:
        for i, results in enumerate(pool.imap_unordered(run_one_task, tasks)):
            with open(OUT, 'a') as f:
                for r in results:
                    f.write(json.dumps(r) + '\n')
            r0 = results[0]
            status = (f'ERROR: {r0["error"][:50]}' if 'error' in r0
                      else f'acc={r0.get("acc_mean", 0):.3f} red={r0.get("red_mean", 0):.3f}')
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
            print(f'[{i+1}/{total}] {r0.get("dataset", "?"):14s} {r0.get("base", "?"):6s} '
                  f'run={r0.get("run", "?")} {status} ({elapsed/3600:.1f}h, ETA {eta/3600:.1f}h)',
                  flush=True)

    print(f'\nDone: {time.time()-t0:.0f}s → {OUT}')
