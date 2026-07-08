#!/usr/bin/env python3
"""补 DLBCL GWO runs 3,4 (D=5469, 无 MI pre-filter, 原协议). Append to bench_foldsafe_sota.jsonl."""
import sys, os, json, time, numpy as np
import multiprocessing as mp

CODE_DIR = '/Users/gicisky/Desktop/论文研读分析/crow/crow_search1/code'
FILTER_DIR = '/Users/gicisky/Desktop/论文研读分析/crow/crow_search1/filter'
DATA_DIR = '/Users/gicisky/Desktop/论文研读分析/crow/crow_search1/dataCSV_high/all'
sys.path.insert(0, CODE_DIR); sys.path.insert(0, FILTER_DIR); os.chdir(CODE_DIR)

import bench_gam
from bench_gam import GAM, gwo_gam_safe
from Filter import load_csv, reliefFScore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif as mi_c
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

RES_DIR = '/Users/gicisky/Desktop/论文研读分析/crow/crow_search1/experiments/gam_20260531/results'
OUT = f'{RES_DIR}/bench_foldsafe_sota.jsonl'

def safe_redundancy(X, idx):
    if len(idx) <= 1: return 0.0
    Xp = X[:, idx]; Xc = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-12)
    C = np.abs(np.corrcoef(Xc.T))
    iu = np.triu_indices_from(C, k=1)
    return float(np.mean(C[iu]))

def compute_hmrf(X, Y):
    mi = mi_c(X, Y, random_state=42)
    rf = reliefFScore(X, Y)
    mn = (mi - mi.min()) / (mi.max() - mi.min() + 1e-12)
    rn = (rf - rf.min()) / (rf.max() - rf.min() + 1e-12)
    return 0.5 * mn + 0.5 * rn

DS_FILE = 'DLBCL(77,5469).csv'
MODES = [('GAM-lambda0', 0.0), ('GAM', 0.6)]
N, T, OUTER_FOLDS, INNER_FOLDS, KNN_K = 100, 100, 5, 5, 5

def run_one_task(run_idx):
    ds_name = 'DLBCL'
    try:
        X_raw, Y_raw = load_csv(f'{DATA_DIR}/{DS_FILE}')
        if Y_raw.dtype.kind in ('U','S','O'): Y_raw = LabelEncoder().fit_transform(Y_raw)
        Y_raw = Y_raw.astype(int)
        D = X_raw.shape[1]; k = max(1, int(D * 0.05))
        run_seed = 42 + run_idx * 7
        skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=run_seed)
        fold_splits = list(skf.split(X_raw, Y_raw))

        # Protocol A: Full
        scaler_full = MinMaxScaler(); X_full = scaler_full.fit_transform(X_raw)
        scores_full = compute_hmrf(X_full, Y_raw)
        proto_a = {m: {'accs': [], 'reds': []} for m, _ in MODES}
        for fi, (tr, te) in enumerate(fold_splits):
            X_tr, Y_tr = X_full[tr].copy(), Y_raw[tr]; X_te, Y_te = X_full[te], Y_raw[te]
            for mode_name, lam in MODES:
                gam = GAM(X_tr, scores_full)
                o = gam.safe_crossover
                gam.safe_crossover = lambda p1,p2,k,g=0,mg=100,**kw: o(p1,p2,k,gen=g,max_gen=mg,penalty=lam)
                orig_e = bench_gam.evaluate
                bench_gam.evaluate = lambda X_,Y_,idx, Xtr=X_tr,Ytr=Y_tr: (
                    0.0 if len(idx)==0 else np.mean(cross_val_score(
                        KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1),
                        Xtr[:,idx], Ytr, cv=KFold(INNER_FOLDS, shuffle=True, random_state=42))))
                np.random.seed(run_seed)
                sol = gwo_gam_safe(X_tr, Y_tr, k, gam, N, T)
                bench_gam.evaluate = orig_e
                knn = KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1)
                knn.fit(X_tr[:, sol], Y_tr)
                acc = knn.score(X_te[:, sol], Y_te) if len(sol) > 0 else 0.0
                proto_a[mode_name]['accs'].append(float(acc))
                red_val = safe_redundancy(X_tr, sol)
                proto_a[mode_name]['reds'].append(red_val if not np.isnan(red_val) else 0.0)

        # Protocol B: Fold-safe
        proto_b = {m: {'accs': [], 'reds': []} for m, _ in MODES}
        for fi, (tr, te) in enumerate(fold_splits):
            sf = MinMaxScaler(); X_tr = sf.fit_transform(X_raw[tr].copy())
            X_te = sf.transform(X_raw[te]); Y_tr, Y_te = Y_raw[tr], Y_raw[te]
            scores_fold = compute_hmrf(X_tr, Y_tr)
            for mode_name, lam in MODES:
                gam = GAM(X_tr, scores_fold)
                o = gam.safe_crossover
                gam.safe_crossover = lambda p1,p2,k,g=0,mg=100,**kw: o(p1,p2,k,gen=g,max_gen=mg,penalty=lam)
                orig_e = bench_gam.evaluate
                bench_gam.evaluate = lambda X_,Y_,idx, Xtr=X_tr,Ytr=Y_tr: (
                    0.0 if len(idx)==0 else np.mean(cross_val_score(
                        KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1),
                        Xtr[:,idx], Ytr, cv=KFold(INNER_FOLDS, shuffle=True, random_state=42))))
                np.random.seed(run_seed)
                sol = gwo_gam_safe(X_tr, Y_tr, k, gam, N, T)
                bench_gam.evaluate = orig_e
                knn = KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1)
                knn.fit(X_tr[:, sol], Y_tr)
                acc = knn.score(X_te[:, sol], Y_te) if len(sol) > 0 else 0.0
                proto_b[mode_name]['accs'].append(float(acc))
                red_val = safe_redundancy(X_tr, sol)
                proto_b[mode_name]['reds'].append(red_val if not np.isnan(red_val) else 0.0)

        results = []
        for mode_name, lam in MODES:
            for proto, proto_data in [('full', proto_a), ('fold-safe', proto_b)]:
                d = proto_data[mode_name]
                results.append({
                    'dataset': ds_name, 'base': 'GWO', 'mode': mode_name,
                    'lambda': lam, 'protocol': proto, 'run': run_idx,
                    'D': D, 'k': k, 'N': N, 'T': T,
                    'acc_mean': float(np.mean(d['accs'])),
                    'acc_std': float(np.std(d['accs'])),
                    'red_mean': float(np.nanmean(d['reds'])),
                    'red_std': float(np.nanstd(d['reds'])),
                    'per_fold_acc': d['accs'],
                })
        return results
    except Exception as e:
        import traceback; traceback.print_exc()
        return [{'dataset': ds_name, 'base': 'GWO', 'run': run_idx, 'error': str(e)[:200]}]

if __name__ == '__main__':
    runs_needed = [3, 4]
    print(f'DLBCL GWO runs {runs_needed} — D=5469, N={N} T={T}, 2 protocols × 2 modes')
    t0 = time.time()
    with mp.Pool(2) as pool:
        for i, task_results in enumerate(pool.imap_unordered(run_one_task, runs_needed)):
            with open(OUT, 'a') as f:
                for r in task_results: f.write(json.dumps(r) + '\n')
            r0 = task_results[0]
            elapsed = time.time() - t0
            print(f'[{i+1}/2] run={r0.get("run")} acc={r0.get("acc_mean",0):.3f} ({elapsed/60:.0f}m)', flush=True)
    print(f'Done: {time.time()-t0:.0f}s → {OUT}')
