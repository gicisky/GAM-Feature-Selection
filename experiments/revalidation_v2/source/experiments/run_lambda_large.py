"""
λ Sensitivity — Large-D datasets with MI top-5000 pre-filter.
All λ values compared on identical reduced feature space.
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
from bench_gam import GAM, ga_gam_safe
from bench_gam_sota import ao_gam_safe
from Filter import load_csv, reliefFScore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif as mi_c
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
RES_DIR = os.path.join(REPO_ROOT, 'results')
OUT = f'{RES_DIR}/bench_lambda_sensitivity.jsonl'
DATASETS = ['Brain_Tumor_2(49,10368).csv', 'arcene(200,10000).csv']
BASES = ['GA', 'DE', 'AO']
LAMBDAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
NR, KNN_K, INNER_FOLDS = (3, 5, 5)
N, T = (100, 100)
PRE_N = 5000

def safe_redundancy(X, idx):
    if len(idx) <= 1:
        return 0.0
    Xp = X[:, idx]
    Xc = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-12)
    C = np.abs(np.corrcoef(Xc.T))
    iu = np.triu_indices_from(C, k=1)
    return float(np.mean(C[iu]))

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
BASE_FN = {'GA': ga_gam_safe, 'DE': de_gam_safe, 'AO': ao_gam_safe}

def run_one_task(args):
    ds_file, base_name, lam, run_idx = args
    ds_name = ds_file.split('(')[0]
    try:
        X_raw, Y_raw = load_csv(f'{DATA_DIR}/{ds_file}')
        if Y_raw.dtype.kind in ('U', 'S', 'O'):
            Y_raw = LabelEncoder().fit_transform(Y_raw)
        Y_raw = Y_raw.astype(int)
        D_orig = X_raw.shape[1]
        k_orig = int(D_orig * 0.05)
        k = min(200, k_orig)
        seed = 42 + run_idx * 7
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        tr, te = next(skf.split(X_raw, Y_raw))
        sf = MinMaxScaler()
        X_tr_full = sf.fit_transform(X_raw[tr].copy())
        X_te_full = sf.transform(X_raw[te])
        Y_tr, Y_te = (Y_raw[tr], Y_raw[te])
        mi = mi_c(X_tr_full, Y_tr, random_state=42)
        pre_idx = np.argsort(-mi)[:min(PRE_N, D_orig)]
        X_tr = X_tr_full[:, pre_idx]
        X_te = X_te_full[:, pre_idx]
        rf = reliefFScore(X_tr, Y_tr)
        mi_fold = mi_c(X_tr, Y_tr, random_state=42)
        scores = 0.5 * (mi_fold - mi_fold.min()) / (mi_fold.max() - mi_fold.min() + 1e-12) + 0.5 * (rf - rf.min()) / (rf.max() - rf.min() + 1e-12)
        gam = GAM(X_tr, scores)
        o = gam.safe_crossover
        gam.safe_crossover = lambda p1, p2, k, gen=0, max_gen=100, **kw: o(p1, p2, k, gen=gen, max_gen=max_gen, penalty=lam)
        orig_e = bench_gam.evaluate
        try:
            bench_gam.evaluate = lambda X_, Y_, idx: 0.0 if len(idx) == 0 else np.mean(cross_val_score(KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1), X_tr[:, idx], Y_tr, cv=KFold(INNER_FOLDS, shuffle=True, random_state=42)))
            np.random.seed(seed)
            fn = BASE_FN[base_name]
            sol = fn(X_tr, Y_tr, k, gam, N, T)
        finally:
            bench_gam.evaluate = orig_e
        knn = KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1)
        acc = float(knn.fit(X_tr[:, sol], Y_tr).score(X_te[:, sol], Y_te)) if len(sol) > 0 else 0.0
        red = safe_redundancy(X_tr, sol)
        return [{'dataset': ds_name, 'base': base_name, 'lambda': lam, 'run': run_idx, 'D_orig': D_orig, 'D_reduced': len(pre_idx), 'k': k, 'acc': acc, 'red': red if not np.isnan(red) else 0.0, 'n_sel': len(sol), 'prefilter': f'MI-top-{PRE_N}'}]
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [{'dataset': ds_name, 'base': base_name, 'lambda': lam, 'run': run_idx, 'error': str(e)[:200]}]
if __name__ == '__main__':
    tasks = [(ds, b, lam, r) for ds in DATASETS for b in BASES for lam in LAMBDAS for r in range(NR)]
    total_t = len(tasks)
    completed = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for l in f:
                d = json.loads(l)
                if 'error' not in d:
                    completed.add((d['dataset'], d['base'], d['lambda'], d['run']))
        print(f'Resuming: {len(completed)} tasks done')
    tasks = [t for t in tasks if (t[0].split('(')[0], t[1], t[2], t[3]) not in completed]
    print(f'{len(DATASETS)}DS × {len(BASES)}bases × {len(LAMBDAS)}λ × {NR}runs = {len(tasks)} remaining')
    print(f'MI-top-{PRE_N}, N={N} T={T} k≤200, Output: {OUT}')
    t0 = time.time()
    with mp.Pool(10) as pool:
        for i, results in enumerate(pool.imap_unordered(run_one_task, tasks)):
            with open(OUT, 'a') as f:
                for r in results:
                    f.write(json.dumps(r) + '\n')
            r0 = results[0]
            status = f"ERROR: {r0['error'][:50]}" if 'error' in r0 else f"λ={r0.get('lambda', '?'):.1f} acc={r0.get('acc', 0):.3f} red={r0.get('red', 0):.3f}"
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(tasks) - i - 1) if i > 0 else 0
            print(f"[{i + 1}/{len(tasks)}] {r0.get('dataset', '?'):14s} {r0.get('base', '?'):4s} {status} ({elapsed / 60:.0f}m, ETA {eta / 60:.0f}m)", flush=True)
    print(f'\nDone: {time.time() - t0:.0f}s → {OUT}')
