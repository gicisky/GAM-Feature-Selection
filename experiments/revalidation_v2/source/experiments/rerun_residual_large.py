"""Residual: 5 large datasets × 7 bases. Main→full.jsonl, SOTA→sota.jsonl."""
import sys, os, json, time, numpy as np, multiprocessing as mp
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_ROOT, 'code')
UTILS_DIR = os.path.join(REPO_ROOT, 'utils')
DATA_DIR = os.path.join(REPO_ROOT, 'data')
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, UTILS_DIR)
os.chdir(CODE_DIR)
import bench_gam
from bench_gam import GAM, ga_gam_safe, csa_gam_safe, ssa_gam_safe, pso_gam_safe, gwo_gam_safe
from bench_gam_sota import ao_gam_safe
from Filter import load_csv, reliefFScore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif as mi_c
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
RES_DIR = os.path.join(REPO_ROOT, 'results')
OUT_MAIN = f'{RES_DIR}/bench_foldsafe_full.jsonl'
OUT_SOTA = f'{RES_DIR}/bench_foldsafe_sota.jsonl'

def safe_redundancy(X, idx):
    if len(idx) <= 1:
        return 0.0
    Xp = X[:, idx]
    Xc = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-12)
    C = np.abs(np.corrcoef(Xc.T))
    iu = np.triu_indices_from(C, k=1)
    return float(np.mean(C[iu]))
DATASETS = ['SMKCAN187(187,19993).csv', 'BreastCancer2(104,22283).csv', 'CrohnDisease(127,22283).csv', 'GLI(85,22283).csv', 'BreastCancer1(96,24482).csv']
BASES = ['GA', 'CSA', 'SSA', 'PSO', 'DE', 'AO', 'GWO']
BASE_FN = {'GA': ga_gam_safe, 'CSA': csa_gam_safe, 'SSA': ssa_gam_safe, 'PSO': pso_gam_safe, 'AO': ao_gam_safe, 'GWO': gwo_gam_safe}
SOTA_BASES = {'AO', 'GWO'}
MODES = [('GAM-lambda0', 0.0), ('GAM', 0.6)]
N, T, NR = (100, 100, 5)
OUTER_FOLDS, INNER_FOLDS, KNN_K = (5, 5, 5)

def de_gam_safe(X, Y, k, gam, N=100, T=100, CR=0.9):
    pg = gam.p
    kg = min(k, pg)
    pop = gam.safe_init(N, kg)
    fit = np.array([bench_gam.evaluate(X, Y, np.where(pop[i])[0]) for i in range(N)])
    bi = fit.argmax()
    bf, best = (fit[bi], pop[bi].copy())
    for t in range(T):
        for i in range(N):
            cand = [j for j in range(N) if j != i]
            a, b, c = np.random.choice(cand, 3, replace=False)
            d = pop[b] != pop[c]
            donor = pop[a].copy()
            donor[d] = pop[b][d]
            jr = np.random.randint(pg)
            trial = pop[i].copy()
            for j in range(pg):
                if np.random.rand() < CR or j == jr:
                    trial[j] = donor[j]
            ch = gam.safe_crossover(pop[i], trial, kg, gen=t, max_gen=T)
            ch = gam.safe_mutate(ch, kg)
            nf = bench_gam.evaluate(X, Y, np.where(ch)[0])
            if nf >= fit[i]:
                pop[i] = ch
                fit[i] = nf
            if nf > bf:
                bf = nf
                best = ch.copy()
    return np.where(best)[0]
BASE_FN['DE'] = de_gam_safe

def compute_hmrf(X, Y):
    mi = mi_c(X, Y, random_state=42)
    rf = reliefFScore(X, Y)
    return 0.5 * (mi - mi.min()) / (mi.max() - mi.min() + 1e-12) + 0.5 * (rf - rf.min()) / (rf.max() - rf.min() + 1e-12)

def run_one_task(args):
    ds_file, base_name, run_idx = args
    ds_name = ds_file.split('(')[0]
    try:
        X_raw, Y_raw = load_csv(f'{DATA_DIR}/{ds_file}')
        if Y_raw.dtype.kind in ('U', 'S', 'O'):
            Y_raw = LabelEncoder().fit_transform(Y_raw)
        Y_raw = Y_raw.astype(int)
        D = X_raw.shape[1]
        k = max(1, int(D * 0.05))
        run_seed = 42 + run_idx * 7
        base_fn = BASE_FN[base_name]
        skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=run_seed)
        fold_splits = list(skf.split(X_raw, Y_raw))
        sf_full = MinMaxScaler()
        X_full = sf_full.fit_transform(X_raw)
        scores_full = compute_hmrf(X_full, Y_raw)
        proto_a = {m: {'accs': [], 'reds': []} for m, _ in MODES}
        for fi, (tr, te) in enumerate(fold_splits):
            X_tr, Y_tr = (X_full[tr].copy(), Y_raw[tr])
            X_te, Y_te = (X_full[te], Y_raw[te])
            for mode_name, lam in MODES:
                gam = GAM(X_tr, scores_full)
                o = gam.safe_crossover
                gam.safe_crossover = lambda p1, p2, k, gen=0, max_gen=100, **kw: o(p1, p2, k, gen=gen, max_gen=max_gen, penalty=lam)
                orig_e = bench_gam.evaluate
                bench_gam.evaluate = lambda X_, Y_, idx, Xtr=X_tr, Ytr=Y_tr: 0.0 if len(idx) == 0 else np.mean(cross_val_score(KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1), Xtr[:, idx], Ytr, cv=KFold(INNER_FOLDS, shuffle=True, random_state=42)))
                np.random.seed(run_seed)
                sol = base_fn(X_tr, Y_tr, k, gam, N, T)
                bench_gam.evaluate = orig_e
                knn = KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1)
                knn.fit(X_tr[:, sol], Y_tr)
                acc = knn.score(X_te[:, sol], Y_te) if len(sol) > 0 else 0.0
                proto_a[mode_name]['accs'].append(float(acc))
                rv = safe_redundancy(X_tr, sol)
                proto_a[mode_name]['reds'].append(rv if not np.isnan(rv) else 0.0)
        proto_b = {m: {'accs': [], 'reds': []} for m, _ in MODES}
        for fi, (tr, te) in enumerate(fold_splits):
            sf = MinMaxScaler()
            X_tr = sf.fit_transform(X_raw[tr].copy())
            X_te = sf.transform(X_raw[te])
            Y_tr, Y_te = (Y_raw[tr], Y_raw[te])
            scores_fold = compute_hmrf(X_tr, Y_tr)
            for mode_name, lam in MODES:
                gam = GAM(X_tr, scores_fold)
                o = gam.safe_crossover
                gam.safe_crossover = lambda p1, p2, k, gen=0, max_gen=100, **kw: o(p1, p2, k, gen=gen, max_gen=max_gen, penalty=lam)
                orig_e = bench_gam.evaluate
                bench_gam.evaluate = lambda X_, Y_, idx, Xtr=X_tr, Ytr=Y_tr: 0.0 if len(idx) == 0 else np.mean(cross_val_score(KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1), Xtr[:, idx], Ytr, cv=KFold(INNER_FOLDS, shuffle=True, random_state=42)))
                np.random.seed(run_seed)
                sol = base_fn(X_tr, Y_tr, k, gam, N, T)
                bench_gam.evaluate = orig_e
                knn = KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1)
                knn.fit(X_tr[:, sol], Y_tr)
                acc = knn.score(X_te[:, sol], Y_te) if len(sol) > 0 else 0.0
                proto_b[mode_name]['accs'].append(float(acc))
                rv = safe_redundancy(X_tr, sol)
                proto_b[mode_name]['reds'].append(rv if not np.isnan(rv) else 0.0)
        results = []
        for mode_name, lam in MODES:
            for proto, proto_data in [('full', proto_a), ('fold-safe', proto_b)]:
                d = proto_data[mode_name]
                results.append({'dataset': ds_name, 'base': base_name, 'mode': mode_name, 'lambda': lam, 'protocol': proto, 'run': run_idx, 'D': D, 'k': k, 'N': N, 'T': T, 'acc_mean': float(np.mean(d['accs'])), 'acc_std': float(np.std(d['accs'])), 'red_mean': float(np.nanmean(d['reds'])), 'red_std': float(np.nanstd(d['reds'])), 'per_fold_acc': d['accs']})
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [{'dataset': ds_name, 'base': base_name, 'run': run_idx, 'error': str(e)[:200]}]
if __name__ == '__main__':
    tasks = [(ds, b, r) for ds in DATASETS for b in BASES for r in range(NR)]
    completed = set()
    for out_path in [OUT_MAIN, OUT_SOTA]:
        if os.path.exists(out_path):
            with open(out_path) as f:
                for l in f:
                    d = json.loads(l)
                    if 'error' not in d:
                        completed.add((d['dataset'], d['base'], d['run']))
    print(f'Already done: {len(completed)} tasks')
    tasks = [t for t in tasks if (t[0].split('(')[0], t[1], t[2]) not in completed]
    print(f'Residual: {len(DATASETS)}DS×{len(BASES)}bases×{NR}runs = {len(tasks)} remaining tasks')
    t0 = time.time()
    with mp.Pool(8) as pool:
        for i, tr in enumerate(pool.imap_unordered(run_one_task, tasks)):
            base = tr[0].get('base', '?')
            out_path = OUT_SOTA if base in SOTA_BASES else OUT_MAIN
            with open(out_path, 'a') as f:
                for r in tr:
                    f.write(json.dumps(r) + '\n')
            r0 = tr[0]
            st = 'ERR' if 'error' in r0 else 'ok'
            dst = 'SOTA' if base in SOTA_BASES else 'MAIN'
            elapsed = time.time() - t0
            rem = len(tasks) - i - 1
            eta = elapsed / (i + 1) * rem if i > 0 else 0
            print(f"[{i + 1}/{len(tasks)}] {r0.get('dataset', '?'):14s} {r0.get('base', '?'):6s} {st} -> {dst} (elapsed {elapsed / 3600:.1f}h)", flush=True)
    print(f'Done: {time.time() - t0:.0f}s')
