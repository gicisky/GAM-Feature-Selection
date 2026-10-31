"""Ablation: A(Bare) vs B(GAM-λ0) vs C(Full GAM λ=0.6). Fold-safe. Main: 6DS×3bases."""
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
from bench_gam import GAM
from bench_bare import ga_bare, csa_bare, de_bare
from Filter import load_csv, reliefFScore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif as mi_c
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
RES_DIR = os.path.join(REPO_ROOT, 'results')
OUT = f'{RES_DIR}/ablation_main.jsonl'

def safe_redundancy(X, idx):
    if len(idx) <= 1:
        return 0.0
    Xp = X[:, idx]
    Xc = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-12)
    C = np.abs(np.corrcoef(Xc.T))
    iu = np.triu_indices_from(C, k=1)
    return float(np.mean(C[iu]))
DATASETS = ['colon(62,2000).csv', '9_Tumors(60,5726).csv', 'Brain_Tumor1(90,5920).csv', 'Brain_Tumor_2(49,10368).csv', 'DLBCL(77,5469).csv', 'arcene(200,10000).csv', 'SRBCT(83,2308).csv', 'Leukemia1(72,5327).csv', 'TOX-171(171,5748).csv', 'CNS(59,7130).csv', 'CLLSUB(111,11340).csv', 'MLL(71,12583).csv', 'Prostate(102,12600).csv', 'Lung(202,12601).csv', 'Eleven_Tumor(173,12534).csv']
MAIN_DS = {'colon', '9_Tumors', 'Brain_Tumor1', 'Brain_Tumor_2', 'DLBCL', 'arcene'}
BASES = ['GA', 'CSA', 'DE']
GROUPS = [('A-Bare', None), ('B-GAM-l0', 0.0), ('C-Full-GAM', 0.6)]
N, T, NR = (100, 100, 5)
OUTER_FOLDS, INNER_FOLDS, KNN_K = (5, 5, 5)

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
        bare_fn = {'GA': ga_bare, 'CSA': csa_bare, 'DE': de_bare}[base_name]
        skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=run_seed)
        fold_splits = list(skf.split(X_raw, Y_raw))
        acc = {g_name: {'accs': [], 'reds': []} for g_name, _ in GROUPS}
        for fi, (tr, te) in enumerate(fold_splits):
            sf = MinMaxScaler()
            X_tr = sf.fit_transform(X_raw[tr].copy())
            X_te = sf.transform(X_raw[te])
            Y_tr, Y_te = (Y_raw[tr], Y_raw[te])
            scores_fold = compute_hmrf(X_tr, Y_tr)
            knn = KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1)
            orig_e = bench_gam.evaluate
            bench_gam.evaluate = lambda X_, Y_, idx: 0.0 if len(idx) == 0 else np.mean(cross_val_score(KNeighborsClassifier(KNN_K, metric='manhattan', n_jobs=1), X_tr[:, idx], Y_tr, cv=KFold(INNER_FOLDS, shuffle=True, random_state=42)))
            np.random.seed(run_seed)
            sol_a = bare_fn(X_tr, Y_tr, k, N, T)
            knn.fit(X_tr[:, sol_a], Y_tr)
            acc['A-Bare']['accs'].append(float(knn.score(X_te[:, sol_a], Y_te)) if len(sol_a) > 0 else 0.0)
            acc['A-Bare']['reds'].append(safe_redundancy(X_tr, sol_a))
            gam = GAM(X_tr, scores_fold)
            o = gam.safe_crossover
            for g_name, lam in GROUPS:
                if lam is None:
                    continue
                gam.safe_crossover = lambda p1, p2, k, gen=0, max_gen=100, **kw: o(p1, p2, k, gen=gen, max_gen=max_gen, penalty=lam)
                np.random.seed(run_seed)
                fn = {'GA': bench_gam.ga_gam_safe, 'CSA': bench_gam.csa_gam_safe, 'DE': bench_gam.ga_gam_safe}[base_name]
                if base_name == 'DE':

                    def de_fn(X, Y, k, gam, N, T, CR=0.9):
                        pg = gam.p
                        kg = min(k, pg)
                        pop = gam.safe_init(N, kg)
                        fit = np.array([bench_gam.evaluate(X, Y, np.where(pop[i])[0]) for i in range(N)])
                        bi = fit.argmax()
                        bf, best = (fit[bi], pop[bi].copy())
                        for tt in range(T):
                            for ii in range(N):
                                cand = [j for j in range(N) if j != ii]
                                a, b, c = np.random.choice(cand, 3, replace=False)
                                d = pop[b] != pop[c]
                                donor = pop[a].copy()
                                donor[d] = pop[b][d]
                                jr = np.random.randint(pg)
                                trial = pop[ii].copy()
                                for jj in range(pg):
                                    if np.random.rand() < CR or jj == jr:
                                        trial[jj] = donor[jj]
                                ch = gam.safe_crossover(pop[ii], trial, kg, gen=tt, max_gen=T)
                                ch = gam.safe_mutate(ch, kg)
                                nf = bench_gam.evaluate(X, Y, np.where(ch)[0])
                                if nf >= fit[ii]:
                                    pop[ii] = ch
                                    fit[ii] = nf
                                if nf > bf:
                                    bf = nf
                                    best = ch.copy()
                        return np.where(best)[0]
                    fn = de_fn
                sol = fn(X_tr, Y_tr, k, gam, N, T)
                knn.fit(X_tr[:, sol], Y_tr)
                acc[g_name]['accs'].append(float(knn.score(X_te[:, sol], Y_te)) if len(sol) > 0 else 0.0)
                acc[g_name]['reds'].append(safe_redundancy(X_tr, sol))
            bench_gam.evaluate = orig_e
        results = []
        for g_name, _ in GROUPS:
            results.append({'dataset': ds_name, 'base': base_name, 'group': g_name, 'run': run_idx, 'D': D, 'k': k, 'N': N, 'T': T, 'acc_mean': float(np.mean(acc[g_name]['accs'])), 'acc_std': float(np.std(acc[g_name]['accs'])), 'red_mean': float(np.nanmean(acc[g_name]['reds'])), 'red_std': float(np.nanstd(acc[g_name]['reds'])), 'per_fold_acc': acc[g_name]['accs'], 'main_paper': ds_name in MAIN_DS})
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [{'dataset': ds_name, 'base': base_name, 'run': run_idx, 'error': str(e)[:200]}]
if __name__ == '__main__':
    tasks = [(ds, b, r) for ds in DATASETS for b in BASES for r in range(NR)]
    completed = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for l in f:
                d = json.loads(l)
                if 'error' not in d:
                    completed.add((d['dataset'], d['base'], d['run']))
        print(f'Resuming: {len(completed)} tasks done')
    tasks = [t for t in tasks if (t[0].split('(')[0], t[1], t[2]) not in completed]
    main_t = [t for t in tasks if t[0].split('(')[0] in MAIN_DS]
    print(f'Ablation: {len(DATASETS)}DS × {len(BASES)}bases = {len(tasks)} remaining ({len(main_t)} main)')
    t0 = time.time()
    with mp.Pool(8) as pool:
        for i, tr in enumerate(pool.imap_unordered(run_one_task, tasks)):
            with open(OUT, 'a') as f:
                for r in tr:
                    f.write(json.dumps(r) + '\n')
            r0 = tr[0]
            st = 'ERR' if 'error' in r0 else 'ok'
            elapsed = time.time() - t0
            rem = len(tasks) - i - 1
            eta = elapsed / (i + 1) * rem if i > 0 else 0
            print(f"[{i + 1}/{len(tasks)}] {r0.get('dataset', '?'):14s} {r0.get('base', '?')} {r0.get('group', '?'):12s} {st} (elapsed {elapsed / 3600:.1f}h)", flush=True)
    print(f'Done: {time.time() - t0:.0f}s -> {OUT}')
