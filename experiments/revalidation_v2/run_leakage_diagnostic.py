#!/usr/bin/env python3
"""Full-data vs fold-safe HMRF leakage diagnostic (header=None revalidation).

Purpose
-------
Quantify the optimism induced by building the scaler, HMRF supervision
(MI + ReliefF) and the feature-correlation graph on the FULL dataset instead
of the outer training fold. The wrapper search and the held-out evaluation
stay fold-safe in *both* arms, so the only difference between the two arms is
whether preprocessing and guidance use the held-out test fold. This is a joint leakage diagnostic, not an isolated HMRF effect.

This reproduces the old "Fig. 2 leakage diagnostic" of the manuscript, but on
the header=None revalidated data (16 CSVs, first row kept as a real sample).

Protocol (fold-safe arm is intended to match the primary protocol; equivalence requires verification)
--------------------------------------------------------------------
- 15 primary datasets, GA backbone, N=100, T=100, lambda=0.6 (full GAM).
- outer   : StratifiedKFold(5, shuffle, seed = 42 + 7*run)
- inner   : KFold(5, shuffle, seed = 42), deterministic fitness cache
- fold-safe arm : scaler/HMRF/graph built on X[tr] only.
- full-data arm : scaler/HMRF/graph built on X (whole dataset); search on X[tr].
- Both arms share the same outer splits and search seed, so the per-dataset
  accuracy gap is a paired descriptive difference for the joint pipeline change.

Usage
-----
    python run_leakage_diagnostic.py --pilot                       # 1 ds, 1 run
    python run_leakage_diagnostic.py --runs 3 --workers 4          # full 15 ds
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'

import sys, json, time, argparse, hashlib
from pathlib import Path
import multiprocessing as mp

ROOT = (Path(os.environ['GAM_REVALIDATION_ROOT']) if 'GAM_REVALIDATION_ROOT' in os.environ
        else Path(__file__).resolve().parent).resolve()
# Override GAM_REVALIDATION_ROOT when running the script from outside the repository.
sys.path[:0] = [str(ROOT / 'source/code'), str(ROOT / 'source/utils')]

import numpy as np
import pandas as pd
import bench_gam as bg
from Filter import reliefFScore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

DATA = ROOT.parent.parent / 'data'
OUT = ROOT.parent.parent / 'results'
OUT_JSONL = OUT / 'bench_leakage_diagnostic.jsonl'
PROTOCOL = 'joint-scaler-hmrf-graph-v2'
RUN_DIR = None

def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2))
    tmp.replace(path)

def identity(name, run, leakage):
    file = file_for(name)
    source = ROOT / 'source'
    return {'protocol': PROTOCOL, 'dataset': name, 'run': run, 'leakage': bool(leakage),
            'data_sha256': sha(DATA / file), 'runner_sha256': sha(__file__),
            'source_hashes': {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*.py'))},
            'N': 100, 'T': 100, 'lambda': 0.6, 'outer_folds': 5, 'inner_folds': 5}

def task_id(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:24]

OUT_SUMMARY = OUT / 'leakage_diagnostic_summary.json'

DATASETS = ['9_Tumors', 'Brain_Tumor1', 'Brain_Tumor_2', 'CLLSUB', 'CNS',
            'DLBCL', 'Eleven_Tumor', 'Leukemia1', 'Lung', 'MLL', 'Prostate',
            'SRBCT', 'TOX-171', 'arcene', 'colon']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def file_for(name):
    hits = list(DATA.glob(f'{name}(*.csv'))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0].name


def getdata(file):
    d = pd.read_csv(DATA / file, header=None)          # first row is a sample
    X = d.iloc[:, :-1].to_numpy(dtype=float)
    Y = LabelEncoder().fit_transform(d.iloc[:, -1])
    return X, Y, sha(DATA / file)


def hmrf(X, Y, alpha=0.5):
    mi = mutual_info_classif(X, Y, random_state=42)
    rf = reliefFScore(X, Y)
    mn = (mi - mi.min()) / (mi.max() - mi.min() + 1e-12)
    rn = (rf - rf.min()) / (rf.max() - rf.min() + 1e-12)
    return alpha * mn + (1 - alpha) * rn


class Evaluator:
    """Inner-CV KNN fitness with a deterministic cache (mirrors run_revalidation)."""

    def __init__(self, X, Y, folds=5, seed=42):
        self.X, self.Y = X, Y
        self.cv = list(KFold(folds, shuffle=True, random_state=seed).split(X, Y))
        self.cache = {}

    def __call__(self, X, Y, idx):
        idx = np.sort(np.asarray(idx, dtype=int))
        key = idx.tobytes()
        if key not in self.cache:
            self.cache[key] = float(np.mean(cross_val_score(
                KNeighborsClassifier(5, metric='manhattan', n_jobs=1),
                self.X[:, idx], self.Y, cv=self.cv, n_jobs=1))) if len(idx) else 0.0
        return self.cache[key]


def run_arm(X, Y, tr, te, k, seed, leakage):
    """One outer fold of one arm.

    leakage=False -> scaler/HMRF/graph on X[tr] (fold-safe).
    leakage=True  -> scaler/HMRF/graph on X     (full-data, leaks the test fold).
    The search still runs on X[tr] and is scored on X[te] in both cases.
    """
    yt, ye = Y[tr], Y[te]
    if leakage:
        sc = MinMaxScaler(); Xsc = sc.fit_transform(X)   # full-data scaler
        xt, xe = Xsc[tr], Xsc[te]
        scores = hmrf(Xsc, Y)                            # full-data HMRF
        gam = bg.GAM(Xsc, scores)                        # full-data graph
    else:
        sc = MinMaxScaler(); xt = sc.fit_transform(X[tr]); xe = sc.transform(X[te])
        scores = hmrf(xt, yt)                            # fold-safe HMRF
        gam = bg.GAM(xt, scores)                         # fold-safe graph

    ev = Evaluator(xt, yt, 5, 42)
    original_cross = gam.safe_crossover
    def fixed_cross(p1, p2, kk, gen=0, max_gen=100, **kw):
        return original_cross(p1, p2, kk, gen=gen, max_gen=max_gen, penalty=0.6)
    gam.safe_crossover = fixed_cross
    old = bg.evaluate
    bg.evaluate = ev
    np.random.seed(seed)
    try:
        idx = bg.ga_gam_safe(xt, yt, k, gam, N=100, T=100)
    finally:
        bg.evaluate = old

    idx = np.sort(np.asarray(idx, dtype=int))
    knn = KNeighborsClassifier(5, metric='manhattan', n_jobs=1)
    pred = knn.fit(xt[:, idx], yt).predict(xe[:, idx])
    assert len(idx) == len(np.unique(idx)) == k
    return {'accuracy': float(accuracy_score(ye, pred)),
            'train_indices': tr.tolist(), 'test_indices': te.tolist(),
            'selected_features': idx.tolist(), 'predictions': pred.tolist(),
            'test_labels': ye.tolist(), 'returned_fitness': ev(xt, yt, idx)}


def run_task(args):
    name, run, leakage, output = args
    folder = Path(output)
    spec = identity(name, run, leakage)
    ident = task_id(spec)
    target = folder / (ident + ".json")
    if target.exists():
        record = json.loads(target.read_text())
        assert record["identity"] == spec
        return record
    file = file_for(name)
    X, Y, datahash = getdata(file)
    D = X.shape[1]
    k = max(1, int(D * 0.05))
    seed = 42 + 7 * run
    outer = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, Y))
    fold_acc, t0 = [], time.perf_counter()
    for fi, (tr, te) in enumerate(outer):
        fp = folder / f'{ident}_fold{fi}.json'
        if fp.exists():
            fold = json.loads(fp.read_text())
            assert fold['identity'] == spec
            assert fold['train_indices'] == tr.tolist() and fold['test_indices'] == te.tolist()
        else:
            fold = run_arm(X, Y, tr, te, k, seed, leakage)
            fold.update({'identity': spec, 'fold': fi, 'n_samples': len(X), 'D': D, 'k': k})
            atomic(fp, fold)
        fold_acc.append(fold['accuracy'])
    rec = {
        'dataset': name, 'file': file, 'run': run,
        'leakage': bool(leakage), 'k': k, 'D': D, 'n_samples': len(X),
        'data_sha256': datahash, 'seed': seed,
        'identity': spec, 'task_id': ident, 'fold_acc': fold_acc,
        'acc_mean': float(np.mean(fold_acc)),
        'elapsed_s': round(time.perf_counter() - t0, 1),
    }
    atomic(target, rec)
    return rec


def _summarize(records):
    paired = {}
    for r in records:
        key = (r['dataset'], r['run'])
        arm = 'full' if r['leakage'] else 'safe'
        if arm in paired.setdefault(key, {}):
            raise ValueError(f'Duplicate task: {key}, {arm}')
        paired[key][arm] = r['acc_mean']
    rows = []
    for name in sorted({k[0] for k in paired}):
        pairs = [v for (ds, run), v in paired.items() if ds == name]
        if any(set(v) != {'full', 'safe'} for v in pairs):
            raise ValueError(f'Incomplete paired results: {name}')
        diffs = [v['full'] - v['safe'] for v in pairs]
        rows.append({'dataset': name, 'n_runs': len(diffs),
                     'safe_accuracy': float(np.mean([v['safe'] for v in pairs])),
                     'full_accuracy': float(np.mean([v['full'] for v in pairs])),
                     'accuracy_difference': float(np.mean(diffs)),
                     'difference_min': float(min(diffs)), 'difference_max': float(max(diffs))})
    if not rows:
        raise ValueError('No complete paired datasets')
    # Exact conditional signed-rank sign enumeration, including tied absolute ranks.
    from scipy.stats import rankdata
    delta = np.array([r['accuracy_difference'] for r in rows])
    nonzero = delta[delta != 0]
    test = None
    if len(rows) >= 2 and len(nonzero):
        ranks = np.rint(2 * rankdata(np.abs(nonzero))).astype(int)
        observed = int(ranks[nonzero > 0].sum()); total = int(ranks.sum())
        counts = {0: 1}
        for rank in ranks:
            updated = dict(counts)
            for score, count in counts.items():
                updated[score + int(rank)] = updated.get(score + int(rank), 0) + count
            counts = updated
        tail = min(observed, total - observed)
        p = min(1., 2 * sum(v for k, v in counts.items() if k <= tail) / 2**len(nonzero))
        test = {'method': 'exact conditional signed-rank sign enumeration; zero differences excluded',
                'statistic': tail / 2., 'pvalue': p, 'nonzero_datasets': len(nonzero)}
    return {'protocol': PROTOCOL, 'n_datasets': len(rows), 'per_dataset': rows,
            'macro_accuracy_difference': float(delta.mean()), 'signed_rank': test,
            'interpretation': 'Full-data minus outer-training-only preprocessing and guidance. Negative differences are allowed. This does not isolate HMRF leakage.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, default=OUT / 'leakage_diagnostic_v2')
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--no-mp', action='store_true')
    args = ap.parse_args()

    ds = args.datasets or DATASETS
    if args.pilot:
        ds, args.runs = ds[:1], 1

    if args.runs < 1 or args.workers < 1:
        ap.error('runs and workers must be positive')
    if len(ds) != len(set(ds)):
        ap.error('datasets must be unique')
    folder = args.output_dir.resolve(); folder.mkdir(parents=True, exist_ok=True)
    tasks = [(d, r, leak, str(folder)) for d in ds
             for r in range(args.runs) for leak in (False, True)]
    plan = [identity(d, r, leak) for d, r, leak, _ in tasks]
    plan_id = task_id({'tasks': plan})
    atomic(folder / f'plan_{plan_id}.json', plan)
    print(f'Tasks={len(tasks)}; validated existing tasks and folds will be reused', flush=True)
    records = []
    if args.no_mp or len(tasks) == 1:
        for t in tasks:
            records.append(run_task(t)); print(f'Completed {len(records)}/{len(tasks)}', flush=True)
    else:
        with mp.get_context('spawn').Pool(args.workers) as pool:
            for record in pool.imap_unordered(run_task, tasks, chunksize=1):
                records.append(record); print(f'Completed {len(records)}/{len(tasks)}', flush=True)
    assert len(records) == len(tasks)
    atomic(folder / f'summary_{plan_id}.json', _summarize(records))
    print(json.dumps(_summarize(records), indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
