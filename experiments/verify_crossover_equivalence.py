#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GAM 原版/优化版等价性回归测试（无需数据集、Filter或sklearn）。

用法：
    python verify_crossover_equivalence.py
    python verify_crossover_equivalence.py --gam /path/to/bench_gam.py
    python verify_crossover_equivalence.py --trials 3

从指定 bench_gam.py 的 AST 中仅加载 GAM 类，避免原项目依赖；
原版参考交叉逻辑摘自用户上传的 code.zip/code/bench_gam.py。
验证输出、随机状态、分支覆盖、边界条件和简化GA搜索轨迹。
退出码：PASS=0，FAIL=1。
"""
import argparse
import ast
from pathlib import Path
import sys
import time
import numpy as np


def load_gam(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    cls = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GAM'), None)
    if cls is None:
        raise ValueError(f'找不到 GAM 类: {path}')
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    ns = {'np': np}
    exec(compile(module, str(path), 'exec'), ns)
    return ns['GAM']


def old_safe_crossover(gam, p1, p2, k, gen=0, max_gen=100, penalty=0.6):
    # 原版逐候选实现，保留原始随机调用顺序与 cnt 逻辑
    progress = gen / max_gen
    ratio_original = 0.8 - 0.5 * progress
    if np.random.rand() < ratio_original:
        child = np.zeros(gam.p, dtype=bool)
        mask = np.random.rand(gam.p) < 0.5
        child[mask] = p1[mask]
        child[~mask] = p2[~mask]
        cnt = np.sum(child)
        if cnt > k:
            pos = np.where(child)[0]
            np.random.shuffle(pos)
            child[pos[k:]] = False
        while cnt < k:
            pos = np.where(~child)[0]
            child[np.random.choice(pos)] = True
            cnt += 1
        return child
    u = np.unique(np.concatenate([np.where(p1)[0], np.where(p2)[0]]))
    if len(u) <= k:
        child = np.zeros(gam.p, dtype=bool)
        child[u] = True
        while np.sum(child) < k:
            child[np.random.choice(np.where(~child)[0])] = True
        return child
    rel = gam.scores[u]
    selected, candidates = [], list(range(len(u)))
    while len(selected) < k and candidates:
        if not selected:
            best = candidates[np.argmax(rel[candidates])]
        else:
            best = max(candidates, key=lambda c: (
                rel[c] - penalty * np.mean([gam.C[u[c], u[s]] for s in selected[-10:]])
            ))
        selected.append(best)
        candidates.remove(best)
    child = np.zeros(gam.p, dtype=bool)
    child[u[selected]] = True
    return child


def rng_equal(a, b):
    return (a[0] == b[0] and np.array_equal(a[1], b[1])
            and a[2] == b[2] and a[3] == b[3] and a[4] == b[4])


def run_with_seed(func, seed):
    np.random.seed(seed)
    result = func()
    return result, np.random.get_state()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gam', type=Path, default=Path(__file__).with_name('bench_gam.py'))
    parser.add_argument('--trials', type=int, default=2, help='每个 k/penalty 组合的重复数，默认2')
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials 必须 >= 1')
    if not args.gam.is_file():
        parser.error(f'文件不存在：{args.gam}')

    GAM = load_gam(args.gam)
    rng = np.random.default_rng(20261025)
    p, n = 1500, 48
    X = rng.normal(size=(n, p))
    scores = rng.random(p)
    gam = GAM(X, scores)
    k_values = [100, 200, 500, 630]
    penalties = [0.0, 0.6, 0.8]
    mismatches = []
    branches = {'ordinary': 0, 'graph': 0, 'fill': 0}
    cases = 0
    old_seconds = new_seconds = 0.0

    def check(p1, p2, k, gen, penalty, seed, label):
        nonlocal cases, old_seconds, new_seconds
        progress = gen / 50
        np.random.seed(seed)
        ordinary = np.random.rand() < (0.8 - 0.5 * progress)
        u_size = int(np.count_nonzero(p1 | p2))
        branch = 'ordinary' if ordinary else ('fill' if u_size <= k else 'graph')
        branches[branch] += 1

        t = time.perf_counter()
        old, old_state = run_with_seed(lambda: old_safe_crossover(
            gam, p1, p2, k, gen=gen, max_gen=50, penalty=penalty), seed)
        old_seconds += time.perf_counter() - t
        t = time.perf_counter()
        new, new_state = run_with_seed(lambda: gam.safe_crossover(
            p1, p2, k, gen=gen, max_gen=50, penalty=penalty), seed)
        new_seconds += time.perf_counter() - t
        cases += 1
        if (not np.array_equal(old, new) or not rng_equal(old_state, new_state)
                or old.sum() != k or new.sum() != k):
            mismatches.append((label, k, gen, penalty, seed, branch,
                               bool(np.array_equal(old, new)), rng_equal(old_state, new_state)))

    # 逐组合强制覆盖普通/图引导两种分支；图引导用较晚迭代，减小运行量
    for k in k_values:
        for penalty in penalties:
            for trial in range(args.trials):
                p1 = np.zeros(p, dtype=bool)
                p2 = np.zeros(p, dtype=bool)
                p1[rng.choice(p, k, replace=False)] = True
                p2[rng.choice(p, k, replace=False)] = True
                for gen, desired in [(0, 'ordinary'), (49, 'graph')]:
                    seed = 100000 + cases
                    # 找到真正触发所需分支的种子，不以 |u|>k 代替分支判断
                    for _ in range(1000):
                        np.random.seed(seed)
                        is_ordinary = np.random.rand() < (0.8 - 0.5 * gen / 50)
                        if is_ordinary == (desired == 'ordinary'):
                            break
                        seed += 1
                    check(p1, p2, k, gen, penalty, seed, desired)

    # 覆盖 |u| <= k 的补齐路径，使用两组相同父代
    for k in [100, 630]:
        p1 = np.zeros(p, dtype=bool)
        p1[rng.choice(p, k, replace=False)] = True
        seed = 900000
        while True:
            np.random.seed(seed)
            if np.random.rand() >= 0.8 - 0.5 * 49 / 50:
                break
            seed += 1
        check(p1, p1.copy(), k, 49, 0.6, seed, 'fill')

    # 简化搜索轨迹测试：相同初始种群、相同适应度、相同随机数；
    # 检查多代交叉及其后的随机数序列，不能代替实际GA+CV完整复现。
    k, N, T = 30, 8, 4
    init_pop = np.zeros((N, p), dtype=bool)
    for i in range(N):
        init_pop[i, rng.choice(p, k, replace=False)] = True
    weights = rng.normal(size=p)
    def toy_search(crossover):
        pop = init_pop.copy()
        trace = []
        np.random.seed(777)
        for gen in range(T):
            fitness = pop.astype(float) @ weights
            new_pop = []
            for _ in range(N):
                a, b = np.random.choice(N, 2, replace=False)
                child = crossover(gam, pop[a], pop[b], k, gen=gen, max_gen=T, penalty=0.6)
                new_pop.append(child)
            pop = np.asarray(new_pop)
            trace.append((pop.copy(), float(np.max(fitness))))
        return trace, np.random.get_state()
    old_trace, old_state = toy_search(old_safe_crossover)
    new_trace, new_state = toy_search(lambda _gam, *a, **kw: gam.safe_crossover(*a, **kw))
    if not (all(np.array_equal(a[0], b[0]) and a[1] == b[1]
                for a, b in zip(old_trace, new_trace)) and rng_equal(old_state, new_state)):
        mismatches.append(('toy_search',))

    print(f'GAM: {args.gam.resolve()}')
    print(f'交叉测试: {cases} 组；分支覆盖: {branches}')
    print(f'原版累计耗时: {old_seconds:.3f}s；新版累计耗时: {new_seconds:.3f}s')
    if new_seconds:
        print(f'本测试累计交叉加速: {old_seconds / new_seconds:.2f}x（非E7/E9整体加速）')
    print('简化搜索轨迹：', 'PASS' if not any(m[0] == 'toy_search' for m in mismatches) else 'FAIL')
    for m in mismatches[:10]:
        print('MISMATCH:', m)
    print('OVERALL:', 'PASS' if not mismatches else f'FAIL ({len(mismatches)} mismatches)')
    return 0 if not mismatches else 1


if __name__ == '__main__':
    sys.exit(main())
