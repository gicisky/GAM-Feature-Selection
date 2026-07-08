#!/usr/bin/env python3
"""
Bare Backbone Implementations — No HMRF, No GAM, No Graph.
Pure algorithm baselines for ablation Group A:
  GA: tournament selection + uniform crossover + random mutation
  CSA: crow search with awareness probability + flight length
  DE: DE/rand/1 mutation + binomial crossover

All share: random init (k features from all D), k-constraint, same fitness.
"""
import numpy as np


def ga_bare(X, Y, k, N=100, T=100):
    p = X.shape[1]; k_eff = min(k, p)
    pop = np.zeros((N, p), dtype=bool)
    for i in range(N):
        sel = np.random.choice(p, k_eff, replace=False)
        pop[i, sel] = True

    from bench_gam import evaluate
    fit = np.array([evaluate(X, Y, np.where(pop[i])[0]) for i in range(N)])
    best_idx = np.argmax(fit); best = pop[best_idx].copy(); best_fit = fit[best_idx]

    for t in range(T):
        for i in range(N):
            cand = np.random.choice(N, 3, replace=False)
            parent = pop[cand[fit[cand].argmax()]].copy()
            child = np.zeros(p, dtype=bool)
            mask = np.random.rand(p) < 0.5
            child[mask] = pop[i][mask]
            child[~mask] = parent[~mask]
            cnt = np.sum(child)
            if cnt > k_eff:
                pos = np.where(child)[0]; np.random.shuffle(pos)
                child[pos[k_eff:]] = False
            elif cnt < k_eff:
                z = np.where(~child)[0]
                child[np.random.choice(z, k_eff - cnt, replace=False)] = True
            if np.random.rand() < 0.05:
                o = np.where(child)[0]; z = np.where(~child)[0]
                if len(o) > 0 and len(z) > 0:
                    child[np.random.choice(o)] = False
                    child[np.random.choice(z)] = True
            nf = evaluate(X, Y, np.where(child)[0])
            if nf > fit[i]: pop[i] = child; fit[i] = nf
            if nf > best_fit: best = child.copy(); best_fit = nf

    return np.where(best)[0]


def csa_bare(X, Y, k, N=100, T=100, AP=0.1, fl=2.0):
    p = X.shape[1]; k_eff = min(k, p)
    pop = np.zeros((N, p), dtype=bool)
    memory = np.zeros((N, p), dtype=bool)
    for i in range(N):
        sel = np.random.choice(p, k_eff, replace=False)
        pop[i, sel] = True; memory[i, sel] = True

    from bench_gam import evaluate
    fit = np.array([evaluate(X, Y, np.where(pop[i])[0]) for i in range(N)])
    mem_fit = fit.copy()
    best_idx = np.argmax(fit); best = pop[best_idx].copy(); best_fit = fit[best_idx]

    for t in range(T):
        for i in range(N):
            j = np.random.randint(N)
            while j == i: j = np.random.randint(N)
            cont = pop[i].astype(float)
            if np.random.rand() < AP:
                new_cont = cont + np.random.randn(p) * 0.5
            else:
                mem_j = memory[j].astype(float)
                new_cont = cont + fl * np.random.rand(p) * (mem_j - cont)
            probs = 1.0 / (1.0 + np.exp(-np.clip(new_cont, -50, 50)))
            child = probs > np.random.rand(p)
            cnt = np.sum(child)
            if cnt > k_eff:
                pos = np.where(child)[0]; np.random.shuffle(pos)
                child[pos[k_eff:]] = False
            elif cnt < k_eff:
                z = np.where(~child)[0]
                child[np.random.choice(z, k_eff - cnt, replace=False)] = True
            nf = evaluate(X, Y, np.where(child)[0])
            if nf > fit[i]: pop[i] = child; fit[i] = nf
            if nf > mem_fit[i]: memory[i] = child.copy(); mem_fit[i] = nf
            if nf > best_fit: best = child.copy(); best_fit = nf

    return np.where(best)[0]


def de_bare(X, Y, k, N=100, T=100, CR=0.9):
    p = X.shape[1]; k_eff = min(k, p)
    pop = np.zeros((N, p), dtype=bool)
    for i in range(N):
        sel = np.random.choice(p, k_eff, replace=False)
        pop[i, sel] = True

    from bench_gam import evaluate
    fit = np.array([evaluate(X, Y, np.where(pop[i])[0]) for i in range(N)])
    best_idx = np.argmax(fit); best = pop[best_idx].copy(); best_fit = fit[best_idx]

    for t in range(T):
        for i in range(N):
            candidates = [j for j in range(N) if j != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            diff = pop[b] != pop[c]; donor = pop[a].copy(); donor[diff] = pop[b][diff]
            jrand = np.random.randint(p); trial = pop[i].copy()
            for j in range(p):
                if np.random.rand() < CR or j == jrand: trial[j] = donor[j]
            cnt = np.sum(trial)
            if cnt > k_eff:
                pos = np.where(trial)[0]; np.random.shuffle(pos)
                trial[pos[k_eff:]] = False
            elif cnt < k_eff:
                z = np.where(~trial)[0]
                trial[np.random.choice(z, k_eff - cnt, replace=False)] = True
            nf = evaluate(X, Y, np.where(trial)[0])
            if nf >= fit[i]: pop[i] = trial; fit[i] = nf
            if nf > best_fit: best = trial.copy(); best_fit = nf

    return np.where(best)[0]


# ===========================================================================
# PSO Bare — Binary Particle Swarm (Kennedy & Eberhart, 1997)
# ===========================================================================
def pso_bare(X, Y, k, N=100, T=100):
    p = X.shape[1]; k_eff = min(k, p)
    pos = np.zeros((N, p), dtype=bool)
    for i in range(N):
        sel = np.random.choice(p, k_eff, replace=False)
        pos[i, sel] = True

    from bench_gam import evaluate
    fit = np.array([evaluate(X, Y, np.where(pos[i])[0]) for i in range(N)])
    pbest = pos.copy(); pbest_fit = fit.copy()
    gbest_idx = np.argmax(fit); gbest = pos[gbest_idx].copy(); gbest_fit = fit[gbest_idx]
    vel = np.random.uniform(-1, 1, (N, p))

    for t in range(T):
        w = 0.9 - 0.5 * t / T
        for i in range(N):
            r1, r2 = np.random.rand(p), np.random.rand(p)
            vel[i] = (w * vel[i]
                      + 2.0 * r1 * (pbest[i].astype(float) - pos[i].astype(float))
                      + 2.0 * r2 * (gbest.astype(float) - pos[i].astype(float)))
            vel[i] = np.clip(vel[i], -6, 6)
            prob = 1.0 / (1.0 + np.exp(-vel[i]))
            child = prob > np.random.rand(p)
            cnt = np.sum(child)
            if cnt > k_eff:
                on_idx = np.where(child)[0]; np.random.shuffle(on_idx)
                child[on_idx[k_eff:]] = False
            elif cnt < k_eff:
                off_idx = np.where(~child)[0]
                child[np.random.choice(off_idx, k_eff - cnt, replace=False)] = True
            nf = evaluate(X, Y, np.where(child)[0])
            if nf > fit[i]: pos[i] = child; fit[i] = nf
            if nf > pbest_fit[i]: pbest[i] = child.copy(); pbest_fit[i] = nf
            if nf > gbest_fit: gbest = child.copy(); gbest_fit = nf
    return np.where(gbest)[0]


# ===========================================================================
# SSA Bare — Binary Salp Swarm (Mirjalili et al., 2017)
# ===========================================================================
def ssa_bare(X, Y, k, N=100, T=100):
    p = X.shape[1]; k_eff = min(k, p)
    pop = np.zeros((N, p), dtype=bool)
    for i in range(N):
        sel = np.random.choice(p, k_eff, replace=False)
        pop[i, sel] = True

    from bench_gam import evaluate
    fit = np.array([evaluate(X, Y, np.where(pop[i])[0]) for i in range(N)])
    best_idx = np.argmax(fit); best = pop[best_idx].copy(); best_fit = fit[best_idx]

    for t in range(T):
        c1 = 2.0 * np.exp(-(4.0 * t / T) ** 2)
        for i in range(N):
            if i == 0:
                child = pop[0].copy()
                mask = np.random.rand(p) < c1
                child[mask] = best[mask]
            else:
                child = pop[i].copy()
                mask = np.random.rand(p) < 0.5
                child[mask] = pop[i - 1][mask]
            cnt = np.sum(child)
            if cnt > k_eff:
                on_idx = np.where(child)[0]; np.random.shuffle(on_idx)
                child[on_idx[k_eff:]] = False
            elif cnt < k_eff:
                off_idx = np.where(~child)[0]
                child[np.random.choice(off_idx, k_eff - cnt, replace=False)] = True
            nf = evaluate(X, Y, np.where(child)[0])
            if nf > fit[i]: pop[i] = child; fit[i] = nf
            if nf > best_fit: best = child.copy(); best_fit = nf
    return np.where(best)[0]
