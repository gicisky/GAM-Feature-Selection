"""
POA+GAM, AO+GAM, TSO+GAM — Binary GAM-wrapped SOTA algorithms.
All use: exact-k topk_binary, graph-guided refinement (gen=T to skip random branch),
and pass penalty parameter for λ0/λ0.6 ablation.
"""
import numpy as np
import math

def sigmoid(x):
    x = np.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def topk_binary(scores, k_eff):
    """Exact-k selection: pick top k_eff features by score."""
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    idx = np.argsort(-scores)[:k_eff]
    b = np.zeros(len(scores), dtype=bool)
    b[idx] = True
    return b

def levy_flight(dims, beta=1.5):
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dims) * sigma
    v = np.random.randn(dims)
    step = u / (np.abs(v) ** (1 / beta) + 1e-12)
    return np.clip(step, -10, 10)

def poa_gam_safe(X, Y, k, gam, N=100, T=100, penalty=0.6):
    import bench_gam
    p = gam.p
    k_eff = min(k, p)
    evaluate = bench_gam.evaluate
    pop = gam.safe_init(N, k_eff)
    fit = np.array([evaluate(X, Y, np.where(pop[i])[0]) for i in range(N)])
    best_idx = np.argmax(fit)
    best = pop[best_idx].copy()
    best_fit = fit[best_idx]
    for t in range(T):
        for i in range(N):
            prey_idx = np.random.randint(N)
            while prey_idx == i:
                prey_idx = np.random.randint(N)
            I_factor = 1 if np.random.rand() < 0.5 else 2
            cont = pop[i].astype(float)
            prey_cont = pop[prey_idx].astype(float)
            direction = prey_cont - I_factor * cont
            new_cont = cont + np.random.rand(p) * direction
            new = topk_binary(sigmoid(new_cont), k_eff)
            child = gam.safe_crossover(new, pop[i], k_eff, gen=t, max_gen=T, penalty=penalty)
            child = gam.safe_mutate(child, k_eff)
            nf = evaluate(X, Y, np.where(child)[0])
            if nf > fit[i]:
                pop[i] = child
                fit[i] = nf
            if nf > best_fit:
                best = child.copy()
                best_fit = nf
        for i in range(N):
            R = 0.2 * (1.0 - t / T)
            cont = pop[i].astype(float)
            perturbation = R * (2 * np.random.rand(p) - 1)
            new_cont = cont + perturbation
            new = topk_binary(sigmoid(new_cont), k_eff)
            child = gam.safe_crossover(new, pop[i], k_eff, gen=t, max_gen=T, penalty=penalty)
            child = gam.safe_mutate(child, k_eff)
            nf = evaluate(X, Y, np.where(child)[0])
            if nf > fit[i]:
                pop[i] = child
                fit[i] = nf
            if nf > best_fit:
                best = child.copy()
                best_fit = nf
    return np.where(best)[0]

def ao_gam_safe(X, Y, k, gam, N=100, T=100, penalty=0.6):
    import bench_gam
    p = gam.p
    k_eff = min(k, p)
    evaluate = bench_gam.evaluate
    pop = gam.safe_init(N, k_eff)
    fit = np.array([evaluate(X, Y, np.where(pop[i])[0]) for i in range(N)])
    best_idx = np.argmax(fit)
    best = pop[best_idx].copy()
    best_fit = fit[best_idx]
    for t in range(T):
        progress = (t + 1) / T
        g1 = 2 * np.random.rand() - 1
        g2 = 2 * (1 - progress)
        for i in range(N):
            cont = pop[i].astype(float)
            best_cont = best.astype(float)
            if progress <= 2.0 / 3.0:
                r = np.pi * np.random.rand() * 0.5
                phi = np.pi * np.random.rand() * 0.5
                x_coord = r * np.sin(phi)
                y_coord = r * np.cos(phi)
                if np.random.rand() < 0.5:
                    pop_mean_cont = np.mean(pop.astype(float), axis=0)
                    levy = levy_flight(p, beta=1.5)
                    new_cont = best_cont * (1 - progress) + (pop_mean_cont - best_cont * np.random.rand()) * levy
                else:
                    rand_idx = np.random.randint(N)
                    rand_cont = pop[rand_idx].astype(float)
                    levy = levy_flight(p, beta=1.5)
                    new_cont = best_cont * levy + rand_cont + np.random.rand(p) * (y_coord - x_coord)
            elif np.random.rand() < 0.5:
                pop_mean_cont = np.mean(pop.astype(float), axis=0)
                alpha, delta = (0.1, 0.1)
                new_cont = (best_cont - pop_mean_cont) * alpha - np.random.rand(p) + np.random.rand(p) * delta
            else:
                QF = progress ** (2 * np.random.rand() - 1)
                levy = levy_flight(p, beta=1.5)
                new_cont = QF * best_cont - g1 * cont * np.random.rand() - g2 * levy + np.random.rand(p) * g1
            new = topk_binary(sigmoid(new_cont), k_eff)
            child = gam.safe_crossover(new, pop[i], k_eff, gen=t, max_gen=T, penalty=penalty)
            child = gam.safe_mutate(child, k_eff)
            nf = evaluate(X, Y, np.where(child)[0])
            if nf > fit[i]:
                pop[i] = child
                fit[i] = nf
            if nf > best_fit:
                best = child.copy()
                best_fit = nf
    return np.where(best)[0]

def tso_gam_safe(X, Y, k, gam, N=100, T=100, penalty=0.6):
    import bench_gam
    p = gam.p
    k_eff = min(k, p)
    evaluate = bench_gam.evaluate
    pop = gam.safe_init(N, k_eff)
    fit = np.array([evaluate(X, Y, np.where(pop[i])[0]) for i in range(N)])
    order = np.argsort(-fit)
    pop = pop[order]
    fit = fit[order]
    best = pop[0].copy()
    best_fit = fit[0]
    for t in range(T):
        c1 = 1.0 + t / max(T, 1)
        c2 = np.random.rand()
        c3 = np.random.rand()
        for i in range(N):
            cont = pop[i].astype(float)
            best_cont = best.astype(float)
            P_factor = (c1 + c2 - 2 * c3) / max(c1, 1e-12)
            rand_P = np.random.rand()
            if rand_P >= 0.5:
                PD = np.abs(best_cont - np.random.rand() * P_factor * cont)
                new_cont = best_cont + PD
            else:
                PD = np.abs(best_cont - np.random.rand() * P_factor * cont)
                new_cont = best_cont - PD
            if i > 0:
                neighbor_cont = pop[i - 1].astype(float)
                new_cont = (new_cont + neighbor_cont) / 2.0
            new = topk_binary(sigmoid(new_cont), k_eff)
            child = gam.safe_crossover(new, pop[i], k_eff, gen=t, max_gen=T, penalty=penalty)
            child = gam.safe_mutate(child, k_eff)
            nf = evaluate(X, Y, np.where(child)[0])
            if nf > fit[i]:
                pop[i] = child
                fit[i] = nf
            if nf > best_fit:
                best = child.copy()
                best_fit = nf
        order = np.argsort(-fit)
        pop = pop[order]
        fit = fit[order]
        if fit[0] > best_fit:
            best = pop[0].copy()
            best_fit = fit[0]
    return np.where(best)[0]
