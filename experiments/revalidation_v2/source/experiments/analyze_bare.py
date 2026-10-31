"""
Post-Bare Analysis: Table A (Acc), Table B (Red), Table C (ΔAcc/ΔRed).
Run after bench_foldsafe_bare.jsonl is complete.
"""
import os, json, numpy as np
from collections import defaultdict
RES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

def load(path):
    return [json.loads(l) for l in open(path) if 'error' not in json.loads(l)]
bare = load(f'{RES_DIR}/bench_foldsafe_bare.jsonl')
main = load(f'{RES_DIR}/bench_foldsafe_full.jsonl')
g0 = [d for d in main if d['protocol'] == 'fold-safe' and d['mode'] == 'GAM-lambda0']
gam = [d for d in main if d['protocol'] == 'fold-safe' and d['mode'] == 'GAM']
BASES = ['GA', 'CSA', 'SSA', 'PSO', 'DE']

def get(dlist, ds, base):
    entries = [d for d in dlist if d['dataset'] == ds and d['base'] == base]
    if not entries:
        return None
    return (np.mean([d['acc_mean'] for d in entries]), np.std([d['acc_mean'] for d in entries]))

def get_red(dlist, ds, base):
    entries = [d for d in dlist if d['dataset'] == ds and d['base'] == base]
    if not entries:
        return None
    return (np.mean([d['red_mean'] for d in entries]), np.std([d['red_mean'] for d in entries]))
datasets = sorted(set((d['dataset'] for d in bare)))
common_ds = [ds for ds in datasets if all((len([d for d in lst if d['dataset'] == ds]) > 0 for lst in [bare, g0, gam]))]
print('=' * 100)
print('Table A: Accuracy (mean ± std) — Bare vs GAM-λ0 vs GAM λ=0.6')
print('=' * 100)
header = f"{'Dataset':16s} {'Bare':>14s} {'GAM-λ0':>14s} {'GAM λ=0.6':>14s}"
print(header)
print('-' * len(header))
for ds in common_ds:
    b = get(bare, ds, 'GA')
    g0_v = get(g0, ds, 'GA')
    g1_v = get(gam, ds, 'GA')
    if b and g0_v and g1_v:
        print(f'{ds:16s} {b[0]:6.4f}±{b[1]:.4f}  {g0_v[0]:6.4f}±{g0_v[1]:.4f}  {g1_v[0]:6.4f}±{g1_v[1]:.4f}')
print('\n' + '=' * 100)
print('Table B: Redundancy (mean ± std) — Bare vs GAM-λ0 vs GAM λ=0.6')
print('=' * 100)
for ds in common_ds:
    b = get_red(bare, ds, 'GA')
    g0_v = get_red(g0, ds, 'GA')
    g1_v = get_red(gam, ds, 'GA')
    if b and g0_v and g1_v:
        print(f'{ds:16s} {b[0]:6.4f}±{b[1]:.4f}  {g0_v[0]:6.4f}±{g0_v[1]:.4f}  {g1_v[0]:6.4f}±{g1_v[1]:.4f}')
print('\n' + '=' * 100)
print('Table C: Per-Base ΔAcc and ΔRed (macro avg across all datasets)')
print('=' * 100)
print(f"{'Base':8s} {'Bare→GAM-λ0 ΔAcc':>16s} {'GAM-λ0→GAM ΔAcc':>16s} {'Bare→GAM ΔAcc':>16s} {'ΔRed(Bare→GAM)':>16s}")
print('-' * 80)
for base in ['GA', 'CSA', 'SSA', 'PSO', 'DE']:
    bare_base = base
    if base not in set((d['base'] for d in bare)):
        continue
    bare_accs, g0_accs, g1_accs = ([], [], [])
    bare_reds, g0_reds, g1_reds = ([], [], [])
    for ds in common_ds:
        b = get(bare, ds, bare_base)
        g0_v = get(g0, ds, base)
        g1_v = get(gam, ds, base)
        br = get_red(bare, ds, bare_base)
        g0_r = get_red(g0, ds, base)
        g1_r = get_red(gam, ds, base)
        if all([b, g0_v, g1_v, br, g0_r, g1_r]):
            bare_accs.append(b[0])
            g0_accs.append(g0_v[0])
            g1_accs.append(g1_v[0])
            bare_reds.append(br[0])
            g0_reds.append(g0_r[0])
            g1_reds.append(g1_r[0])
    if bare_accs:
        d1 = np.mean(g0_accs) - np.mean(bare_accs)
        d2 = np.mean(g1_accs) - np.mean(g0_accs)
        d3 = np.mean(g1_accs) - np.mean(bare_accs)
        dr = np.mean(g1_reds) - np.mean(bare_reds)
        print(f'{base:8s} {d1:+16.4f} {d2:+16.4f} {d3:+16.4f} {dr:+16.4f}')
print('-' * 80)
all_b, all_g0, all_g1 = ([], [], [])
all_rb, all_rg0, all_rg1 = ([], [], [])
for base in ['GA', 'CSA', 'SSA', 'PSO', 'DE']:
    for ds in common_ds:
        b = get(bare, ds, base)
        g0_v = get(g0, ds, base)
        g1_v = get(gam, ds, base if base in set((d['base'] for d in bare)) else 'GA')
        if all([b, g0_v, g1_v]):
            all_b.append(b[0])
            all_g0.append(g0_v[0])
            all_g1.append(g1_v[0])
            br = get_red(bare, ds, base)
            g0_r = get_red(g0, ds, base)
            g1_r = get_red(gam, ds, base)
            if all([br, g0_r, g1_r]):
                all_rb.append(br[0])
                all_rg0.append(g0_r[0])
                all_rg1.append(g1_r[0])
d1 = np.mean(all_g0) - np.mean(all_b)
d2 = np.mean(all_g1) - np.mean(all_g0)
d3 = np.mean(all_g1) - np.mean(all_b)
dr = np.mean(all_rg1) - np.mean(all_rb)
print(f"{'Overall':8s} {d1:+16.4f} {d2:+16.4f} {d3:+16.4f} {dr:+16.4f}")
