# GAM Revalidation (v2)

This directory contains the **fold-safe revalidation** of the main GAM benchmark, run after a code–data consistency audit that identified three implementation issues in the originally submitted scripts. The revalidation reruns the affected experiments and confirms the manuscript's core conclusions.

## Why revalidate

The pre-submission audit found three defects that could affect the primary results:

1. **SSA λ-invariance** — the original `ssa_gam_safe` never read the redundancy penalty `λ`, so `GAM-λ0` and `GAM λ=0.6` were identical for SSA. Rewritten here to integrate GAM's graph-guided crossover/mutation with the exact discrete best retained.
2. **`gen`/`max_gen` swallowed by the crossover wrapper** — `lambda p1, p2, k, g=0, mg=100, **kw: … gen=g, max_gen=mg` forwarded the *defaults* (0, 100) instead of the caller's generation counters, freezing the schedule (e.g. `r_t` stuck at `gen=0`). Fixed to `gen=0, max_gen=100, **kw: … gen=gen, max_gen=max_gen`.
3. **AO/POA/TSO generation call** — `gam.safe_crossover(..., gen=T, max_gen=T)` used the total budget `T` instead of the current generation `t`. Fixed to `gen=t`.

All three fixes are applied **in place** in the repository's `code/` and `experiments/` (the revalidation ran against the fixed `source/` snapshot reproduced here for provenance).

## What was rerun

| Experiment | Purpose | Tasks |
|-----------|---------|------:|
| `main` | Primary Bare / GAM-λ0 / GAM λ=0.6 benchmark (15 datasets × 5 base × 5 runs) | 776 |
| `lambda` | λ sensitivity | 270 |
| `E7` | Large-dimensional feasibility | 72 |
| `E4` | External optimizer transfer (AO/GWO) | 100 |
| `E10_matched` | HMRF alpha sensitivity (matched splits/seeds) | 45 |
| `convergence` | Search dynamics | 72 |
| `independent` | Independent binary wrappers | 63 |
| `filter_guided` | Filter-guided GAM variants | 126 |
| `component_ablation` | Strict GA component ablation (groups A/B/C/D) | 60 |

Total: 1524 main-queue tasks + 60 component tasks, **0 errors**. Bare arms whose per-fold redundancy was already finite and nonzero under the finite-pair convention were *retained* from the original run (`retained_unchanged_bare.jsonl`) rather than rerun, since the Bare algorithms were not modified.

## Key results

`finalize_statistics.py` produces the primary benchmark inference (15 datasets × 5 base optimizers × 3 modes × 5 runs = 1125 records):

| Mode | Macro Accuracy | Macro Redundancy (finite-pair) |
|------|---------------:|-------------------------------:|
| Bare | 0.7672 | 0.2139 |
| GAM-λ0 | 0.8168 | 0.3018 |
| GAM λ=0.6 | **0.8236** | 0.2718 |

- Friedman test over the three modes: χ² = 26.53, **p = 1.73 × 10⁻⁶**.
- Mean ranks: GAM 1.13 < GAM-λ0 1.87 < Bare 3.00.
- Exact paired Wilcoxon: Bare vs GAM **p = 6.1 × 10⁻⁵**; GAM-λ0 vs GAM **p = 0.0015**.

The revalidation reproduces the manuscript's accuracy gain (**+5.64 pp** Bare → GAM λ=0.6) and the λ-guided redundancy reduction (≈ −10% relative to GAM-λ0), with the redundancy absolute values shifted slightly because of the corrected finite-pair definition (the legacy definition set the mean to zero whenever any pair was undefined).

### Component ablation (mechanism decomposition)

Same GA implementation across groups; only the listed component changes:

| Group | Components | Accuracy | Redundancy |
|-------|-----------|---------:|-----------:|
| A | uniform init + random crossover | 0.7130 | 0.2467 |
| B | + HMRF top-pool init | 0.7459 | 0.2740 |
| C | + relevance-guided crossover (λ=0) | 0.7506 | 0.2809 |
| D | + λ=0.6 penalty | 0.7546 | 0.2695 |

The accuracy gain is driven primarily by **HMRF-guided initialization** (A→B ≈ +3.3 pp), with smaller contributions from the relevance-guided crossover and the λ penalty; λ's main role is reducing redundancy (C→D).

## How to reproduce

```bash
# 1. (one-time) snapshot the repository code/utils/experiments into source/ and apply fixes
python experiments/revalidation_v2/prepare.py

# 2. run the resumable revalidation (component ablation first, then the main queue)
python experiments/revalidation_v2/run_fast.py --workers <N>

# 3. aggregate raw fold JSONs
python experiments/revalidation_v2/summarize.py

# 4. primary benchmark inference (Friedman + exact Wilcoxon)
python experiments/revalidation_v2/finalize_statistics.py
```

Raw fold-level results (per-fold metrics, selected subsets, predictions, redundancy) are stored in `results/revalidation_v2/`.

## Directory layout

```text
experiments/revalidation_v2/
├── run_revalidation.py         # resumable fold-safe runner (main queue)
├── run_fast.py                 # coordinator: component ablation first, then main queue
├── component_ablation.py       # strict GA component ablation
├── prepare.py                  # snapshot + fix code (one-time)
├── fast_evaluator.py           # validated fast equivalence kernel
├── fast_crossover.py           # validated fast crossover kernel
├── summarize.py                # fold -> task-level aggregation
├── finalize_statistics.py      # primary benchmark inference
├── monitor_daemon.py           # health check / auto-restart / backup
├── source/                     # fixed code snapshot used by the revalidation
├── plan.json                   # full task plan (1524 tasks)
├── revision_notes.json         # description of the three fixes
├── execution_manifest.json     # source hashes, seed policy, environment
└── retained_unchanged_bare.jsonl  # retained Bare arms (not rerun)
```
