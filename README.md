# Graph-Augmented Metaheuristic (GAM)

> **A Fold-Safe Plug-and-Play Framework for Wrapper Feature Selection in High-Dimensional Biomedical Data**

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/Status-Submitted-lightgrey)
![Domain](https://img.shields.io/badge/Domain-Feature%20Selection-orange)

---

## Graphical Abstract

<p align="center">
  <img src="docs/FIGURES/graphical_abstract.png" width="950">
</p>

<p align="center">
  <em>Fold-safe graph-augmented operator framework for redundancy-aware wrapper feature selection.</em>
</p>

---

## Introduction

**GAM** (**Graph-Augmented Metaheuristic**) is a fold-safe, plug-and-play wrapper feature-selection framework designed for high-dimensional biomedical classification.

Instead of proposing another standalone optimizer, GAM augments existing metaheuristic searchers with fold-internal supervised guidance, an intra-fold feature-correlation graph, graph-guided operators, and a λ-controlled redundancy penalty. All supervised guidance is rebuilt within each training fold, preventing full-data scoring from leaking information into model selection.

This repository accompanies the manuscript submitted to **Applied Soft Computing**. It provides the source code, processed benchmark datasets, raw experimental outputs, and documentation required to reproduce the main experiments.

---

## Highlights

- **Fold-safe protocol**: all supervised scoring, scaling, and graph construction are performed inside the training folds.
- **Plug-and-play operator layer**: GAM can be attached to heterogeneous metaheuristic optimizers without replacing their core search rules.
- **Graph-guided search**: an intra-fold feature-correlation graph guides initialization, crossover, and mutation.
- **Redundancy-aware penalty**: a λ-controlled operator-level penalty reduces redundant feature combinations.
- **Broad validation**: experiments cover classical optimizers, recent searchers, filter baselines, ablations, convergence behavior, and large-dimensional datasets.

---

## Main Results Preview

| Metric | Bare | GAM-λ0 | GAM λ=0.6 |
|--------|------|--------|-----------|
| Macro Accuracy | 0.7672 | 0.8187 | **0.8236** |
| Macro Redundancy | 0.2036 | 0.2967 | **0.2660** |
| ΔRed vs GAM-λ0 | — | — | **−10.33%** |

GAM improves the macro-average fold-safe accuracy from **0.7672** to **0.8236**, corresponding to a **5.64 percentage-point gain** over Bare optimizers. The λ-guided redundancy penalty further reduces redundancy by **10.33%** relative to GAM-λ0 while preserving the same accuracy tier.

---

## Supported Optimizers

| Category | Algorithms |
|----------|------------|
| Classical evolutionary / swarm optimizers | GA, CSA, SSA, PSO, DE |
| External SOTA optimizers | AO, GWO |
| Recent metaheuristic searchers | SCSO, COA |
| Independent binary wrapper baselines | BGWO, BPSO |

---

## Experimental Matrix

| Experiment | Purpose | Output |
|-----------|---------|--------|
| E1 Bare ablation | Native optimizers without HMRF/graph/λ | `bench_foldsafe_bare.jsonl` |
| E2 Main benchmark | GAM-λ0 vs GAM λ=0.6 under full and fold-safe protocols | `bench_foldsafe_full.jsonl` |
| E3 λ sensitivity | Robustness of λ across datasets and bases | `bench_lambda_sensitivity.jsonl` |
| E4 SOTA comparison | AO/GWO external validation | `bench_foldsafe_sota.jsonl` |
| E5 Filter baselines | MI, ReliefF, mRMR, F-score | `bench_foldsafe_filter.jsonl` |
| E6 Large-scale evaluation | Feasibility under high-dimensional limited-budget setting | `bench_foldsafe_large.jsonl` |
| E7 Convergence analysis | Search-process behavior | `bench_convergence.jsonl` |
| E8 Filter-guided GAM variants | Scoring-signal ablation | `bench_filter_guided_gam.jsonl` |
| E9 Independent wrappers | BGWO/BPSO/GAM-GA comparison | `bench_independent_wrappers.jsonl` |
| E10 Recent searchers | SCSO/COA transfer validation | `bench_recent_searchers.jsonl` |

---

## Repository Structure

The repository is organized according to the current project layout:

```text
GAM/
├── README.md
├── LICENSE
├── requirements.txt
│
├── code/
│   ├── bench_gam.py
│   ├── bench_bare.py
│   └── bench_gam_sota.py
│
├── data/
│   ├── colon(62,2000).csv
│   ├── DLBCL(77,5469).csv
│   ├── arcene(200,10000).csv
│   └── ...
│
├── docs/
│   ├── EXPERIMENT_DATA.md
│   ├── README.md
│   └── FIGURES/
│       └── graphical_abstract.png
│
├── experiments/
│   ├── run_bare.py
│   ├── analyze_ablation.py
│   ├── rerun_sota_foldsafe.py
│   └── ...
│
├── results/
│   ├── bench_foldsafe_full.jsonl
│   ├── bench_foldsafe_bare.jsonl
│   ├── bench_lambda_sensitivity.jsonl
│   ├── bench_foldsafe_sota.jsonl
│   ├── bench_foldsafe_filter.jsonl
│   ├── bench_foldsafe_large.jsonl
│   ├── bench_convergence.jsonl
│   ├── bench_filter_guided_gam.jsonl
│   ├── bench_independent_wrappers.jsonl
│   └── bench_recent_searchers.jsonl
│
└── utils/
    ├── Filter.py
    └── ReliefF.py
```

---

## Installation

```bash
git clone https://github.com/gicisky/GAM-Feature-Selection.git
cd GAM-Feature-Selection

pip install -r requirements.txt
```

Optional check:

```bash
python -c "import numpy, sklearn; print('Environment ready')"
```

---

## Quick Start

Run the main benchmark:

```bash
python experiments/run_main.py
```

Run the ablation study:

```bash
python experiments/run_bare.py
python experiments/analyze_ablation.py
```

Run external SOTA validation:

```bash
python experiments/rerun_sota_foldsafe.py
```

Run λ-sensitivity analysis:

```bash
python experiments/run_lambda_sensitivity.py
```

The exact script names may vary slightly depending on the released version. See `docs/EXPERIMENT_DATA.md` for the complete experiment-to-file mapping.

---

## Datasets

The repository contains processed benchmark CSV files for reproducibility. These datasets originate from public biomedical repositories and widely used feature-selection benchmark collections.

Users should refer to `docs/DATASETS.md` or the paper for the original data sources and citations.

Representative datasets include:

| Dataset | Samples | Features |
|---------|--------:|---------:|
| colon | 62 | 2,000 |
| SRBCT | 83 | 2,308 |
| Leukemia1 | 72 | 5,327 |
| DLBCL | 77 | 5,469 |
| 9_Tumors | 60 | 5,726 |
| Brain_Tumor1 | 90 | 5,920 |
| arcene | 200 | 10,000 |
| Brain_Tumor_2 | 49 | 10,368 |
| Lung | 202 | 12,601 |
| Prostate | 102 | 12,600 |
| CrohnDisease | 127 | 22,283 |

---

## Fold-Safe Protocol

For each outer fold:

1. Fit scaler on the outer training set only.
2. Compute MI and ReliefF scores only on the outer training set.
3. Construct the feature-correlation graph only from the outer training set.
4. Run wrapper search using inner cross-validation on the outer training set.
5. Evaluate the selected subset on the held-out outer test fold.

This protocol prevents full-data supervised scoring from inflating performance estimates.

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{Ma2026GAM,
  title   = {Graph-Augmented Metaheuristic Framework for Wrapper Feature Selection},
  author  = {Ma, Qinwei and Zhang, Pengyi and Chen, Kansong},
  journal = {Applied Soft Computing},
  year    = {2026},
  note    = {Submitted},
  url     = {https://github.com/gicisky/GAM-Feature-Selection}
}
```

---

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## Contact

**Qinwei Ma**  
School of Computer Science, Hubei University  
Email: mqw@stu.hubu.edu.cn

If you encounter any issue, please open an issue in this repository.
