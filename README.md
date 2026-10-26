# Graph-Augmented Metaheuristic (GAM)

> **A Fold-Safe Plug-and-Play Framework for Wrapper Feature Selection in High-Dimensional Biomedical Data**


[![DOI](https://zenodo.org/badge/1293359484.svg)](https://doi.org/10.5281/zenodo.21272539)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/Status-Submitted-lightgrey)
![Domain](https://img.shields.io/badge/Domain-Feature%20Selection-orange)

---

## Graphical Abstract

<p align="center">
  <img src="docs/graphical_abstract.png" width="950">
</p>

<p align="center">
  <em>Fold-safe graph-augmented operator framework for redundancy-aware wrapper feature selection.</em>
</p>

---

## Introduction

**GAM** (**Graph-Augmented Metaheuristic**) is a fold-safe, plug-and-play wrapper feature-selection framework for high-dimensional biomedical classification.

Instead of proposing another standalone optimizer, GAM augments existing metaheuristic searchers with fold-internal supervised guidance, an intra-fold feature-correlation graph, graph-guided operators, and a λ-controlled redundancy penalty. All supervised guidance is rebuilt within each training fold, preventing full-data scoring from leaking information into model selection.

This repository accompanies the manuscript submitted to **Applied Soft Computing**. It provides source code, processed benchmark datasets, raw experimental outputs, reproducibility scripts, and supplementary analyses.

---

## Highlights

- **Fold-safe protocol**: scaling, supervised scoring, candidate-pool construction, and graph construction are performed inside training folds.
- **Plug-and-play operator layer**: GAM can be attached to heterogeneous metaheuristic optimizers without replacing their core search rules.
- **Graph-guided search**: an intra-fold feature-correlation graph guides initialization, crossover, mutation, and repair.
- **Redundancy-aware penalty**: a λ-controlled operator-level penalty reduces redundant combinations within the GAM framework.
- **Broad validation**: experiments cover ablations, classical optimizers, recent searchers, external optimizers, filter baselines, binary wrappers, convergence behavior, large-dimensional settings, stability analysis, embedded baselines, multimetric endpoint robustness, and HMRF alpha sensitivity.

---

## Main Results Preview

| Metric | Bare | GAM-λ0 | GAM λ=0.6 |
|--------|-----:|-------:|----------:|
| Macro Accuracy | 0.7672 | 0.8187 | **0.8236** |
| Macro Redundancy | 0.2036 | 0.2967 | **0.2660** |
| ΔRed vs GAM-λ0 | — | — | **−10.33%** |

GAM improves the macro-average fold-safe accuracy from **0.7672** to **0.8236**, corresponding to a **5.64 percentage-point gain** over Bare optimizers. Within the GAM framework, the λ-guided redundancy penalty reduces redundancy by **10.33%** relative to GAM-λ0 while preserving the same accuracy tier.

---

## Additional Supplementary Analyses

| Analysis | Main purpose | Output file |
|----------|--------------|-------------|
| Minimal reproducible demo | Verify installation and fold-safe workflow | terminal output |
| Convergence monitoring | Inspect search dynamics and inner-CV fitness trajectories | `results/bench_convergence.jsonl` |
| Jaccard selected-subset stability | Measure reproducibility of selected feature subsets across folds and runs | `results/bench_jaccard_stability.jsonl` |
| Embedded baselines | Boundary comparison with LASSO, Elastic Net, and Random Forest feature importance | `results/bench_embedded_baselines.jsonl` |
| Multimetric endpoint robustness | Check accuracy, balanced accuracy, macro-F1, and weighted-F1 under the same fold-safe wrapper logic | `results/bench_multimetric_robustness_clean.jsonl` |
| HMRF alpha sensitivity | Evaluate whether the fixed MI/ReliefF mixing weight is a fragile design choice | `results/bench_alpha_sensitivity_shard_0..4_of_5.jsonl` |

### Jaccard stability preview

For 5 monitored datasets, each method produced 25 selected subsets from 5 runs × 5 folds, giving C(25,2)=300 subset pairs per dataset/method.

| Method | Macro Jaccard stability |
|--------|------------------------:|
| Bare-GA | 0.0209 |
| GAM-λ0 | 0.1184 |
| GAM λ=0.6 | **0.1273** |

The stability analysis shows that GAM produces substantially more reproducible selected subsets than Bare-GA on the monitored datasets.

### Multimetric endpoint robustness preview

The wrapper fitness remains inner-CV KNN accuracy, while the additional endpoints are computed only on held-out outer folds.

| Metric | Bare-GA | GAM-GA | Δ | W/L/T | Wilcoxon p |
|--------|--------:|-------:|--:|:-----:|-----------:|
| Accuracy | 0.7765 | **0.8264** | +0.0499 | 7/0/0 | 0.0156 |
| Balanced accuracy | 0.7427 | **0.8029** | +0.0601 | 7/0/0 | 0.0156 |
| Macro-F1 | 0.7231 | **0.7887** | +0.0656 | 7/0/0 | 0.0156 |
| Weighted-F1 | 0.7586 | **0.8125** | +0.0539 | 7/0/0 | 0.0156 |

This analysis supports the manuscript's endpoint-robustness statement: the GAM-GA gain is not limited to raw accuracy. It is reported as supplementary evidence rather than as a replacement for the 15-dataset primary benchmark.

### HMRF alpha sensitivity preview

The primary experiments use equal MI/ReliefF weighting (`alpha=0.5`) as a pre-specified default. The supplementary alpha analysis evaluates `alpha ∈ {0.2, 0.4, 0.5, 0.6, 0.8}` on colon, DLBCL, and 9_Tumors using GAM-GA under a fold-safe 5-fold, 3-run protocol.

| Alpha | Accuracy | Balanced Acc | Macro-F1 | Redundancy |
|------:|---------:|-------------:|---------:|-----------:|
| 0.2 | 0.7468 | 0.7376 | 0.7017 | 0.2543 |
| 0.4 | 0.7565 | 0.7465 | 0.7214 | 0.2618 |
| 0.5 | 0.7307 | 0.7232 | 0.6962 | 0.2641 |
| 0.6 | 0.7596 | 0.7510 | **0.7242** | 0.2665 |
| 0.8 | **0.7620** | **0.7604** | 0.7201 | 0.2635 |

These results should not be used to retune the main benchmark. They show that HMRF weighting is not a single-point artifact and that mid-to-high MI weights (`alpha=0.6-0.8`) are promising for future adaptive weighting.

### Embedded baseline preview

Representative embedded feature-selection baselines are evaluated under the same fold-safe top-k KNN protocol.

| Method | Macro Accuracy |
|--------|---------------:|
| LASSO | 0.7726 |
| Elastic Net | **0.7927** |
| Random Forest | 0.7871 |

These baselines are reported as supplementary boundary comparisons. They are competitive under fixed-k accuracy evaluation, but they do not provide optimizer-agnostic graph-guided variation operators or explicit redundancy-aware wrapper search.

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

| Experiment | Purpose | Main output |
|-----------|---------|-------------|
| E1 Bare ablation | Native optimizers without HMRF/graph/λ | `bench_foldsafe_bare.jsonl` |
| E2 Main GAM benchmark | GAM-λ0 vs GAM λ=0.6 under fold-safe protocol | `bench_foldsafe_full.jsonl` |
| E3 λ sensitivity | Robustness of λ=0.6 across representative datasets and base optimizers | `bench_lambda_sensitivity.jsonl` |
| E4 External optimizer transfer | AO/GWO external validation | `bench_foldsafe_sota.jsonl` |
| E5 Recent searcher transfer | SCSO/COA transfer validation | `bench_recent_searchers.jsonl` |
| E6 Filter baselines | MI, ReliefF, mRMR, and F-score boundary baselines | `bench_foldsafe_filter.jsonl` |
| E7 Large-dimensional feasibility | Feasibility under high-dimensional limited-budget setting | `bench_foldsafe_large_optimized.jsonl` |
| E8 Independent wrappers | BGWO/BPSO/GAM-GA comparison | `bench_independent_wrappers.jsonl` |
| E9 Multimetric endpoint robustness | Accuracy, balanced accuracy, macro-F1, and weighted-F1 | `bench_multimetric_robustness_clean.jsonl` |
| E10 HMRF alpha sensitivity | Sensitivity of the MI/ReliefF mixing weight | `bench_alpha_sensitivity_shard_0..4_of_5.jsonl` |
| Supplementary convergence | Search-process behavior | `bench_convergence.jsonl` |
| Supplementary stability | Selected-subset reproducibility across folds and runs | `bench_jaccard_stability.jsonl` |
| Supplementary embedded baselines | LASSO, Elastic Net, and Random Forest boundary baselines | `bench_embedded_baselines.jsonl` |

---

## Repository Structure

```text
GAM-Feature-Selection/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
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
│   └── ... (16 processed benchmark CSVs)
│
├── docs/
│   ├── EXPERIMENT_DATA.md
│   └── graphical_abstract.png
│
├── experiments/
│   ├── verify_crossover_equivalence.py   # new/old crossover equivalence test
│   ├── run_large_scale.py                # E7
│   ├── run_multimetric_robustness.py     # E9
│   ├── run_alpha_sensitivity.py          # E10
│   ├── run_jaccard_stability.py
│   ├── run_embedded_baselines.py
│   ├── run_bare.py
│   ├── run_convergence.py
│   ├── run_ablation.py
│   ├── run_external_baselines.py
│   ├── run_filter_baselines.py
│   ├── run_filter_wrapper.py
│   ├── run_independent_wrappers.py
│   ├── run_lambda_large.py
│   ├── run_lambda_sensitivity.py
│   ├── run_recent_searchers.py
│   ├── run_sota_large.py
│   ├── rerun_residual_large.py
│   ├── rerun_sota_foldsafe.py
│   ├── patch_dlbcl_gwo.py
│   ├── analyze_ablation.py
│   ├── analyze_alpha_sensitivity.py
│   ├── analyze_bare.py
│   ├── analyze_multimetric_robustness.py
│   └── ...
│
├── results/
│   ├── bench_foldsafe_full.jsonl
│   ├── bench_foldsafe_bare.jsonl
│   ├── bench_lambda_sensitivity.jsonl
│   ├── bench_foldsafe_sota.jsonl
│   ├── bench_foldsafe_filter.jsonl
│   ├── bench_foldsafe_large_optimized.jsonl
│   ├── bench_convergence.jsonl
│   ├── bench_filter_guided_gam.jsonl
│   ├── bench_independent_wrappers.jsonl
│   ├── bench_recent_searchers.jsonl
│   ├── bench_multimetric_robustness_clean.jsonl
│   ├── bench_phase_breakdown.jsonl
│   ├── bench_alpha_sensitivity_shard_0..4_of_5.jsonl
│   ├── bench_jaccard_stability.jsonl
│   └── bench_embedded_baselines.jsonl
│
└── utils/
    ├── Filter.py
    └── ReliefF.py
```

## Installation

```bash
git clone https://github.com/gicisky/GAM-Feature-Selection.git
cd GAM-Feature-Selection
pip install -r requirements.txt
```

Optional environment check:

```bash
python -c "import numpy, sklearn; print('Environment ready')"
```

---

## Minimal Reproducible Demo

Run a lightweight workflow-verification demo:

```bash
python experiments/run_demo.py
```

The default demo runs a reduced fold-safe GAM workflow on the colon dataset using GA and a lightweight search budget. It is intended for installation and pipeline verification only; paper-level claims are based on the full benchmark reported in the manuscript.

A typical full-demo run on the colon dataset illustrates the expected mechanism:

| Method | Accuracy | Redundancy |
|--------|---------:|-----------:|
| Bare-GA | 0.7325 | 0.3775 |
| GAM-λ0 | 0.7868 | 0.3699 |
| GAM λ=0.6 | 0.7709 | 0.3404 |

The demo shows the intended three-level behavior: GAM-λ0 improves search quality over Bare-GA, while GAM λ=0.6 reduces redundancy within the GAM framework. Exact numbers may vary slightly across machines, package versions, and random seeds.

For a more stable but slower demonstration, use the optional full-demo setting if available:

```bash
python experiments/run_demo.py --full-demo
```

---

## Running Supplementary Analyses

Selected-subset stability:

```bash
python experiments/run_jaccard_stability.py
```

Embedded feature-selection baselines:

```bash
python experiments/run_embedded_baselines.py
```

Multimetric endpoint robustness:

```bash
python experiments/run_multimetric_robustness.py
```

HMRF alpha sensitivity:

```bash
python experiments/run_alpha_sensitivity.py --num-shards 1 --shard-index 0
python experiments/analyze_alpha_sensitivity.py \
  --input results/bench_alpha_sensitivity_shard_0..4_of_5.jsonl \
  --out-dir results/alpha_sensitivity_summary_5proc
```

---

## Datasets

The repository contains processed benchmark CSV files for reproducibility. These datasets originate from public biomedical repositories and widely used feature-selection benchmark collections.

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

Users should refer to the paper for the original data sources and citations.

---

## Fold-Safe Protocol

For each outer fold:

1. Fit the scaler on the outer training set only.
2. Compute MI and ReliefF scores only on the outer training set.
3. Construct the feature-correlation graph only from the outer training set.
4. Run wrapper search using inner cross-validation on the outer training set.
5. Evaluate the selected subset on the held-out outer test fold.

This protocol prevents full-data supervised scoring from inflating performance estimates.

---

## Reproducibility Notes

The full benchmark is computationally intensive. Users are encouraged to first run `experiments/run_demo.py` to verify the environment and workflow. Raw result files used in the manuscript are provided in `results/`.

The supplementary scripts are provided for transparency, robustness checks, and boundary analysis. They should not be interpreted as replacing the full fold-safe benchmark reported in the manuscript. In particular, multimetric endpoints and HMRF alpha sensitivity are reported as supplementary checks rather than as new optimization objectives or post-hoc retuning evidence.

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{Ma2026GAM,
  title   = {Fold-Safe Graph-Augmented Metaheuristic Feature Selection with Redundancy-Aware Operators for High-Dimensional Biomedical Classification},
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
