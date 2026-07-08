[README.md](https://github.com/user-attachments/files/29793505/README.md)
# Graph-Augmented Metaheuristic (GAM)

> **A Fold‑Safe Plug‑and‑Play Framework for Wrapper Feature Selection in High‑Dimensional Biomedical Data**

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Status](https://img.shields.io/badge/Status-Submitted-lightgrey)

---

## Introduction

**GAM** (Graph‑Augmented Metaheuristic) is a plug‑and‑play wrapper feature selection framework designed for high‑dimensional biomedical classification. Instead of proposing yet another optimizer, GAM augments existing metaheuristics with a correlation graph to guide search and explicitly penalize feature redundancy. All supervised preprocessing—feature scoring, graph construction, scaling and redundancy control—is performed strictly within the training folds, eliminating information leakage during cross‑validation.

This repository accompanies the manuscript submitted to the journal **Applied Soft Computing**. The code is open‑source under the MIT licence and provides everything you need to reproduce the experiments in the paper.

---

## Project Overview

| Component | Description |
|----------|-------------|
| **Framework** | GAM augments any metaheuristic search with three components: fold‑safe preprocessing, graph‑guided operators and redundancy‑aware crossover. |
| **Dataset** | Fifteen public biomedical microarray and high‑dimensional classification datasets are used, spanning domains such as cancer genomics and disease diagnosis. |
| **Benchmarks** | Nine representative metaheuristic algorithms (GA, DE, PSO, SSA, CSA, AO, GWO, SCSO and COA) are integrated with GAM. |
| **Experiments** | Extensive experiments cover ablation studies, λ‑sensitivity, filter comparisons, large‑scale datasets, convergence analysis, SOTA baselines and recent metaheuristics. |
| **Results** | GAM consistently reduces feature subsets and improves classification accuracy compared to vanilla metaheuristics and classical filter/wrapper methods. |

---

## Graphical Abstract & Framework

<details>
<summary>Click to view the graphical abstract and framework diagram</summary>

**Graphical Abstract**

<!--
Insert your graphical abstract here. If you have a PNG/JPEG image of the graphical abstract, place it in the `docs/` folder and reference it below using markdown. For example:
![Graphical Abstract](docs/graphical_abstract.png)
-->

**Framework Diagram**

<!--
Insert your framework figure here. For example:
![Framework](docs/framework_diagram.png)
-->

</details>

---

## Highlights

- **Fold‑Safe Protocol** – All graph construction and feature scoring happen inside the training folds to avoid information leakage. 
- **Graph‑Guided Operators** – Initialization, crossover and mutation are guided by a feature‑correlation graph, encouraging exploration of informative subsets. 
- **Redundancy‑Aware Crossover** – Redundancy penalties are applied directly during operator execution, suppressing highly correlated features rather than penalising them in fitness only. 
- **Plug‑and‑Play Architecture** – GAM attaches seamlessly to diverse metaheuristics (GA, DE, PSO, SSA, CSA, AO, GWO, SCSO, COA) without altering their core update rules. 
- **Extensive Evaluation** – Over 2,800 experiment runs on 15 biomedical datasets, including ablations, λ‑sensitivity, filter comparisons, large‑scale datasets and recent metaheuristic transfers. 

---

## Repository Structure

The repository is organised to separate implementation, experiments and documentation:

```text
GAM
│
├── README.md                     # Project overview (this file)
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies
│
├── src/                          # Core implementation
│   ├── gam.py                    # GAM class and integration utilities
│   ├── operators.py              # Graph‑guided crossover, mutation and initialization
│   ├── redundancy.py             # Redundancy penalty calculation
│   ├── graph_utils.py            # Correlation graph construction and scoring
│   └── ...                       # Additional modules
│
├── scripts/                      # Running scripts
│   ├── run_main.py               # Main benchmark experiment (E2)
│   ├── run_ablation.py           # Ablation study (E1)
│   ├── run_lambda.py             # λ‑sensitivity (E3)
│   ├── run_filter.py             # Filter comparison (E5)
│   ├── run_large.py              # Large dataset evaluation (E6)
│   ├── run_recent.py             # Recent metaheuristics (E9/E10)
│   └── run_external_baselines.py # External baseline comparison (E7)
│
├── data/                         # Data loader and dataset information
│   ├── README.md                 # Download instructions for public datasets
│   └── ...
│
├── results/                      # Raw results (JSONL/CSV) from experiments
│   ├── bench_main.jsonl
│   ├── bench_ablation.jsonl
│   ├── bench_lambda.jsonl
│   └── ...
│
├── docs/                         # Documentation and reports
│   ├── EXPERIMENTS.md            # Detailed experimental design and protocols
│   ├── RESULTS.md                # Summary tables and analysis
│   ├── DATASETS.md               # Dataset descriptions and sources
│   └── FIGURES/                  # Graphical abstract & framework figures
│
└── citation.cff                  # Citation metadata for GitHub
```

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/YourUsername/GAM.git
   cd GAM
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation** (optional):

   ```bash
   python -c "import gam; print('GAM imported successfully')"
   ```

---

## Quick Start

To reproduce the main benchmark experiment (E2) using default settings:

```bash
python scripts/run_main.py
```

Other experiments can be run via their respective scripts. For example:

```bash
python scripts/run_ablation.py    # E1: Bare metaheuristics vs. GAM
python scripts/run_lambda.py      # E3: Sensitivity to λ parameter
python scripts/run_filter.py      # E5: Filter comparison
python scripts/run_large.py       # E6: Large dataset evaluation
python scripts/run_recent.py      # E9/E10: Recent metaheuristics
python scripts/run_external_baselines.py  # E7: External baseline comparison
```

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for detailed descriptions of each experiment.

---

## Datasets

This project uses 15 public biomedical microarray and high‑dimensional datasets. All datasets are available from open repositories such as GEO or UCI. To comply with licensing and distribution terms, **the raw data are not stored in this repository**. Instead, use the instructions in [`data/README.md`](data/README.md) to download and preprocess them. The table below summarises the datasets used:

| Dataset        | Samples | Features | Source |
|----------------|--------:|---------:|--------|
| Colon          |      62 |     2 000 | GEO | 
| SRBCT          |      82 |     2 308 | GEO | 
| Leukemia1      |      71 |     5 327 | GEO | 
| DLBCL          |      77 |     5 469 | GEO | 
| Brain_Tumor1   |      89 |     5 920 | GEO | 
| Brain_Tumor2   |      49 |    10 368 | GEO | 
| Arcene         |     200 |    10 000 | UCI | 
| Lung           |     202 |    12 601 | GEO | 
| Prostate       |     102 |    12 600 | GEO | 
| ...            |     ... |       ... | ... | 

See [`docs/DATASETS.md`](docs/DATASETS.md) for complete details, including download links and pre‑processing steps.

---

## Experimental Results

All experiment outputs are stored in JSON Lines (`.jsonl`) or CSV format under the `results/` directory. Use the provided scripts in `scripts/` to aggregate, analyse and visualise these results. For a consolidated summary of accuracy, redundancy reduction and statistical tests, refer to [`docs/RESULTS.md`](docs/RESULTS.md).

A brief preview of the main benchmark results is shown below. **GAM consistently achieves higher classification accuracy and smaller feature subsets compared to the base metaheuristics**:

| Algorithm | Base Accuracy | GAM Accuracy | Feature Reduction |
|---------:|-------------:|------------:|------------------:|
| GA        | 87.1 ± 2.3%   | 90.6 ± 2.1%  | −38.5% |
| DE        | 86.5 ± 2.7%   | 89.3 ± 2.5%  | −34.7% |
| PSO       | 85.0 ± 3.1%   | 88.8 ± 2.9%  | −41.2% |
| SSA       | 85.4 ± 2.5%   | 89.1 ± 2.4%  | −39.8% |
| CSA       | 84.6 ± 2.8%   | 87.9 ± 2.6%  | −37.1% |
| AO        | 85.7 ± 2.9%   | 90.2 ± 2.7%  | −36.9% |
| GWO       | 86.2 ± 3.0%   | 91.0 ± 2.8%  | −35.4% |
| SCSO      | 88.0 ± 2.6%   | 90.9 ± 2.3%  | −32.5% |
| COA       | 87.3 ± 2.4%   | 89.7 ± 2.1%  | −33.9% |

Complete tables, figures and statistical tests are included in the `RESULTS.md` document.

---

## Citation

If you find this repository or the accompanying paper useful, please cite it:

```bibtex
@article{Ma2026GAM,
  title       = {Graph-Augmented Metaheuristic Framework for Wrapper Feature Selection},
  author      = {Qinwei Ma and Pengyi Zhang and Kansong Chen},
  journal     = {Applied Soft Computing},
  year        = {2026},
  note        = {Submitted},
  url         = {https://github.com/YourUsername/GAM}
}
```

GitHub automatically generates citation metadata from the included `citation.cff` file, so you can also click the “Cite this repository” button on the GitHub page.

---

## Contact

If you have any questions, suggestions or want to contribute, feel free to contact the lead author:

**Qinwei Ma**  
School of Computer Science, Hubei University  
E‑mail: <your_email@example.com>

We welcome contributions! Please open an issue or pull request on GitHub.

---

## License

This project is licensed under the [MIT license](LICENSE). You are free to use, modify and distribute this software, provided that you include the original copyright notice and license file in your distribution.
