# Neural Architecture Search on NATS-Bench

**AutoML — Assignment 2 | Leiden University**  
**Authors**: Apurba Afiat & Marit Kraijenoord

A modular Python framework implementing and comparing five Neural Architecture Search (NAS) algorithms using the [NATS-Bench](https://github.com/D-X-Y/NATS-Bench) tabular benchmark.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Algorithms](#algorithms)
- [Design Decisions](#design-decisions)
- [Setup](#setup)
- [Usage](#usage)
- [Testing](#testing)
- [References](#references)

---

## Overview

NATS-Bench is a **tabular NAS benchmark**: every architecture in the search space has already been trained, so evaluation is a dictionary lookup rather than an actual training run. This project uses the **Topology Search Space (TSS)** containing **15,625** cell-based architectures, each evaluated on three datasets.

### Datasets

| Dataset | Classes | Image Size |
|---------|---------|------------|
| CIFAR-10 | 10 | 32×32 |
| CIFAR-100 | 100 | 32×32 |
| ImageNet16-120 | 120 | 16×16 |

### Search Space

Each architecture is a **4-node directed acyclic graph (DAG)** cell with 6 edges. Each edge carries one of 5 operations:

| Operation | Description |
|-----------|-------------|
| `none` | No connection |
| `skip_connect` | Identity / skip connection |
| `nor_conv_1x1` | 1×1 convolution |
| `nor_conv_3x3` | 3×3 convolution |
| `avg_pool_3x3` | 3×3 average pooling |

Total architectures: 5⁶ = 15,625.

---

## Project Structure

```
nats-bench-nas/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_experiments.py                 # Main CLI entry point
│
├── configs/
│   ├── __init__.py
│   └── experiment_config.py           # ExperimentConfig dataclass (datasets, budget, seeds)
│
├── benchmark/
│   ├── __init__.py
│   └── nats_api.py                    # BenchmarkAPI wrapper (caching, dataset quirks, cost)
│
├── algorithms/
│   ├── __init__.py
│   ├── base.py                        # BaseNAS abstract class (shared interface)
│   ├── random_search.py               # Random Search
│   ├── random_sample.py               # Random Sample
│   ├── bayesian_opt.py                # Bayesian Optimisation (GP + EI)
│   ├── successive_halving.py          # Successive Halving (multi-fidelity)
│   └── learning_curve.py             # Learning Curve Extrapolation
│
├── utils/
│   ├── __init__.py
│   ├── types.py                       # Shared dataclasses: EvalRecord, SearchResult, BudgetConfig
│   └── arch_utils.py                  # Architecture encoding / decoding utilities
│
├── experiments/
│   ├── __init__.py
│   ├── runner.py                      # ExperimentRunner: orchestrates all algorithms × datasets × seeds
│   └── results.py                     # CSV save / load utilities
│
├── visualization/
│   ├── __init__.py
│   ├── plots.py                       # Convergence plots
│   ├── comparison.py                  # Algorithm comparison charts
│   └── learning_curves.py            # Learning curve visualisation
│
├── tests/
│   ├── __init__.py
│   ├── test_benchmark.py              # 8 tests — API wrapper
│   ├── test_random_search.py          # 4 tests — Random Search
│   ├── test_bayesian_opt.py           # 4 tests — Bayesian Optimisation
│   ├── test_successive_halving.py     # 4 tests — Successive Halving
│   ├── test_random_sample.py          # 4 tests — Random Sample
│   ├── test_learning_curve.py         # 6 tests — Learning Curve Extrapolation
│   └── test_integration.py           # 3 end-to-end integration tests
│
├── results/                           # Generated CSV outputs (git-ignored)
│   └── .gitkeep
│
└── figures/                           # Generated plot outputs (git-ignored)
    └── .gitkeep
```

---

## Algorithms

All five algorithms extend `BaseNAS` (`algorithms/base.py`) and implement a `run()` method. They use **validation accuracy** for architecture selection and report **test accuracy** only after the search is complete.

| Algorithm | Module | Description |
|-----------|--------|-------------|
| **Random Search** | `algorithms/random_search.py` | Evaluates architectures sequentially without replacement, tracking the best-so-far at every step (anytime). |
| **Random Sample** | `algorithms/random_sample.py` | Samples a fixed-size subset of architectures upfront and evaluates all of them at full fidelity. |
| **Bayesian Optimisation** | `algorithms/bayesian_opt.py` | Fits a Gaussian Process surrogate (Matérn 5/2 kernel) to observed (architecture, accuracy) pairs and uses Expected Improvement (EI) to select the next candidate. Architectures are encoded as 30-dimensional one-hot vectors. |
| **Successive Halving** | `algorithms/successive_halving.py` | Multi-fidelity search with geometric rung schedule (8 → 24 → 72 → 200 epochs, η = 3). Starts with many candidates at low fidelity and promotes only the top 1/η survivors to the next rung. |
| **Learning Curve Extrapolation** | `algorithms/learning_curve.py` | Evaluates architectures epoch-by-epoch. After a few observations it fits a power-law model (Domhan et al., 2015) to predict final accuracy, stopping candidates early if the prediction falls below the best completed run. |

---

## Design Decisions

### Shared `BaseNAS` Interface

Every algorithm inherits from `BaseNAS` (`algorithms/base.py`), which enforces:

- A single `run(api, dataset, budget_cfg, seed) → SearchResult` method signature.
- Internal RNG via `np.random.RandomState(seed)` for reproducibility.
- A `_evaluate(arch_index, dataset, hp)` helper that queries `BenchmarkAPI`, updates the anytime incumbent, and charges the budget.
- Marginal cost tracking (`_charged_time_per_arch`) so multi-fidelity algorithms are not double-charged for earlier epochs.

### `BenchmarkAPI` Wrapper

`benchmark/nats_api.py` abstracts two NATS-Bench quirks that would otherwise scatter workarounds throughout the algorithm code:

1. **CIFAR-10 split**: validation accuracy must be queried with `'cifar10-valid'`; test accuracy with `'cifar10'`. Callers always pass `'cifar10'` — the wrapper handles the internal split.
2. **Intermediate-epoch keys**: at epochs < 199, CIFAR-100 and ImageNet16-120 expose `'valtest-accuracy'` instead of `'valid-accuracy'`. The wrapper falls back automatically.

An in-memory cache prevents redundant API calls when the same `(arch_index, dataset, hp, iepoch)` tuple is queried more than once.

### Shared Data Types (`utils/types.py`)

| Dataclass | Purpose |
|-----------|---------|
| `EvalRecord` | A single evaluation result: arch index, val accuracy, test accuracy, simulated cost. |
| `SearchResult` | Full search output: entire history of `EvalRecord`s, best arch index/string/accuracy, total cost. |
| `BudgetConfig` | Budget type (`evaluations` or `simulated_time`) and limit value. |

### Architecture Encoding (`utils/arch_utils.py`)

Architectures are stored as strings like `|nor_conv_3x3~0|+|avg_pool_3x3~0|nor_conv_1x1~1|+...`. The utilities provide:

- `parse_arch(arch_str)` — parses the string into a list of (operation, input_node) tuples.
- `arch_to_vector(arch_str)` — 30-dimensional one-hot encoding (6 edges × 5 operations) used by the BO surrogate.
- `vector_to_arch(vec)` — inverse, for decoding surrogate outputs.

### Experiment Runner (`experiments/runner.py`)

`ExperimentRunner` orchestrates the full comparison grid (algorithms × datasets × seeds). After each algorithm returns its `SearchResult`, the runner performs a **full-fidelity post-evaluation** of the selected architecture to record its definitive test accuracy. Results are saved as:

- Per-run CSV: `results/<algorithm>_<dataset>_seed<n>.csv`
- Summary CSV: `results/summary.csv` (one row per algorithm/dataset/seed)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ApurbaAfiat/aml-assignment2.git
cd aml-assignment2/nats-bench-nas
```

### 2. Create a Python environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NATS-Bench benchmark data

| Mode | File | Use when |
|------|------|----------|
| **Fast mode** (recommended) | [`NATS-tss-v1_0-3ffb9-simple.tar`](https://drive.google.com/file/d/17_saCsj_krKjlCBLOJEpNtzPXArMCqxU/view?usp=sharing) | Limited RAM; loads each arch on demand |
| **Full mode** | [`NATS-tss-v1_0-3ffb9.pickle.pbz2`](https://drive.google.com/file/d/1vzyK0UVH2D3fTpa1_dSWnp1gvGpAxRul/view?usp=sharing) | Plenty of RAM; loads everything at startup |

```bash
# Extract fast-mode data into the project root
tar xvf NATS-tss-v1_0-3ffb9-simple.tar
# Creates: NATS-tss-v1_0-3ffb9-simple/
```

---

## Usage

### Run all experiments (default settings)

```bash
python run_experiments.py
```

### Common options

```bash
# Specific algorithms and dataset, 10 seeds, simulated-time budget
python run_experiments.py \
  --algorithms random_search bayesian_opt \
  --datasets cifar10 \
  --budget-type simulated_time \
  --budget 200000 \
  --num-seeds 10

# Evaluation-count budget on all three datasets
python run_experiments.py \
  --datasets cifar10 cifar100 ImageNet16-120 \
  --budget-type evaluations \
  --budget 100 \
  --num-seeds 5
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--datasets` | all 3 | `cifar10`, `cifar100`, `ImageNet16-120` |
| `--algorithms` | all 5 | `random_search`, `random_sample`, `bayesian_opt`, `successive_halving`, `learning_curve` |
| `--budget` | 100 | Budget limit (units depend on `--budget-type`) |
| `--budget-type` | `evaluations` | `evaluations` or `simulated_time` |
| `--num-seeds` | 10 | Number of random seeds (uses seeds 0 … n−1) |
| `--results-dir` | `results/` | Output directory for CSV files |
| `--bo-n-init` | 5 | Random warm-up evaluations for Bayesian Optimisation |
| `--sh-eta` | 3 | Reduction factor η for Successive Halving |

Results are saved to `results/` (CSV). Figures can be generated separately via `python -m visualization.plots`.

---

## Testing

```bash
# Run the full test suite (33 tests)
python -m pytest tests/ -v

# Run a specific module
python -m pytest tests/test_bayesian_opt.py -v
```

All 33 tests pass across unit tests (benchmark wrapper, all 5 algorithms) and end-to-end integration tests.

---

## References

1. Domhan, T., Springenberg, J.T., & Hutter, F. (2015). *Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves.* IJCAI 2015, pp. 3460–3468.

2. Dong, X., Liu, L., Musial, K., & Gabrys, B. (2022). *NATS-Bench: Benchmarking NAS algorithms for architecture topology and size.* IEEE TPAMI, 44(7), 3634–3646.

3. Dong, X., & Yang, Y. (2020). *NAS-Bench-201: Extending the scope of reproducible neural architecture search.* ICLR 2020.

---

*Academic project — AutoML course, Leiden University.*
