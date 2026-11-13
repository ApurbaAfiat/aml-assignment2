# Neural Architecture Search on NATS-Bench

**AutoML — Assignment 2 | Leiden University**  
**Authors**: Apurba Afiat & Marit Kraijenoord

A modular Python framework implementing and comparing five Neural Architecture Search (NAS) algorithms using the [NATS-Bench](https://github.com/D-X-Y/NATS-Bench) tabular benchmark.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How the Components Connect](#how-the-components-connect)
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
aml-assignment2/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_experiments.py                 # Main CLI entry point
│
├── configs/
│   ├── __init__.py
│   └── experiment_config.py           # ExperimentConfig dataclass — central configuration
│
├── benchmark/
│   ├── __init__.py
│   └── nats_api.py                    # BenchmarkAPI wrapper — dataset quirks, caching, cost
│
├── algorithms/
│   ├── __init__.py
│   ├── base.py                        # BaseNAS abstract class — shared interface for all algorithms
│   ├── random_search.py               # Random Search
│   ├── random_sample.py               # Random Sample
│   ├── bayesian_opt.py                # Bayesian Optimisation (GP + EI)
│   ├── successive_halving.py          # Successive Halving (multi-fidelity)
│   └── learning_curve.py             # Learning Curve Extrapolation (power-law)
│
├── utils/
│   ├── __init__.py
│   ├── types.py                       # Shared dataclasses: EvalRecord, SearchResult, BudgetConfig
│   └── arch_utils.py                  # Architecture string parsing and one-hot encoding
│
├── experiments/
│   ├── __init__.py
│   ├── runner.py                      # ExperimentRunner — orchestrates algorithms × datasets × seeds
│   └── results.py                     # CSV save / load utilities
│
├── visualization/
│   ├── __init__.py
│   ├── plots.py                       # Convergence plots (best validation accuracy over steps)
│   ├── comparison.py                  # Algorithm comparison bar charts
│   └── learning_curves.py            # Per-architecture learning curve figures
│
├── tests/
│   ├── __init__.py
│   ├── test_benchmark.py              # 8 tests — BenchmarkAPI wrapper
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
├── figures/                           # Generated plot outputs (git-ignored)
│   └── .gitkeep
│
└── report/                            # LaTeX report files
    ├── main.tex                       # Scientific report source
    ├── results.tex                    # Formatted results table
    └── refs.bib                       # References bibliography
```

---

## How the Components Connect

The diagram below shows the data flow for a single experiment run:

```
ExperimentConfig          ← central config: datasets, budget, seeds,
      │                     and per-algorithm hyperparameters
      ▼
ExperimentRunner          ← loops over algorithms × datasets × seeds
      │
      ├─ creates ──► BenchmarkAPI     ← wraps NATS-Bench, handles quirks & caching
      │                   ▲
      │                   │  queried via _evaluate()
      │
      └─ creates ──► BaseNAS subclass  ← one of the 5 algorithm implementations
                          │
                          │  calls self._evaluate(arch_index, hp, iepoch)
                          │  returns EvalRecord (val_acc, test_acc, train_time)
                          │
                          ▼
                     SearchResult      ← full history + anytime curves
                          │
                          ▼
              ExperimentRunner         ← post-search full-fidelity lookup
                          │            (for multi-fidelity methods only)
                          ▼
                   results/*.csv       ← per-run history + summary.csv
                   figures/*.png       ← convergence plots (via visualization/)
```

**Key rule enforced across all code:** architecture selection always uses `val_accuracy`; `test_accuracy` is only read after the search is complete (for reporting).

---

## Algorithms

All five algorithms inherit from `BaseNAS` (`algorithms/base.py`) and implement a `run()` method. They all use **validation accuracy** for architecture selection and report **test accuracy** only once the search is complete.

| Algorithm | Module | Description |
|-----------|--------|-------------|
| **Random Search** | `algorithms/random_search.py` | Evaluates architectures sequentially without replacement, tracking the best-so-far at every step (anytime-capable). |
| **Random Sample** | `algorithms/random_sample.py` | Samples a fixed-size subset of architectures upfront and evaluates all of them at full fidelity (hp=200). |
| **Bayesian Optimisation** | `algorithms/bayesian_opt.py` | Fits a Gaussian Process surrogate (Matérn 5/2 kernel) to observed (architecture, accuracy) pairs and selects the next candidate via Expected Improvement (EI). Architectures are encoded as 30-dimensional one-hot vectors. |
| **Successive Halving** | `algorithms/successive_halving.py` | Multi-fidelity search with geometric rung schedule (12 → 36 → 108 → 200 epochs, η = 3). Starts with many candidates at low fidelity and promotes only the top 1/η survivors to the next rung. |
| **Learning Curve Extrapolation** | `algorithms/learning_curve.py` | Evaluates architectures epoch-by-epoch. After a few observations, fits a power-law model (Domhan et al., 2015) to predict final accuracy, stopping candidates early if the prediction falls below the best completed run. |

---

## Design Decisions

### Shared `BaseNAS` Interface

Every algorithm inherits from `BaseNAS` (`algorithms/base.py`), which enforces:

- A single `run() → SearchResult` method signature for all algorithms.
- Internal RNG via `np.random.RandomState(seed)` for reproducibility — each algorithm gets an isolated random state.
- A protected `_evaluate(arch_index, hp, iepoch)` helper that queries `BenchmarkAPI`, appends to the evaluation history, updates the anytime incumbent (by validation accuracy), and charges the simulated training cost.
- **Marginal cost tracking** (`_charged_time_per_arch` dict keyed by `(arch_index, hp)`): when a multi-fidelity algorithm revisits the same architecture at a higher epoch, only the *additional* training time since the last visit is charged — preventing double-counting.
- Anytime curves (`best_val_at_step`, `best_test_at_step`, `cumulative_cost`) are recorded after every call to `_evaluate()`, enabling convergence plots independent of algorithm internals.

### `BenchmarkAPI` Wrapper (`benchmark/nats_api.py`)

Abstracts two NATS-Bench quirks that would otherwise scatter workarounds throughout every algorithm:

1. **CIFAR-10 split**: validation accuracy must be queried with `'cifar10-valid'`; test accuracy with `'cifar10'`. Callers always pass `'cifar10'` — the wrapper handles the internal split transparently.
2. **Intermediate-epoch key fallback**: at epochs < 199, CIFAR-100 and ImageNet16-120 expose `'valtest-accuracy'` instead of `'valid-accuracy'`. The wrapper falls back automatically via `.get('valid-accuracy', info.get('valtest-accuracy', 0.0))`.

An in-memory cache (keyed by `(arch_index, dataset, hp, iepoch)`) prevents redundant API calls when the same query is repeated, e.g., by LCE or SH revisiting an architecture.

### Shared Data Types (`utils/types.py`)

Three dataclasses form the common language between all modules:

| Dataclass | Key fields | Purpose |
|-----------|-----------|---------|
| `EvalRecord` | `arch_index`, `val_accuracy`, `test_accuracy`, `train_time`, `eval_number` | One benchmark query result |
| `SearchResult` | `best`, `history`, `best_val_at_step`, `cumulative_cost`, `metadata` | Full output of one algorithm run |
| `BudgetConfig` | `budget_type`, `total_budget` | Budget definition with validation |

### Architecture Encoding (`utils/arch_utils.py`)

Architectures are stored as strings like `|nor_conv_3x3~0|+|avg_pool_3x3~0|nor_conv_1x1~1|+...`. Three utilities are provided:

- `parse_arch(arch_str)` — parses the string into a list of `(src, dst, operation)` tuples for the 6 edges.
- `arch_to_vector(arch_str)` — produces a **30-dimensional one-hot vector** (6 edges × 5 operations) used as input features for the Bayesian Optimisation GP surrogate.
- `vector_to_arch(vec)` — inverse mapping, to decode back to a valid architecture string.

### `ExperimentConfig` (`configs/experiment_config.py`)

Central dataclass that parameterises every aspect of an experiment run:

- **Scope**: which `datasets` and `algorithms` to run, number of `seeds`.
- **Budget**: `budget` value and `budget_type` (`'evaluations'` or `'simulated_time'`).
- **Algorithm hyperparameters**: `bo_n_init` (BO warm-up size), `bo_exploration_weight`, `sh_reduction_factor` (η), `sh_min_epoch`, `sh_max_epoch`, `lce_observed_epochs`, `lce_target_epoch`.
- **Output paths**: `results_dir`, `figures_dir`.

A `__post_init__` validator raises `ValueError` for nonsensical configurations (e.g. negative budget, `bo_n_init ≥ budget`).

### `ExperimentRunner` (`experiments/runner.py`)

Orchestrates the full comparison grid:

1. For each `(algorithm, dataset, seed)` triple, instantiates the correct `BaseNAS` subclass with parameters from `ExperimentConfig`.
2. Calls `algo.run()` to get a `SearchResult`.
3. For multi-fidelity algorithms (SH, LCE) where the winning architecture may have been evaluated at partial fidelity, performs a **full-fidelity post-evaluation** (hp=200) to record an unbiased test accuracy — this does not influence the search outcome.
4. Saves per-run CSVs via `experiments/results.py` and prints a Mean ± Std summary table.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ApurbaAfiat/aml-assignment2.git
cd aml-assignment2
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
# Run specific algorithms on a specific dataset with a simulated-time budget
python run_experiments.py \
  --algorithms random_search bayesian_opt \
  --datasets cifar10 \
  --budget-type simulated_time \
  --budget 200000 \
  --num-seeds 10

# Run all algorithms on all datasets with an evaluation-count budget
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
| `--bo-n-init` | 5 | Random warm-up evaluations before BO starts using the GP |
| `--sh-eta` | 3 | Reduction factor η for Successive Halving |

Results are saved to `results/` as CSV files. Figures are generated separately via `python -m visualization.plots`.

---

## Testing

```bash
# Run the full test suite (33 tests)
python -m pytest tests/ -v

# Run a specific module
python -m pytest tests/test_bayesian_opt.py -v
```

The test suite covers:
- **Unit tests** for each algorithm: budget adherence, anytime monotonicity, seed reproducibility, and algorithm-specific behaviour (e.g. EI correctness for BO, rung promotion for SH, early stopping for LCE).
- **API wrapper tests**: dataset queries, cost lookups, edge-case error handling.
- **End-to-end integration tests**: full pipeline execution, CSV schema validation, and `BaseNAS` contract compliance.

All 33 tests pass.

---

## References

1. Domhan, T., Springenberg, J.T., & Hutter, F. (2015). *Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves.* IJCAI 2015, pp. 3460–3468.

2. Dong, X., Liu, L., Musial, K., & Gabrys, B. (2022). *NATS-Bench: Benchmarking NAS algorithms for architecture topology and size.* IEEE TPAMI, 44(7), 3634–3646.

3. Dong, X., & Yang, Y. (2020). *NAS-Bench-201: Extending the scope of reproducible neural architecture search.* ICLR 2020.

---

*Academic project — AutoML course, Leiden University.*
