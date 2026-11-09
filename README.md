# Neural Architecture Search on NATS-Bench

**AutoML — Assignment 2 | Leiden University**

A modular Python framework implementing and comparing five Neural Architecture Search (NAS) algorithms using the [NATS-Bench](https://github.com/D-X-Y/NATS-Bench) tabular benchmark.

---

## Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Team & Work Division](#team--work-division)
- [Setup](#setup)
- [Usage](#usage)
- [NATS-Bench API Quick Reference](#nats-bench-api-quick-reference)
- [Development Workflow](#development-workflow)
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

## Algorithms

| # | Algorithm | Description | Owner |
|---|-----------|-------------|-------|
| 1 | **Random Search** | Sample architectures one at a time without replacement; track best-so-far (anytime). | Developer A |
| 2 | **Random Sample** | Sample a fixed-size subset upfront; evaluate all; return the best. | Developer B |
| 3 | **Bayesian Optimisation** | GP surrogate with Expected Improvement acquisition; one-hot architecture encoding. | Developer A |
| 4 | **Successive Halving** | Multi-fidelity: start with many candidates at low fidelity, progressively eliminate and increase fidelity. | Developer A |
| 5 | **Learning Curve Extrapolation** | Fit a power-law model to early training epochs to predict final accuracy (Domhan et al., 2015). | Developer B |

All algorithms share a common `BaseNAS` interface, use **validation accuracy** for architecture selection, and report **test accuracy** only after the search is complete.

---

## Project Structure

```
nats-bench-nas/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_experiments.py                 # Main entry point
│
├── configs/
│   ├── __init__.py
│   └── experiment_config.py           # Experiment configuration dataclasses
│
├── benchmark/
│   ├── __init__.py
│   └── nats_api.py                    # NATS-Bench API wrapper
│
├── algorithms/
│   ├── __init__.py
│   ├── base.py                        # BaseNAS abstract class (shared interface)
│   ├── random_search.py               # Random Search
│   ├── random_sample.py               # Random Sample              ★ Dev B
│   ├── bayesian_opt.py                # Bayesian Optimisation
│   ├── successive_halving.py          # Successive Halving
│   └── learning_curve.py             # Learning Curve Extrapolation ★ Dev B
│
├── utils/
│   ├── __init__.py
│   ├── types.py                       # Shared dataclasses (EvalRecord, SearchResult)
│   ├── arch_utils.py                  # Architecture encoding/decoding
│   └── budget.py                      # Budget tracking
│
├── experiments/
│   ├── __init__.py
│   ├── runner.py                      # ExperimentRunner
│   └── results.py                     # CSV save/load utilities
│
├── visualization/                     # ★ Dev B
│   ├── __init__.py
│   ├── plots.py                       # Convergence plots
│   ├── comparison.py                  # Algorithm comparison charts
│   └── learning_curves.py            # Learning curve visualisation
│
├── tests/
│   ├── __init__.py
│   ├── test_benchmark.py              # Tests for API wrapper
│   ├── test_random_search.py          # Tests for Random Search
│   ├── test_bayesian_opt.py           # Tests for BO
│   ├── test_successive_halving.py     # Tests for SH
│   ├── test_random_sample.py          # Tests for Random Sample    ★ Dev B
│   ├── test_learning_curve.py         # Tests for LCE              ★ Dev B
│   └── test_integration.py           # End-to-end integration tests
│
├── results/                           # Generated CSV outputs
│   └── .gitkeep
│
├── figures/                           # Generated plot outputs
│   └── .gitkeep
│
└── report/                            # ★ Dev B — LaTeX report
    ├── report.tex
    ├── references.bib
    └── figures/
```

> Files marked with **★** are owned by Developer B. All other code files are owned by Developer A.

---

## Team & Work Division

### Developer A — Apurba (70% coding)

| Responsibility | Details |
|---------------|---------|
| **Project scaffolding** | Directory structure, `requirements.txt`, `README.md`, configuration |
| **NATS-Bench integration** | API wrapper with CIFAR-10 quirk handling, caching, error handling |
| **Common interfaces** | `BaseNAS` abstract class, `EvalRecord`/`SearchResult` dataclasses |
| **Architecture utilities** | Encoding/decoding arch strings, one-hot feature vectors for BO |
| **Random Search** | Full implementation and unit tests |
| **Bayesian Optimisation** | GP surrogate, EI acquisition function, full implementation and tests |
| **Successive Halving** | Multi-fidelity bracket logic, candidate elimination, full implementation and tests |
| **Experiment framework** | `ExperimentRunner`, budget management, CSV result storage |
| **Integration & debugging** | Wiring all algorithms, final debugging, reproducibility |

### Developer B — Marit (30% coding + report)

| Responsibility | Details |
|---------------|---------|
| **Random Sample** | Full implementation and unit tests |
| **Learning Curve Extrapolation** | Power-law fitting, early-stopping prediction, full implementation and tests |
| **Visualisation** | Convergence plots, algorithm comparison charts, learning curve figures |
| **Validation** | Verifying implementations against the NATS-Bench API |
| **Scientific report** | Full 4-page report: methodology, experimental setup, results, discussion, references |

### Integration Contract

Both developers follow these rules to ensure smooth integration:

1. **All algorithms extend `BaseNAS`** and implement the `run()` method.
2. **Architecture IDs** are always integer indices (0–15,624).
3. **Dataset names** are passed as `'cifar10'`, `'cifar100'`, or `'ImageNet16-120'` — the wrapper handles `'cifar10-valid'` internally.
4. **Budget** is expressed as number of full-fidelity evaluations (hp=200).
5. **Random seeds** are passed to the constructor; algorithms use `np.random.RandomState(seed)`.
6. **Evaluations** go through `self._evaluate()` (never call `api.get_more_info()` directly).
7. **Selection** uses `val_accuracy`; `test_accuracy` is for reporting only.
8. **`run()` returns a `SearchResult`** dataclass with full history and anytime curves.

---

## Implementation Progress

### Completed by Apurba (Lead Developer — 70%)
- [x] **Project Scaffolding**: Modular directory structure, `.gitignore`, and `requirements.txt`.
- [x] **Benchmark Setup**: Verified fast-mode data loading across all 15,625 architectures on CIFAR-10, CIFAR-100, and ImageNet16-120.
- [x] **Standard Data Types (`utils/types.py`)**: `EvalRecord`, `SearchResult`, and `BudgetConfig` dataclasses defining the contract between algorithms and evaluation runners.
- [x] **Benchmark API Wrapper (`benchmark/nats_api.py`)**: Encapsulates dataset quirks (such as the `cifar10-valid` vs `cifar10` split), key variation handling, in-memory query caching, and metric extraction.
- [x] **Benchmark Unit Tests (`tests/test_benchmark.py`)**: 8 comprehensive test cases passing for dataset queries, cost lookups, and boundary error handling.
- [x] **Abstract Base NAS Class (`algorithms/base.py`)**: `BaseNAS` abstract class enforcing the `run()` contract, managing private RNG states (`np.random.RandomState`), sequential evaluation counters, validation-based incumbent tracking, and anytime curve generation.
- [x] **Master Experiment Config (`configs/experiment_config.py`)**: Centralized `ExperimentConfig` dataclass parameterizing datasets, budgets, seeds, and algorithm-specific parameters with automated sanity checks.
- [x] **Architecture Utilities (`utils/arch_utils.py`)**: Graph parsing, canonical edge ordering, and 30-dimensional one-hot encoding/decoding for surrogate models with verified round-trip preservation.
- [x] **Random Search (`algorithms/random_search.py`)**: Full implementation of the baseline search algorithm with sampling without replacement, dual budget support (evaluations & simulated time), and incumbent tracking.

### Next Stages
- [ ] **Random Search Unit Tests (`tests/test_random_search.py`)** *(Apurba)*
- [ ] **Random Sample (`algorithms/random_sample.py`)** *(Marit)*
- [ ] **Bayesian Optimisation (`algorithms/bayesian_opt.py`)** *(Apurba)*
- [ ] **Successive Halving (`algorithms/successive_halving.py`)** *(Apurba)*
- [ ] **Learning Curve Extrapolation (`algorithms/learning_curve.py`)** *(Marit)*
- [ ] **Experiment Runner & Evaluation Pipeline (`experiments/runner.py`)** *(Apurba)*
- [ ] **Visualisation & Scientific Report (`visualization/`, `report/`)** *(Marit)*

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

Download the TSS benchmark for your setup:

| Mode | File | Use when |
|------|------|----------|
| **Fast mode** (laptop) | [`NATS-tss-v1_0-3ffb9-simple.tar`](https://drive.google.com/file/d/17_saCsj_krKjlCBLOJEpNtzPXArMCqxU/view?usp=sharing) | Limited RAM; loads each arch on demand |
| **Full mode** (lab PC) | [`NATS-tss-v1_0-3ffb9.pickle.pbz2`](https://drive.google.com/file/d/1vzyK0UVH2D3fTpa1_dSWnp1gvGpAxRul/view?usp=sharing) | Plenty of RAM; loads everything at startup |

**For fast mode** (recommended):
```bash
# Extract the tar file into the project root
tar xvf NATS-tss-v1_0-3ffb9-simple.tar
# This creates: NATS-tss-v1_0-3ffb9-simple/
```

Alternatively, set the environment variable:
```bash
export NATS_BENCH_PATH=/path/to/NATS-tss-v1_0-3ffb9-simple
```

### 5. Verify installation

```bash
python -c "
from nats_bench import create
api = create('NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)
print(f'Loaded {len(api)} architectures')
print(f'Architecture 0: {api.arch(0)}')
"
```

Expected output:
```
Loaded 15625 architectures
Architecture 0: |none~0|+|none~0|none~1|+|none~0|none~1|none~2|
```

---

## Usage

### Run all experiments

```bash
python run_experiments.py
```

### Run a specific algorithm on a specific dataset

```bash
python run_experiments.py --algorithm random_search --dataset cifar10 --budget 100 --seeds 10
```

### Run tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_random_search.py -v
```

### Generate figures

```bash
python -m visualization.plots
python -m visualization.comparison
```

Results are saved to `results/` (CSV) and `figures/` (PNG).

---

## NATS-Bench API Quick Reference

```python
from nats_bench import create

# Load the benchmark
api = create('NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)

# Architecture lookups
arch_str = api.arch(0)                          # index → string
idx = api.query_index_by_arch(arch_str)         # string → index
total = len(api)                                 # 15625 for TSS

# Performance query
info = api.get_more_info(idx, 'cifar10-valid', hp=200, is_random=False)
# Keys: 'train-accuracy', 'valid-accuracy', 'train-loss', 'valid-loss',
#        'train-per-time', 'train-all-time', 'valid-per-time', 'valid-all-time'

# Test accuracy (for reporting only)
test_info = api.get_more_info(idx, 'cifar10', hp=200, is_random=False)
# Keys: 'test-accuracy', 'test-loss', ...

# Cost info
cost = api.get_cost_info(idx, 'cifar10', hp=200)
# Keys: 'flops', 'params', 'latency', 'train_time' (per-epoch)

# Learning curve (per-epoch)
for epoch in range(200):
    info = api.get_more_info(idx, 'cifar10-valid', hp=200, iepoch=epoch, is_random=False)
    val_acc = info['valid-accuracy']
```

> **Important**: For CIFAR-10, use `'cifar10-valid'` for validation accuracy and `'cifar10'` for test accuracy. For CIFAR-100 and ImageNet16-120, the same dataset name returns both.

---

## Development Workflow

### Branch Strategy

```
main                         ← stable, tested code
├── dev/setup                ← project scaffold and shared interfaces
├── dev/random-search        ← Dev A: Random Search
├── dev/bayesian-opt         ← Dev A: Bayesian Optimisation
├── dev/successive-halving   ← Dev A: Successive Halving
├── dev/experiment-runner    ← Dev A: experiment framework
├── dev/random-sample        ← Dev B: Random Sample
├── dev/learning-curve       ← Dev B: Learning Curve Extrapolation
├── dev/visualization        ← Dev B: visualisation code
└── dev/integration          ← final integration
```

### For Developer A

```bash
git checkout -b dev/random-search
# ... implement ...
git add algorithms/random_search.py tests/test_random_search.py
git commit -m "feat: implement Random Search algorithm"
git push origin dev/random-search
# Create Pull Request → merge to main
```

### For Developer B

```bash
git checkout main && git pull origin main
git checkout -b dev/random-sample
# ... implement using the shared BaseNAS interface ...
git add algorithms/random_sample.py tests/test_random_sample.py
git commit -m "feat: implement Random Sample algorithm"
git push origin dev/random-sample
# Create Pull Request → Dev A reviews and merges
```

### Rules to Avoid Conflicts

1. Each developer edits **only** their own files (see ★ markers in project structure).
2. Shared interface files (`base.py`, `types.py`) are edited only by Developer A.
3. Developer B proposes interface changes via discussion, not direct edits.
4. Always pull `main` before creating a new branch.
5. Run `pytest tests/` before creating a Pull Request.

---

## Experimental Design

### Budget Fairness

All algorithms receive the **same computational budget** (number of full-fidelity evaluations or equivalent simulated training time) for fair comparison.

### Evaluation Protocol

- **Selection**: Based on **validation accuracy** only.
- **Reporting**: **Test accuracy** of the final selected architecture.
- **Seeds**: Each algorithm runs with **multiple random seeds** (default: 10). Results report mean ± standard deviation.
- **Metrics**: Anytime validation/test curves, final best accuracy, cumulative cost.

### Output Files

| File | Description |
|------|-------------|
| `results/<algorithm>_<dataset>_<seed>.csv` | Per-evaluation history |
| `results/summary.csv` | Best architecture per algorithm/dataset/seed |
| `figures/convergence_<dataset>.png` | Anytime performance curves |
| `figures/comparison.png` | Algorithm comparison chart |

---

## References

1. Domhan, T., Springenberg, J.T., & Hutter, F. (2015). *Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves.* IJCAI 2015, pp. 3460–3468.

2. Dong, X., Liu, L., Musial, K., & Gabrys, B. (2022). *NATS-Bench: Benchmarking NAS algorithms for architecture topology and size.* IEEE TPAMI, 44(7), 3634–3646.

3. Dong, X., & Yang, Y. (2020). *NAS-Bench-201: Extending the scope of reproducible neural architecture search.* ICLR 2020.

---

## License

This project is an academic assignment for the AutoML course at Leiden University.

