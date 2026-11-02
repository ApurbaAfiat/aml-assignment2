"""
Shared data types for the NAS benchmark project.

These dataclasses define the common "language" across all modules:
- EvalRecord:   one architecture evaluation
- SearchResult: the complete output of a NAS run
- BudgetConfig: how computational budget is defined

All algorithms, the experiment runner, and result storage use these types.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class EvalRecord:
    """Record of a single architecture evaluation.

    Created every time an algorithm queries the benchmark.
    Stored in the algorithm's history list in chronological order.

    Attributes:
        arch_index:   Integer index in NATS-Bench (0–15624 for TSS).
        arch_str:     Human-readable architecture string, e.g.
                      '|nor_conv_3x3~0|+|none~0|skip_connect~1|+...'
        dataset:      Dataset used ('cifar10', 'cifar100', 'ImageNet16-120').
        hp:           Training budget in epochs (12 or 200 for TSS).
        iepoch:       Specific epoch queried (None means final epoch of the
                      hp schedule, i.e., epoch hp-1).
        val_accuracy: Validation accuracy in percent.  Used for SELECTION.
        test_accuracy: Test accuracy in percent.  Used for REPORTING only.
                       Never use this to select architectures.
        train_time:   Simulated training time in seconds (from the benchmark).
        eval_number:  Sequential counter — the n-th evaluation in this run.
    """
    arch_index: int
    arch_str: str
    dataset: str
    hp: int
    iepoch: Optional[int]
    val_accuracy: float
    test_accuracy: float
    train_time: float
    eval_number: int


@dataclass
class SearchResult:
    """Complete result of one NAS algorithm run.

    Returned by BaseNAS.run().  The experiment runner collects these
    across seeds and datasets, then serialises them to CSV.

    Attributes:
        algorithm:        Algorithm name (e.g. 'RandomSearch', 'BayesianOpt').
        dataset:          Dataset used.
        seed:             Random seed for this run.
        budget:           Budget that was allocated.
        budget_type:      How budget is measured ('evaluations' or 'simulated_time').
        best:             The best EvalRecord found (by validation accuracy).
        history:          Full evaluation history in chronological order.
        wall_time:        Actual wall-clock time taken (seconds).
        best_val_at_step: Best validation accuracy seen after each evaluation.
                          Length == len(history).  For anytime performance plots.
        best_test_at_step: Test accuracy of the incumbent at each step.
                           Only for reporting — not used for selection.
        cumulative_cost:  Cumulative simulated training time after each eval.
                          For cost-based anytime plots.
        metadata:         Algorithm-specific extra info (e.g. GP hyperparams,
                          bracket sizes, extrapolation errors).
    """
    algorithm: str
    dataset: str
    seed: int
    budget: int
    budget_type: str
    best: EvalRecord
    history: List[EvalRecord]
    wall_time: float
    best_val_at_step: List[float]
    best_test_at_step: List[float]
    cumulative_cost: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetConfig:
    """Configuration for computational budget.

    The budget can be measured in two ways:
    1. 'evaluations' — number of full-fidelity (hp=200) architecture evals.
       Multi-fidelity algorithms (SH, LCE) convert internally.
    2. 'simulated_time' — total simulated training time in seconds.
       Each eval costs train_time from the benchmark.

    Attributes:
        budget_type:  'evaluations' or 'simulated_time'.
        total_budget: The numeric limit.
    """
    budget_type: str  # 'evaluations' or 'simulated_time'
    total_budget: int

    def __post_init__(self):
        valid_types = ('evaluations', 'simulated_time')
        if self.budget_type not in valid_types:
            raise ValueError(
                f"budget_type must be one of {valid_types}, "
                f"got '{self.budget_type}'"
            )
        if self.total_budget <= 0:
            raise ValueError(
                f"total_budget must be positive, got {self.total_budget}"
            )

