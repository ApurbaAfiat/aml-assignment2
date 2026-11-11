"""
Abstract base class for all Neural Architecture Search (NAS) algorithms.

Defines the shared interface and handles evaluation tracking, history logging,
and anytime performance curve recording.
"""

from abc import ABC, abstractmethod
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from benchmark.nats_api import BenchmarkAPI
from utils.types import EvalRecord, SearchResult


class BaseNAS(ABC):
    """Abstract base class that all 5 NAS algorithms must inherit from."""

    def __init__(
        self,
        benchmark: BenchmarkAPI,
        dataset: str = "cifar10",
        seed: int = 42,
        budget: int = 100,
        budget_type: str = "evaluations",
    ):
        """
        Args:
            benchmark: An initialized BenchmarkAPI instance.
            dataset: Target dataset ('cifar10', 'cifar100', 'ImageNet16-120').
            seed: Integer random seed for reproducibility.
            budget: Maximum budget allowed (e.g. 100 evaluations).
            budget_type: 'evaluations' (count of evaluations) or
                         'simulated_time' (accumulated training time in seconds).
        """
        self.benchmark = benchmark
        self.dataset = dataset
        self.seed = seed
        self.budget = budget
        self.budget_type = budget_type

        # Dedicated random state for this algorithm instance
        self.rng = np.random.RandomState(seed)

        # Tracking state
        self.history: List[EvalRecord] = []
        self._eval_counter: int = 0
        self._best_val: float = -1.0
        self._best_record: Optional[EvalRecord] = None

        # Step-by-step curves for anytime performance plotting
        self.best_val_at_step: List[float] = []
        self.best_test_at_step: List[float] = []
        self.cumulative_cost: List[float] = []
        # Track highest cumulative training time already billed per (arch_index, hp)
        self._charged_time_per_arch: Dict[Tuple[int, int], float] = {}

    @abstractmethod
    def run(self) -> SearchResult:
        """Execute the search strategy. Must be implemented by each subclass.

        Returns:
            SearchResult containing the full history and incumbent curves.
        """
        pass

    def _evaluate(
        self,
        arch_index: int,
        hp: int = 200,
        iepoch: Optional[int] = None,
    ) -> EvalRecord:
        """Protected method: Evaluate an architecture via the benchmark.

        Every algorithm MUST call this method to evaluate candidates.
        It automatically updates history, incumbent tracking, and anytime curves.

        Args:
            arch_index: Integer index of architecture in NATS-Bench (0–15624).
            hp: Training schedule (200 for full fidelity, 12 for low fidelity).
            iepoch: Specific epoch (0-indexed). None means final epoch.

        Returns:
            The recorded EvalRecord.
        """
        self._eval_counter += 1

        record = self.benchmark.evaluate(
            arch_index=arch_index,
            dataset=self.dataset,
            hp=hp,
            iepoch=iepoch,
            eval_number=self._eval_counter,
        )

        self.history.append(record)

        # Update incumbent based on VALIDATION accuracy only
        if record.val_accuracy > self._best_val:
            self._best_val = record.val_accuracy
            self._best_record = record

        # Record anytime curves at this evaluation step
        self.best_val_at_step.append(self._best_val)
        self.best_test_at_step.append(self._best_record.test_accuracy)

        # Accumulate marginal simulated training time (avoid double-counting earlier epochs)
        arch_key = (arch_index, hp)
        prev_charged = self._charged_time_per_arch.get(arch_key, 0.0)
        marginal_cost = max(0.0, record.train_time - prev_charged)
        if record.train_time > prev_charged:
            self._charged_time_per_arch[arch_key] = record.train_time

        prev_cost = self.cumulative_cost[-1] if self.cumulative_cost else 0.0
        self.cumulative_cost.append(prev_cost + marginal_cost)

        return record

    def get_best(self) -> EvalRecord:
        """Return the best architecture evaluated so far (by validation accuracy)."""
        if self._best_record is None:
            raise ValueError("No architectures have been evaluated yet.")
        return self._best_record

    def get_history(self) -> List[EvalRecord]:
        """Return the complete evaluation history."""
        return list(self.history)

    def _create_search_result(
        self,
        algorithm_name: str,
        start_wall_time: float,
        metadata: Optional[dict] = None,
    ) -> SearchResult:
        """Helper to construct the final SearchResult object."""
        if self._best_record is None:
            raise ValueError("Cannot create SearchResult: no architectures evaluated.")

        wall_time = time.time() - start_wall_time

        return SearchResult(
            algorithm=algorithm_name,
            dataset=self.dataset,
            seed=self.seed,
            budget=self.budget,
            budget_type=self.budget_type,
            best=self._best_record,
            history=list(self.history),
            wall_time=wall_time,
            best_val_at_step=list(self.best_val_at_step),
            best_test_at_step=list(self.best_test_at_step),
            cumulative_cost=list(self.cumulative_cost),
            metadata=metadata or {},
        )

