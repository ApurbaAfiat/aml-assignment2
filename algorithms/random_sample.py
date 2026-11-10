"""
Random Sample for NATS-Bench TSS.

Samples architectures upfront without replacement, evaluates them at full
fidelity, and returns the best-so-far result.
"""

import time
from typing import Set
from algorithms.base import BaseNAS
from utils.types import SearchResult


class RandomSample(BaseNAS):
    def run(self) -> SearchResult:
        start_time = time.time()
        total_archs = self.benchmark.num_architectures
        visited: Set[int] = set()

        if self.budget_type == "evaluations":
            sample_size = min(int(self.budget), total_archs)
            sampled_archs = self.rng.choice(
                total_archs,
                size=sample_size,
                replace=False,
            )
        elif self.budget_type == "simulated_time":
            # Evaluation costs vary, so sample an order upfront and stop
            # once the cumulative cost reaches the time budget.
            sampled_archs = self.rng.permutation(total_archs)
        else:
            raise ValueError(f"Unknown budget type: {self.budget_type}")

        for candidate_idx in sampled_archs:
            if self.budget_type == "simulated_time":
                current_cost = (
                    self.cumulative_cost[-1] if self.cumulative_cost else 0.0
                )
                if current_cost >= self.budget:
                    break

            candidate_idx = int(candidate_idx)
            visited.add(candidate_idx)

            # Evaluate at full fidelity (hp=200, final epoch).
            self._evaluate(
                arch_index=candidate_idx,
                hp=200,
                iepoch=None,
            )

        return self._create_search_result(
            algorithm_name="RandomSample",
            start_wall_time=start_time,
            metadata={"num_visited": len(visited)},
        )