"""
Random Search (RS) with sampling without replacement for NATS-Bench TSS.

Sequentially samples random architectures, evaluates them at full fidelity (hp=200),
tracks the best-so-far incumbent on validation accuracy, and stops when the budget is reached.
"""

import time
from typing import Set
from algorithms.base import BaseNAS
from utils.types import SearchResult


class RandomSearch(BaseNAS):
    def run(self) -> SearchResult:
        start_time = time.time()
        total_archs = self.benchmark.num_architectures
        visited: Set[int] = set()
        while True:
            if self.budget_type == "evaluations":
                if len(visited) >= self.budget:
                    break
            elif self.budget_type == "simulated_time":
                current_cost = self.cumulative_cost[-1] if self.cumulative_cost else 0.0
                if current_cost >= self.budget:
                    break
            if len(visited) >= total_archs:
                break
            candidate_idx = int(self.rng.randint(0, total_archs))
            while candidate_idx in visited:
                candidate_idx = int(self.rng.randint(0, total_archs))

            visited.add(candidate_idx)

            # Evaluate at full fidelity (hp=200, final epoch)
            self._evaluate(arch_index=candidate_idx, hp=200, iepoch=None)

        return self._create_search_result(
            algorithm_name="RandomSearch",
            start_wall_time=start_time,
            metadata={"num_visited": len(visited)},
        )