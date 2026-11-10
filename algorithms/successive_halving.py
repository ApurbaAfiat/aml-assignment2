"""
Successive Halving (SH) algorithm for NATS-Bench TSS.

A multi-fidelity optimization algorithm that starts with a large pool of
candidate architectures evaluated at low fidelity (few epochs), progressively
prunes the bottom fraction, and promotes the top survivors to higher fidelities.
"""

import time
from typing import List, Tuple
import numpy as np

from algorithms.base import BaseNAS
from utils.types import SearchResult


class SuccessiveHalving(BaseNAS):
    def __init__(
        self,
        benchmark,
        dataset: str = "cifar10",
        seed: int = 42,
        budget: int = 100,
        budget_type: str = "evaluations",
        eta: int = 3,
        min_fidelity: int = 8,
        max_fidelity: int = 200,
    ):
        
        super().__init__(benchmark, dataset, seed, budget, budget_type)
        self.eta = eta
        self.min_fidelity = min_fidelity
        self.max_fidelity = max_fidelity
        rungs = []
        curr = float(min_fidelity)
        while curr < max_fidelity:
            rungs.append(int(round(curr)))
            curr *= eta
        if not rungs or rungs[-1] < max_fidelity:
            rungs.append(max_fidelity)
        self.rungs: List[int] = rungs
        self.num_rungs: int = len(self.rungs)

    def _compute_initial_pool_size(self) -> int:
        if self.budget_type == "evaluations":
            geom_sum = sum(1.0 / (self.eta ** k) for k in range(self.num_rungs))
            n0 = int(round(self.budget / geom_sum))
            min_n0 = self.eta ** (self.num_rungs - 1)
            max_n0 = self.benchmark.num_architectures
            return max(min_n0, min(n0, max_n0))
        else:
            return max(self.eta ** (self.num_rungs - 1), 27)

    def run(self) -> SearchResult:
        
        start_time = time.time()
        total_archs = self.benchmark.num_architectures
        n0 = self._compute_initial_pool_size()

        # Step 1: Sample initial pool of N0 unique candidate architectures
        sampled_indices = self.rng.choice(total_archs, size=n0, replace=False).tolist()
        current_candidates: List[int] = sampled_indices

        budget_exhausted = False
        rung_records_metadata = []

        # Step 2: Iterate through fidelity rungs
        for rung_idx, fidelity_epochs in enumerate(self.rungs):
            if budget_exhausted:
                break

            rung_results: List[Tuple[int, float]] = []
            iepoch = fidelity_epochs - 1

            for arch_idx in current_candidates:
                if self.budget_type == "evaluations":
                    if len(self.history) >= self.budget:
                        budget_exhausted = True
                        break
                elif self.budget_type == "simulated_time":
                    current_cost = self.cumulative_cost[-1] if self.cumulative_cost else 0.0
                    if current_cost >= self.budget:
                        budget_exhausted = True
                        break

                record = self._evaluate(
                    arch_index=arch_idx,
                    hp=200,
                    iepoch=iepoch,
                )
                rung_results.append((arch_idx, record.val_accuracy))

            rung_records_metadata.append({
                "rung": rung_idx,
                "fidelity_epochs": fidelity_epochs,
                "candidates_evaluated": len(rung_results),
            })

            # Step 3: Halving / Selection for next rung
            if rung_idx < self.num_rungs - 1 and not budget_exhausted:
                # Sort descending by validation accuracy
                rung_results.sort(key=lambda item: item[1], reverse=True)
                # Keep top 1 / eta survivors
                survivor_count = max(1, len(rung_results) // self.eta)
                current_candidates = [item[0] for item in rung_results[:survivor_count]]

        return self._create_search_result(
            algorithm_name="SuccessiveHalving",
            start_wall_time=start_time,
            metadata={
                "rungs": self.rungs,
                "eta": self.eta,
                "n0": n0,
                "rung_details": rung_records_metadata,
            },
        )