"""
Bayesian Optimisation (BO) for Neural Architecture Search on NATS-Bench TSS.

Uses:
- 30-dimensional one-hot architecture encoding as feature representation.
- Gaussian Process (GP) regressor with a Matern 5/2 kernel as surrogate model.
- Expected Improvement (EI) as acquisition function to balance exploration and exploitation.
"""

import time
import warnings
from typing import Set, List
import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from algorithms.base import BaseNAS
from utils.types import SearchResult
from utils.arch_utils import arch_to_vector


class BayesianOptimization(BaseNAS):

    def __init__(
        self,
        benchmark,
        dataset: str = "cifar10",
        seed: int = 42,
        budget: int = 100,
        budget_type: str = "evaluations",
        n_init: int = 10,
        exploration_weight: float = 0.01,
        candidate_pool_size: int = 200,
    ):
        super().__init__(benchmark, dataset, seed, budget, budget_type)
        self.n_init = min(n_init, budget)
        self.xi = exploration_weight
        self.candidate_pool_size = candidate_pool_size

    def _expected_improvement(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        best_val: float,
    ) -> np.ndarray:
        # Ensure std is strictly positive to prevent division by zero
        std = np.maximum(std, 1e-9)
        improvement = mean - best_val - self.xi
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        # Any zero/negative improvement gets clamped to 0
        return np.maximum(ei, 0.0)

    def run(self) -> SearchResult:
        start_time = time.time()
        total_archs = self.benchmark.num_architectures
        visited: Set[int] = set()

        # Cache of encoded feature vectors for evaluated architectures
        features_list: List[np.ndarray] = []
        targets_list: List[float] = []

        # ==========================================
        # Phase 1: Warm-up with Random Evaluations
        # ==========================================
        while len(visited) < self.n_init:
            if self.budget_type == "simulated_time":
                current_cost = self.cumulative_cost[-1] if self.cumulative_cost else 0.0
                if current_cost >= self.budget:
                    break

            candidate_idx = int(self.rng.randint(0, total_archs))
            while candidate_idx in visited:
                candidate_idx = int(self.rng.randint(0, total_archs))

            visited.add(candidate_idx)
            record = self._evaluate(candidate_idx, hp=200, iepoch=None)

            # Record feature vector and validation target
            x_vec = arch_to_vector(record.arch_str)
            features_list.append(x_vec)
            targets_list.append(record.val_accuracy)

        # ==========================================
        # Phase 2: Bayesian Optimization Loop
        # ==========================================
        while True:
            # Check budget exhaustion
            if self.budget_type == "evaluations":
                if len(visited) >= self.budget:
                    break
            elif self.budget_type == "simulated_time":
                current_cost = self.cumulative_cost[-1] if self.cumulative_cost else 0.0
                if current_cost >= self.budget:
                    break

            if len(visited) >= total_archs:
                break

            # 1. Fit Gaussian Process Surrogate Model
            X_train = np.array(features_list, dtype=np.float32)
            y_train = np.array(targets_list, dtype=np.float32)

            # Matern 5/2 kernel with fixed numerical nugget (alpha) for deterministic tabular evaluations
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-4,
                normalize_y=True,
                n_restarts_optimizer=1,
                random_state=self.seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(X_train, y_train)

            # 2. Form candidate pool of unvisited architectures
            unvisited_candidates: List[int] = []
            max_attempts = self.candidate_pool_size * 5
            attempts = 0

            while len(unvisited_candidates) < self.candidate_pool_size and attempts < max_attempts:
                cand = int(self.rng.randint(0, total_archs))
                if cand not in visited and cand not in unvisited_candidates:
                    unvisited_candidates.append(cand)
                attempts += 1

            if not unvisited_candidates:
                break

            # 3. Encode candidate architectures into feature vectors
            cand_features = np.array(
                [arch_to_vector(self.benchmark.arch_str(c)) for c in unvisited_candidates],
                dtype=np.float32,
            )

            # 4. Predict mean and uncertainty using GP
            mu, sigma = gp.predict(cand_features, return_std=True)

            # 5. Compute Expected Improvement for all candidates
            current_best_val = self._best_val
            ei = self._expected_improvement(mu, sigma, current_best_val)

            # 6. Select candidate with maximum Expected Improvement
            best_cand_pos = int(np.argmax(ei))
            selected_idx = unvisited_candidates[best_cand_pos]

            visited.add(selected_idx)
            record = self._evaluate(selected_idx, hp=200, iepoch=None)

            # Append to training set for next iteration
            features_list.append(cand_features[best_cand_pos])
            targets_list.append(record.val_accuracy)

        return self._create_search_result(
            algorithm_name="BayesianOptimization",
            start_wall_time=start_time,
            metadata={
                "n_init": self.n_init,
                "xi": self.xi,
                "num_visited": len(visited),
            },
        )