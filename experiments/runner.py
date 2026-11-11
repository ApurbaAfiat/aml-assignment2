"""
Experiment runner for orchestrating NAS benchmarks across algorithms, datasets, and seeds.

Ensures fair computational budgets, deterministic random seed handling,
CSV logging via experiments.results, and clean summary reporting.
"""

import os
from typing import List, Optional
import numpy as np

from configs.experiment_config import ExperimentConfig
from benchmark.nats_api import BenchmarkAPI
from algorithms.base import BaseNAS
from algorithms.random_search import RandomSearch
from algorithms.random_sample import RandomSample
from algorithms.bayesian_opt import BayesianOptimization
from algorithms.successive_halving import SuccessiveHalving
from experiments.results import save_run_history, save_summary
from utils.types import SearchResult


class ExperimentRunner:

    def __init__(
        self,
        config: ExperimentConfig,
        benchmark: Optional[BenchmarkAPI] = None,
    ):
        self.config = config
        if benchmark is not None:
            self.benchmark = benchmark
        else:
            self.benchmark = BenchmarkAPI(
                bench_path=config.bench_path,
                search_space=config.search_space,
                fast_mode=config.fast_mode,
                verbose=False,
            )

    def _build_algorithm(self, algo_name: str, dataset: str, seed: int) -> BaseNAS:
        """Instantiate an algorithm subclass by name using parameters from config."""
        cfg = self.config

        if algo_name == "random_search":
            return RandomSearch(
                benchmark=self.benchmark,
                dataset=dataset,
                seed=seed,
                budget=cfg.budget,
                budget_type=cfg.budget_type,
            )
        elif algo_name == "random_sample":
            return RandomSample(
                benchmark=self.benchmark,
                dataset=dataset,
                seed=seed,
                budget=cfg.budget,
                budget_type=cfg.budget_type,
            )
        elif algo_name == "bayesian_opt":
            return BayesianOptimization(
                benchmark=self.benchmark,
                dataset=dataset,
                seed=seed,
                budget=cfg.budget,
                budget_type=cfg.budget_type,
                n_init=cfg.bo_n_init,
                exploration_weight=cfg.bo_exploration_weight,
            )
        elif algo_name == "successive_halving":
            return SuccessiveHalving(
                benchmark=self.benchmark,
                dataset=dataset,
                seed=seed,
                budget=cfg.budget,
                budget_type=cfg.budget_type,
                eta=cfg.sh_reduction_factor,
                min_fidelity=cfg.sh_min_epoch,
                max_fidelity=cfg.sh_max_epoch,
            )
        elif algo_name == "learning_curve":
            # Dynamically import LearningCurveExtrapolation when available
            try:
                from algorithms.learning_curve import LearningCurve
            except ImportError as exc:
                raise NotImplementedError(
                    "LearningCurve is not yet implemented in algorithms/learning_curve.py"
                ) from exc

            return LearningCurve(
                benchmark=self.benchmark,
                dataset=dataset,
                seed=seed,
                budget=cfg.budget,
                budget_type=cfg.budget_type,
            )
        else:
            raise ValueError(f"Unknown algorithm name: '{algo_name}'")

    def run_single(self, algo_name: str, dataset: str, seed: int, save_csv: bool = True) -> SearchResult:
        """Execute a single (algorithm, dataset, seed) trial."""
        algo = self._build_algorithm(algo_name, dataset, seed)
        result = algo.run()

        # Post-search full-fidelity lookup for unbiased reporting (does NOT affect search)
        if result.best.iepoch is not None and result.best.iepoch != (self.config.hp_schedule - 1):
            final_eval = self.benchmark.evaluate(
                arch_index=result.best.arch_index,
                dataset=dataset,
                hp=self.config.hp_schedule,
                iepoch=None,
            )
            result.metadata["full_fidelity_val_accuracy"] = final_eval.val_accuracy
            result.metadata["full_fidelity_test_accuracy"] = final_eval.test_accuracy

        if save_csv:
            save_run_history(result, output_dir=self.config.results_dir)

        return result

    def run_all(self, save_csv: bool = True) -> List[SearchResult]:
        all_results: List[SearchResult] = []
        cfg = self.config

        print("=" * 70)
        print(f"Starting NAS Experiments | Budget: {cfg.budget} ({cfg.budget_type}) | Seeds: {len(cfg.seeds)}")
        print("=" * 70)

        for dataset in cfg.datasets:
            print(f"\n>>> Dataset: {dataset}")
            for algo_name in cfg.algorithms:
                algo_results: List[SearchResult] = []

                for seed in cfg.seeds:
                    try:
                        res = self.run_single(algo_name, dataset, seed, save_csv=save_csv)
                        algo_results.append(res)
                        all_results.append(res)
                    except NotImplementedError as warn_msg:
                        print(f"  [SKIP] {algo_name}: {warn_msg}")
                        break

                # Print aggregated Mean +/- Std for this (dataset, algorithm)
                if algo_results:
                    val_accs = [r.best.val_accuracy for r in algo_results]
                    test_accs = [
                        r.metadata.get("full_fidelity_test_accuracy", r.best.test_accuracy)
                        for r in algo_results
                    ]
                    costs = [r.cumulative_cost[-1] if r.cumulative_cost else 0.0 for r in algo_results]
                    print(
                        f"  {algo_name:<20} | "
                        f"Val: {np.mean(val_accs):6.2f} +/- {np.std(val_accs):4.2f}% | "
                        f"Test: {np.mean(test_accs):6.2f} +/- {np.std(test_accs):4.2f}% | "
                        f"Avg Cost: {np.mean(costs):8.1f}s"
                    )

        if save_csv and all_results:
            summary_path = os.path.join(cfg.results_dir, "summary.csv")
            save_summary(all_results, output_path=summary_path)
            print(f"\nMaster summary saved to: {summary_path}")

        return all_results