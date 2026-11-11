"""
End-to-end integration tests for the NATS-Bench NAS pipeline.

Verifies that ExperimentConfig, BenchmarkAPI, all BaseNAS algorithms,
ExperimentRunner, and CSV result serialization integrate seamlessly.
"""

import os
import pytest

from configs.experiment_config import ExperimentConfig
from benchmark.nats_api import BenchmarkAPI
from experiments.runner import ExperimentRunner
from experiments.results import load_summary, load_run_history


@pytest.fixture(scope="module")
def api():
    """Shared BenchmarkAPI instance across integration tests."""
    return BenchmarkAPI(bench_path="NATS-tss-v1_0-3ffb9-simple", fast_mode=True, verbose=False)


def test_end_to_end_pipeline(api, tmp_path):
    """Test running multiple algorithms across datasets and verifying CSV outputs."""
    algos = ["random_search", "random_sample", "bayesian_opt", "successive_halving"]
    datasets = ["cifar10", "cifar100"]
    seeds = [0, 1]

    config = ExperimentConfig(
        datasets=datasets,
        algorithms=algos,
        budget=8,
        bo_n_init=3,
        seeds=seeds,
        num_seeds=len(seeds),
        results_dir=str(tmp_path),
    )

    runner = ExperimentRunner(config, benchmark=api)
    results = runner.run_all(save_csv=True)

    # Expect 2 datasets * 4 algorithms * 2 seeds = 16 runs
    assert len(results) == len(datasets) * len(algos) * len(seeds)

    # Check master summary.csv exists and has all 16 rows
    summary_path = os.path.join(str(tmp_path), "summary.csv")
    assert os.path.exists(summary_path)

    summary_rows = load_summary(summary_path)
    assert len(summary_rows) == len(results)

    # Verify every per-run history file exists and loads cleanly
    for res in results:
        hist_file = os.path.join(
            str(tmp_path), f"{res.algorithm}_{res.dataset}_seed{res.seed}.csv"
        )
        assert os.path.exists(hist_file), f"Missing history CSV: {hist_file}"
        loaded_hist = load_run_history(hist_file)
        assert len(loaded_hist) == len(res.history)


def test_shared_contract_compliance(api, tmp_path):
    """Verify every algorithm satisfies the BaseNAS output contract."""
    algos = ["random_search", "random_sample", "bayesian_opt", "successive_halving"]

    config = ExperimentConfig(
        datasets=["cifar10"],
        algorithms=algos,
        budget=10,
        bo_n_init=3,
        seeds=[42],
        num_seeds=1,
        results_dir=str(tmp_path),
    )

    runner = ExperimentRunner(config, benchmark=api)
    results = runner.run_all(save_csv=False)

    for res in results:
        n = len(res.history)
        assert n > 0
        assert len(res.best_val_at_step) == n
        assert len(res.best_test_at_step) == n
        assert len(res.cumulative_cost) == n
        assert 0.0 <= res.best.val_accuracy <= 100.0
        assert 0.0 <= res.best.test_accuracy <= 100.0

        # Cumulative simulated cost must strictly increase
        for i in range(n - 1):
            assert res.cumulative_cost[i] <= res.cumulative_cost[i + 1]


def test_graceful_learning_curve_handling(api, tmp_path):
    """Verify runner does not crash whether learning_curve is implemented or stubbed."""
    config = ExperimentConfig(
        datasets=["cifar10"],
        algorithms=["random_search", "learning_curve"],
        budget=5,
        bo_n_init=2,
        seeds=[0],
        num_seeds=1,
        results_dir=str(tmp_path),
    )

    runner = ExperimentRunner(config, benchmark=api)
    results = runner.run_all(save_csv=False)

    # At minimum, random_search must succeed without learning_curve breaking the loop
    algo_names = [r.algorithm for r in results]
    assert "RandomSearch" in algo_names