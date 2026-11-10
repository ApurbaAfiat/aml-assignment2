"""
Unit tests for the Bayesian Optimization algorithm.
"""

import pytest
import numpy as np
from benchmark.nats_api import BenchmarkAPI
from algorithms.bayesian_opt import BayesianOptimization


@pytest.fixture(scope="module")
def api():
    """Shared BenchmarkAPI instance for fast testing."""
    return BenchmarkAPI(bench_path="NATS-tss-v1_0-3ffb9-simple", fast_mode=True, verbose=False)


def test_bo_evaluation_budget(api):
    """Test that Bayesian Optimization respects an evaluation count budget."""
    budget = 12
    n_init = 4
    bo = BayesianOptimization(
        api,
        dataset="cifar10",
        seed=42,
        budget=budget,
        n_init=n_init,
        budget_type="evaluations",
    )
    res = bo.run()

    assert len(res.history) == budget
    assert len(res.best_val_at_step) == budget
    assert len(res.cumulative_cost) == budget

    # Verify no duplicate architectures
    evaluated_indices = [r.arch_index for r in res.history]
    assert len(evaluated_indices) == len(set(evaluated_indices)), "Found duplicate architecture in BO evaluation history!"


def test_bo_expected_improvement_calculation(api):
    """Test that Expected Improvement math behaves properly."""
    bo = BayesianOptimization(api, dataset="cifar10", seed=42, budget=10, n_init=3)

    # If predicted mean is far above best_val, EI must be strictly positive
    mean_promising = np.array([95.0])
    std = np.array([2.0])
    best_val = 80.0
    ei_promising = bo._expected_improvement(mean_promising, std, best_val)
    assert ei_promising[0] > 0.0

    # If predicted mean is far below best_val with near-zero uncertainty, EI should be ~0
    mean_poor = np.array([50.0])
    std_small = np.array([1e-5])
    ei_poor = bo._expected_improvement(mean_poor, std_small, best_val)
    assert ei_poor[0] == pytest.approx(0.0, abs=1e-5)


def test_bo_anytime_monotonicity(api):
    """Test that best validation curve is non-decreasing."""
    bo = BayesianOptimization(api, dataset="cifar10", seed=100, budget=10, n_init=4)
    res = bo.run()

    for i in range(len(res.best_val_at_step) - 1):
        assert res.best_val_at_step[i] <= res.best_val_at_step[i + 1] + 1e-6


def test_bo_reproducibility(api):
    """Test that identical seeds yield identical search trajectories."""
    bo1 = BayesianOptimization(api, dataset="cifar10", seed=777, budget=8, n_init=3)
    res1 = bo1.run()

    bo2 = BayesianOptimization(api, dataset="cifar10", seed=777, budget=8, n_init=3)
    res2 = bo2.run()

    indices1 = [r.arch_index for r in res1.history]
    indices2 = [r.arch_index for r in res2.history]
    assert indices1 == indices2, "Same seed produced different BO trajectories!"
    assert res1.best.arch_index == res2.best.arch_index
    assert res1.best.val_accuracy == res2.best.val_accuracy