"""
Unit tests for the Random Search algorithm.
"""

import pytest
from benchmark.nats_api import BenchmarkAPI
from algorithms.random_search import RandomSearch


@pytest.fixture(scope="module")
def api():
    """Shared BenchmarkAPI instance for fast testing."""
    return BenchmarkAPI(bench_path="NATS-tss-v1_0-3ffb9-simple", fast_mode=True, verbose=False)


def test_random_search_evaluation_budget(api):
    """Test that Random Search respects an evaluation count budget."""
    budget = 15
    rs = RandomSearch(api, dataset="cifar10", seed=42, budget=budget, budget_type="evaluations")
    res = rs.run()

    assert len(res.history) == budget
    assert len(res.best_val_at_step) == budget
    assert len(res.cumulative_cost) == budget

    # Verify no duplicates
    evaluated_indices = [r.arch_index for r in res.history]
    assert len(evaluated_indices) == len(set(evaluated_indices)), "Found duplicate architecture in evaluation history!"


def test_random_search_anytime_monotonicity(api):
    """Test that the best validation curve is non-decreasing."""
    rs = RandomSearch(api, dataset="cifar10", seed=123, budget=10)
    res = rs.run()

    # The best-so-far curve should never decrease
    for i in range(len(res.best_val_at_step) - 1):
        assert res.best_val_at_step[i] <= res.best_val_at_step[i + 1] + 1e-6


def test_random_search_reproducibility(api):
    """Test that the same seed produces identical results."""
    rs1 = RandomSearch(api, dataset="cifar10", seed=999, budget=8)
    res1 = rs1.run()

    rs2 = RandomSearch(api, dataset="cifar10", seed=999, budget=8)
    res2 = rs2.run()

    indices1 = [r.arch_index for r in res1.history]
    indices2 = [r.arch_index for r in res2.history]
    assert indices1 == indices2, "Same seed produced different architecture sequences!"
    assert res1.best.arch_index == res2.best.arch_index
    assert res1.best.val_accuracy == res2.best.val_accuracy


def test_random_search_simulated_time_budget(api):
    """Test that simulated_time budget terminates when cost exceeds limit."""
    # 20,000 simulated seconds (roughly 1-2 evaluations)
    time_budget = 20000.0
    rs = RandomSearch(api, dataset="cifar10", seed=42, budget=time_budget, budget_type="simulated_time")
    res = rs.run()

    assert res.cumulative_cost[-1] >= time_budget
    assert len(res.history) >= 1