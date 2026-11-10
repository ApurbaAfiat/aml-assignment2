"""Unit tests for the Random Sample algorithm."""

import pytest

from algorithms.random_sample import RandomSample
from benchmark.nats_api import BenchmarkAPI


@pytest.fixture(scope="module")
def api():
    """Shared BenchmarkAPI instance for fast testing."""
    return BenchmarkAPI(
        bench_path="NATS-tss-v1_0-3ffb9-simple",
        fast_mode=True,
        verbose=False,
    )


def test_random_sample_evaluation_budget(api):
    """Test that Random Sample uses the requested number of evaluations."""
    budget = 15
    rs = RandomSample(
        api,
        dataset="cifar10",
        seed=42,
        budget=budget,
        budget_type="evaluations",
    )
    res = rs.run()

    assert len(res.history) == budget
    assert len(res.best_val_at_step) == budget
    assert len(res.cumulative_cost) == budget

    # Sampling is without replacement
    evaluated_indices = [record.arch_index for record in res.history]
    assert len(evaluated_indices) == len(set(evaluated_indices))


def test_random_sample_anytime_monotonicity(api):
    """Test that the best validation curve never decreases."""
    rs = RandomSample(
        api,
        dataset="cifar10",
        seed=123,
        budget=10,
        budget_type="evaluations",
    )
    res = rs.run()

    for i in range(len(res.best_val_at_step) - 1):
        assert res.best_val_at_step[i] <= res.best_val_at_step[i + 1] + 1e-6


def test_random_sample_reproducibility(api):
    """Test that the same seed produces the same sampled architectures."""
    rs1 = RandomSample(
        api,
        dataset="cifar10",
        seed=999,
        budget=8,
        budget_type="evaluations",
    )
    res1 = rs1.run()

    rs2 = RandomSample(
        api,
        dataset="cifar10",
        seed=999,
        budget=8,
        budget_type="evaluations",
    )
    res2 = rs2.run()

    indices1 = [record.arch_index for record in res1.history]
    indices2 = [record.arch_index for record in res2.history]

    assert indices1 == indices2
    assert res1.best.arch_index == res2.best.arch_index
    assert res1.best.val_accuracy == res2.best.val_accuracy


def test_random_sample_simulated_time_budget(api):
    """Test that sampling stops once the simulated-time budget is reached."""
    time_budget = 20_000.0
    rs = RandomSample(
        api,
        dataset="cifar10",
        seed=42,
        budget=time_budget,
        budget_type="simulated_time",
    )
    res = rs.run()

    assert len(res.history) >= 1
    assert res.cumulative_cost[-1] >= time_budget