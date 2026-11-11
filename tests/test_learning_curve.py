"""Unit tests for Learning Curve Extrapolation."""

import pytest

from algorithms.learning_curve import LearningCurve
from benchmark.nats_api import BenchmarkAPI


@pytest.fixture(scope="module")
def api():
    """Shared BenchmarkAPI instance for fast testing."""
    return BenchmarkAPI(
        bench_path="NATS-tss-v1_0-3ffb9-simple",
        fast_mode=True,
        verbose=False,
    )

def test_learning_curve_evaluation_budget(api):
    """Each epoch query counts as one iteration."""
    budget = 15
    algorithm = LearningCurve(
        api,
        dataset="cifar10",
        seed=42,
        budget=budget,
        budget_type="evaluations",
    )

    result = algorithm.run()

    assert len(result.history) == budget
    assert len(result.best_val_at_step) == budget
    assert len(result.cumulative_cost) == budget


def test_learning_curve_epochs_progress_and_are_bounded(api):
    """Epoch queries should progress from epoch 0 and stay below MAX_EPOCHS."""
    algorithm = LearningCurve(
        api,
        dataset="cifar10",
        seed=123,
        budget=12,
        budget_type="evaluations",
    )
    calls = []
    original_evaluate = algorithm._evaluate

    def tracking_evaluate(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_evaluate(*args, **kwargs)

    algorithm._evaluate = tracking_evaluate
    algorithm.run()

    epochs = [call["iepoch"] for call in calls]
    # With only 12 iterations, the first candidate has no incumbent to compare
    # against, so it should be queried at epochs 0 through 11.
    assert epochs == list(range(12))
    assert all(0 <= epoch < algorithm.MAX_EPOCHS for epoch in epochs)


def test_learning_curve_best_validation_is_monotonic(api):
    """The best-validation curve should never decrease."""
    algorithm = LearningCurve(
        api,
        dataset="cifar10",
        seed=123,
        budget=15,
        budget_type="evaluations",
    )

    result = algorithm.run()

    assert all(
        earlier <= later
        for earlier, later in zip(
            result.best_val_at_step,
            result.best_val_at_step[1:],
        )
    )


def test_learning_curve_reproducibility(api):
    """The same seed should produce the same architecture/epoch sequence."""

    def run_with_seed():
        algorithm = LearningCurve(
            api,
            dataset="cifar10",
            seed=999,
            budget=15,
            budget_type="evaluations",
        )
        calls = []
        original_evaluate = algorithm._evaluate

        def tracking_evaluate(*args, **kwargs):
            calls.append((kwargs["arch_index"], kwargs["iepoch"]))
            return original_evaluate(*args, **kwargs)

        algorithm._evaluate = tracking_evaluate
        result = algorithm.run()
        return calls, result.best_val_at_step

    calls_1, best_curve_1 = run_with_seed()
    calls_2, best_curve_2 = run_with_seed()

    assert calls_1 == calls_2
    assert best_curve_1 == best_curve_2


def test_learning_curve_early_stops_a_poor_candidate(api):
    """A candidate predicted not to beat the incumbent stops after MIN_EPOCHS."""
    algorithm = LearningCurve(
        api,
        dataset="cifar10",
        seed=42,
        budget=205,
        budget_type="evaluations",
    )

    # The first candidate runs all 200 epochs because there is no incumbent yet.
    # Later candidates are then treated as unlikely to beat that incumbent.
    algorithm._probability_of_beating = lambda accuracies, incumbent: 0.0

    calls = []
    original_evaluate = algorithm._evaluate

    def tracking_evaluate(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_evaluate(*args, **kwargs)

    algorithm._evaluate = tracking_evaluate
    algorithm.run()

    assert len(calls) == 200 + algorithm.MIN_EPOCHS
    assert [call["iepoch"] for call in calls[:200]] == list(range(200))
    assert [call["iepoch"] for call in calls[200:]] == list(
        range(algorithm.MIN_EPOCHS)
    )
    assert calls[0]["arch_index"] != calls[200]["arch_index"]


def test_learning_curve_simulated_time_budget(api):
    """A very small time budget should be reached after an evaluation."""
    algorithm = LearningCurve(
        api,
        dataset="cifar10",
        seed=42,
        budget=1e-12,
        budget_type="simulated_time",
    )

    result = algorithm.run()

    assert len(result.history) >= 1
    assert result.cumulative_cost[-1] >= algorithm.budget