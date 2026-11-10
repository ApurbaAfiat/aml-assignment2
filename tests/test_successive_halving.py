import pytest
from benchmark.nats_api import BenchmarkAPI
from algorithms.successive_halving import SuccessiveHalving


@pytest.fixture(scope="module")
def api():
    return BenchmarkAPI(bench_path="NATS-tss-v1_0-3ffb9-simple", fast_mode=True, verbose=False)


def test_sh_evaluation_budget(api):
    budget = 18
    sh = SuccessiveHalving(
        api,
        dataset="cifar10",
        seed=42,
        budget=budget,
        budget_type="evaluations",
        eta=3,
        min_fidelity=8,
    )
    res = sh.run()

    assert len(res.history) == budget
    assert len(res.best_val_at_step) == budget
    assert len(res.cumulative_cost) == budget


def test_sh_multi_fidelity_promotion(api):
    budget = 80
    sh = SuccessiveHalving(
        api,
        dataset="cifar10",
        seed=42,
        budget=budget,
        budget_type="evaluations",
        eta=3,
        min_fidelity=8,
    )
    res = sh.run()

    # Collect all unique fidelities (epochs) queried during the run
    fidelities_queried = {record.iepoch + 1 for record in res.history if record.iepoch is not None}

    # Should have evaluated at lowest fidelity (8 epochs) and at least promoted to rung 1 (24 epochs)
    assert 8 in fidelities_queried
    assert any(f > 8 for f in fidelities_queried), "No candidates were promoted to higher rungs!"


def test_sh_anytime_monotonicity(api):
    sh = SuccessiveHalving(api, dataset="cifar10", seed=101, budget=25, eta=3)
    res = sh.run()

    for i in range(len(res.best_val_at_step) - 1):
        assert res.best_val_at_step[i] <= res.best_val_at_step[i + 1] + 1e-6


def test_sh_reproducibility(api):
    sh1 = SuccessiveHalving(api, dataset="cifar10", seed=555, budget=20, eta=3)
    res1 = sh1.run()

    sh2 = SuccessiveHalving(api, dataset="cifar10", seed=555, budget=20, eta=3)
    res2 = sh2.run()

    history1 = [(r.arch_index, r.iepoch) for r in res1.history]
    history2 = [(r.arch_index, r.iepoch) for r in res2.history]

    assert history1 == history2, "Same seed produced different evaluation sequences!"
    assert res1.best.arch_index == res2.best.arch_index
    assert res1.best.val_accuracy == res2.best.val_accuracy