"""
Unit tests for the BenchmarkAPI wrapper.
"""

import pytest
from benchmark.nats_api import BenchmarkAPI
from utils.types import EvalRecord


@pytest.fixture(scope="module")
def api():
    """Load benchmark API once for all tests in this module."""
    return BenchmarkAPI(bench_path="NATS-tss-v1_0-3ffb9-simple", fast_mode=True, verbose=False)


def test_api_initialization(api):
    assert len(api) == 15625
    assert api.num_architectures == 15625


def test_arch_string_and_index_conversion(api):
    arch_str = api.arch_str(0)
    assert isinstance(arch_str, str)
    assert len(arch_str) > 0
    idx = api.query_index(arch_str)
    assert idx == 0


def test_evaluate_cifar10(api):
    record = api.evaluate(arch_index=0, dataset="cifar10", hp=200, eval_number=1)
    assert isinstance(record, EvalRecord)
    assert record.arch_index == 0
    assert record.dataset == "cifar10"
    assert record.hp == 200
    assert 0.0 <= record.val_accuracy <= 100.0
    assert 0.0 <= record.test_accuracy <= 100.0
    assert record.train_time > 0.0
    assert record.eval_number == 1


def test_evaluate_cifar100(api):
    record = api.evaluate(arch_index=0, dataset="cifar100", hp=200, eval_number=2)
    assert isinstance(record, EvalRecord)
    assert record.dataset == "cifar100"
    assert 0.0 <= record.val_accuracy <= 100.0
    assert 0.0 <= record.test_accuracy <= 100.0
    assert record.eval_number == 2


def test_evaluate_imagenet(api):
    record = api.evaluate(arch_index=0, dataset="ImageNet16-120", hp=200, eval_number=3)
    assert isinstance(record, EvalRecord)
    assert record.dataset == "ImageNet16-120"
    assert 0.0 <= record.val_accuracy <= 100.0
    assert 0.0 <= record.test_accuracy <= 100.0


def test_get_cost_info(api):
    cost = api.get_cost_info(arch_index=0, dataset="cifar10", hp=200)
    assert "flops" in cost
    assert "params" in cost
    assert "latency" in cost
    assert cost["flops"] is not None
    assert cost["params"] is not None


def test_invalid_dataset_raises(api):
    with pytest.raises(ValueError):
        api.evaluate(arch_index=0, dataset="invalid_dataset")


def test_invalid_index_raises(api):
    with pytest.raises(IndexError):
        api.arch_str(99999)

