"""
Unified wrapper around the NATS-Bench API.

Encapsulates dataset quirks, key name variations, caching, and conversion
to standardized EvalRecord dataclasses.
"""

import os
from typing import Optional, Dict, Any, Tuple
from nats_bench import create
from utils.types import EvalRecord


class BenchmarkAPI:
    """Wrapper around NATS-Bench providing a clean, consistent query interface."""

    VALID_DATASETS = ("cifar10", "cifar100", "ImageNet16-120")
    NUM_ARCHITECTURES = 15625  # Total for TSS (Topology Search Space)

    def __init__(
        self,
        bench_path: str = "NATS-tss-v1_0-3ffb9-simple",
        search_space: str = "tss",
        fast_mode: bool = True,
        verbose: bool = False,
    ):
        """Initialize and load the NATS-Bench API.

        Args:
            bench_path: Path to the benchmark directory (fast mode) or
                        .pbz2 file (full mode).
            search_space: Search space name, defaults to 'tss'.
            fast_mode: If True, lazily loads architectures on query.
            verbose: Whether to print benchmark diagnostic messages.
        """
        if not os.path.exists(bench_path):
            raise FileNotFoundError(
                f"Benchmark path not found at: '{bench_path}'. "
                f"Ensure NATS-tss-v1_0-3ffb9-simple is in the project root."
            )

        self.bench_path = bench_path
        self.search_space = search_space
        self.fast_mode = fast_mode
        self._api = create(bench_path, search_space, fast_mode=fast_mode, verbose=verbose)
        self._cache: Dict[Tuple[int, str, int, Optional[int]], EvalRecord] = {}

    def __len__(self) -> int:
        """Return total number of architectures in the search space (15625)."""
        return len(self._api)

    @property
    def num_architectures(self) -> int:
        """Convenience property returning total architectures (15625)."""
        return len(self._api)

    def arch_str(self, index: int) -> str:
        """Get the architecture string representation from its integer index."""
        if not (0 <= index < len(self._api)):
            raise IndexError(f"Index {index} out of bounds for search space of size {len(self._api)}")
        return self._api.arch(index)

    def query_index(self, arch_str: str) -> int:
        """Get the integer index from an architecture string."""
        idx = self._api.query_index_by_arch(arch_str)
        if idx == -1:
            raise ValueError(f"Architecture string '{arch_str}' not found in benchmark.")
        return idx

    def evaluate(
        self,
        arch_index: int,
        dataset: str,
        hp: int = 200,
        iepoch: Optional[int] = None,
        eval_number: int = 1,
    ) -> EvalRecord:
        """Evaluate an architecture on a dataset and return an EvalRecord.

        Handles dataset quirks:
        - For CIFAR-10: queries 'cifar10-valid' for validation accuracy and
          'cifar10' for test accuracy.
        - For CIFAR-100 & ImageNet16-120: queries the dataset directly.

        Args:
            arch_index: Integer index (0 to 15624).
            dataset: One of 'cifar10', 'cifar100', 'ImageNet16-120'.
            hp: Training budget schedule in epochs (default 200, or 12).
            iepoch: Specific epoch (0-indexed). None queries the final epoch.
            eval_number: Sequential evaluation counter for tracking history.

        Returns:
            An EvalRecord with validation accuracy, test accuracy, training time, etc.
        """
        if dataset not in self.VALID_DATASETS:
            raise ValueError(f"Unknown dataset '{dataset}'. Must be one of {self.VALID_DATASETS}")

        cache_key = (arch_index, dataset, hp, iepoch)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # Return new instance with updated eval_number
            return EvalRecord(
                arch_index=cached.arch_index,
                arch_str=cached.arch_str,
                dataset=cached.dataset,
                hp=cached.hp,
                iepoch=cached.iepoch,
                val_accuracy=cached.val_accuracy,
                test_accuracy=cached.test_accuracy,
                train_time=cached.train_time,
                eval_number=eval_number,
            )

        arch_string = self.arch_str(arch_index)

        if dataset == "cifar10":
            # Validation metrics come from 'cifar10-valid'
            valid_info = self._api.get_more_info(
                arch_index, "cifar10-valid", hp=hp, iepoch=iepoch, is_random=False
            )
            # Test metrics come from 'cifar10'
            test_info = self._api.get_more_info(
                arch_index, "cifar10", hp=hp, iepoch=iepoch, is_random=False
            )

            val_acc = float(valid_info["valid-accuracy"])
            test_acc = float(test_info["test-accuracy"])
            train_time = float(valid_info.get("train-all-time", valid_info.get("train-per-time", 0.0)))
        else:
            # CIFAR-100 and ImageNet16-120 have both in the same record
            info = self._api.get_more_info(
                arch_index, dataset, hp=hp, iepoch=iepoch, is_random=False
            )
            val_acc = float(info["valid-accuracy"])
            test_acc = float(info["test-accuracy"])
            train_time = float(info.get("train-all-time", info.get("train-per-time", 0.0)))

        record = EvalRecord(
            arch_index=arch_index,
            arch_str=arch_string,
            dataset=dataset,
            hp=hp,
            iepoch=iepoch,
            val_accuracy=val_acc,
            test_accuracy=test_acc,
            train_time=train_time,
            eval_number=eval_number,
        )

        self._cache[cache_key] = record
        return record

    def get_cost_info(self, arch_index: int, dataset: str, hp: int = 200) -> Dict[str, Any]:
        """Query computational cost metrics (FLOPs, params, latency).

        Args:
            arch_index: Integer architecture index.
            dataset: Dataset name.
            hp: Hyperparameter schedule.

        Returns:
            Dict containing 'flops', 'params', 'latency', etc.
        """
        # CIFAR-10 cost info is indexed under 'cifar10'
        cost_dataset = "cifar10" if dataset == "cifar10" else dataset
        cost = self._api.get_cost_info(arch_index, cost_dataset, hp=hp)
        return {
            "flops": cost.get("flops"),
            "params": cost.get("params"),
            "latency": cost.get("latency"),
            "train_epoch_time": cost.get("T-train@epoch"),
            "train_total_time": cost.get("T-train@total"),
        }

