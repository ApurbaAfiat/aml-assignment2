"""Comparison charts generated from the experiment summary CSV."""

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from experiments.results import load_summary


METRICS = {
    "best_val_accuracy": ("Validation accuracy (%)", "comparison_validation_accuracy.png"),
    "best_test_accuracy": ("Test accuracy (%)", "comparison_test_accuracy.png"),
    "total_simulated_cost": ("Simulated training cost (seconds)", "comparison_simulated_cost.png"),
}

ALGORITHM_LABELS = {
    "RandomSearch": "Random Search",
    "RandomSample": "Random Sample",
    "BayesianOpt": "Bayesian Optimisation",
    "SuccessiveHalving": "Successive Halving",
    "LearningCurve": "Learning Curve Extrapolation",
}

def plot_comparison_charts(
    summary_path: str = "results/summary.csv",
    output_dir: str = "figures",
) -> List[Path]:
    """Plot per-dataset algorithm means with standard-deviation error bars."""
    rows = load_summary(summary_path)
    if not rows:
        raise ValueError(f"No experiment results found in '{summary_path}'.")

    # Keep comparisons meaningful if the CSV contains multiple budget settings.
    budgets_by_dataset: Dict[str, set] = defaultdict(set)
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in rows:
        dataset = row["dataset"]
        algorithm = row["algorithm"]
        budgets_by_dataset[dataset].add((row["budget_type"], row["budget"]))

        for metric in METRICS:
            grouped[(dataset, algorithm)][metric].append(float(row[metric]))

    for dataset, budgets in budgets_by_dataset.items():
        if len(budgets) > 1:
            raise ValueError(
                f"Dataset '{dataset}' has multiple budget settings: {sorted(budgets)}. "
                "Plot each budget separately to avoid comparing unlike runs."
            )

    datasets = sorted({row["dataset"] for row in rows})
    algorithms = list(dict.fromkeys(row["algorithm"] for row in rows))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for metric, (ylabel, filename) in METRICS.items():
        fig, axes = plt.subplots(
            1,
            len(datasets),
            figsize=(5 * len(datasets), 4.5),
            squeeze=False,
        )

        for ax, dataset in zip(axes[0], datasets):
            present = [
                algorithm
                for algorithm in algorithms
                if (dataset, algorithm) in grouped
            ]
            means = [
                np.mean(grouped[(dataset, algorithm)][metric])
                for algorithm in present
            ]
            stds = [
                np.std(grouped[(dataset, algorithm)][metric])
                for algorithm in present
            ]
            labels = [
                ALGORITHM_LABELS.get(algorithm, algorithm)
                for algorithm in present
            ]

            x = np.arange(len(present))
            ax.bar(x, means, yerr=stds, capsize=4)
            ax.set_xticks(x, labels, rotation=25, ha="right")
            ax.set_title(dataset)
            ax.set_ylabel(ylabel)
            if metric in {"best_val_accuracy", "best_test_accuracy"}:
              ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.25)

        fig.suptitle(f"NAS algorithm comparison: {ylabel}")
        fig.tight_layout()

        path = output_path / filename
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths


if __name__ == "__main__":
    for chart_path in plot_comparison_charts():
        print(f"Saved {chart_path}")