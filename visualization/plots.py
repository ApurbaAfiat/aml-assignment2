"""Convergence plots using summary.csv and per-run history CSV files."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from experiments.results import load_run_history, load_summary


ALGORITHM_LABELS = {
    "RandomSearch": "Random Search",
    "RandomSample": "Random Sample",
    "BayesianOpt": "Bayesian Optimisation",
    "SuccessiveHalving": "Successive Halving",
    "LearningCurve": "Learning Curve Extrapolation",
}


def _history_x_values(history: List[dict], x_axis: str) -> np.ndarray:
    """Get positive x-values for a run's history."""
    if x_axis == "evaluations":
        values = np.arange(1, len(history) + 1, dtype=float)
    else:
        values = np.array(
            [record["cumulative_cost"] for record in history],
            dtype=float,
        )

    return values[np.isfinite(values) & (values > 0)]


def plot_convergence(
    summary_path: str = "results/summary.csv",
    output_dir: str = "figures",
    x_axis: str = "evaluations",
) -> List[Path]:
    """Plot mean best-so-far validation accuracy for each algorithm and dataset.

    The summary CSV identifies runs; the matching per-run history CSVs supply
    the convergence data. The standard-deviation band shows variation across
    seeds.
    """
    if x_axis not in {"evaluations", "simulated_time"}:
        raise ValueError("x_axis must be 'evaluations' or 'simulated_time'.")

    summary_file = Path(summary_path)
    rows = load_summary(str(summary_file))
    if not rows:
        raise ValueError(f"No experiment results found in '{summary_path}'.")

    runs: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    seen_runs = set()

    for row in rows:
        dataset = row["dataset"]
        algorithm = row["algorithm"]
        seed = row["seed"]
        run_key = (dataset, algorithm, seed)

        if run_key in seen_runs:
            raise ValueError(
                f"Multiple summary rows found for dataset/algorithm/seed "
                f"{run_key}. The history filename does not distinguish budgets."
            )
        seen_runs.add(run_key)

        history_path = summary_file.parent / (
            f"{algorithm}_{dataset}_seed{seed}.csv"
        )
        history = load_run_history(str(history_path))
        if not history:
            raise ValueError(f"Run history is empty: '{history_path}'.")

        runs[(dataset, algorithm)].append(
            {"seed": seed, "history": history}
        )

    datasets = sorted({dataset for dataset, _ in runs})
    algorithms = list(dict.fromkeys(algorithm for _, algorithm in runs))

    all_x_values = [
        value
        for dataset_runs in runs.values()
        for run in dataset_runs
        for value in _history_x_values(run["history"], x_axis)
    ]
    if not all_x_values:
        raise ValueError("No positive x-values available for the logarithmic axis.")

    x_min = min(all_x_values)
    x_max = max(all_x_values)
    if x_min == x_max:
        x_limits = (x_min / 1.1, x_max * 1.1)
    else:
        x_limits = (x_min / 1.05, x_max * 1.05)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    colors = plt.get_cmap("tab10")
    algorithm_colors = {
        algorithm: colors(index % 10)
        for index, algorithm in enumerate(algorithms)
    }
    algorithm_colors["RandomSearch"] = "black"

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(7 * len(datasets), 5.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    legend_items = {}

    for ax, dataset in zip(axes[0], datasets):
        for algorithm in algorithms:
            algorithm_runs = runs.get((dataset, algorithm))
            if not algorithm_runs:
                continue

            x_values = []
            y_values = []

            for run in algorithm_runs:
                history = run["history"]

                if x_axis == "evaluations":
                    x = np.arange(1, len(history) + 1, dtype=float)
                else:
                    x = np.array(
                        [record["cumulative_cost"] for record in history],
                        dtype=float,
                    )

                y = np.array(
                    [record["best_val_so_far"] for record in history],
                    dtype=float,
                )
                valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
                x_values.append(x[valid])
                y_values.append(y[valid])

            if any(len(x) == 0 for x in x_values):
                continue

            if x_axis == "evaluations":
                common_end = min(len(y) for y in y_values)
                grid = np.arange(1, common_end + 1, dtype=float)
                aligned = np.stack([y[:common_end] for y in y_values])
            else:
                common_start = max(x[0] for x in x_values)
                common_end = min(x[-1] for x in x_values)
                if common_end <= common_start:
                    continue

                grid = np.geomspace(common_start, common_end, 200)
                aligned = np.stack([
                    np.interp(grid, x, y)
                    for x, y in zip(x_values, y_values)
                ])

            mean = aligned.mean(axis=0)
            std = aligned.std(axis=0)
            is_random_search = algorithm == "RandomSearch"
            color = algorithm_colors[algorithm]

            ax.fill_between(
                grid,
                np.maximum(mean - std, 0),
                np.minimum(mean + std, 100),
                color=color,
                alpha=0.12 if is_random_search else 0.16,
                linewidth=0,
                zorder=1,
            )

            line_options = {
                "label": ALGORITHM_LABELS.get(algorithm, algorithm),
                "color": color,
                "linestyle": "--" if is_random_search else "-",
                "zorder": 10 if is_random_search else 3,
            }

            (line,) = ax.plot(grid, mean, **line_options)
            legend_items.setdefault(algorithm, line)

        ax.set_title(dataset, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel(
            "Evaluations"
            if x_axis == "evaluations"
            else "Cumulative simulated cost (s)",
            fontsize=12,
        )
        ax.set_xscale("log")
        ax.set_xlim(*x_limits)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(axis="both", alpha=0.25, linestyle="--")

    axes[0, 0].set_ylabel("Best validation accuracy (%)", fontsize=12)
    fig.suptitle(
        "Search convergence by dataset",
        fontsize=16,
        fontweight="bold",
    )

    if legend_items:
        fig.legend(
            legend_items.values(),
            [
                ALGORITHM_LABELS.get(algorithm, algorithm)
                for algorithm in legend_items
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(legend_items), 5),
            frameon=False,
            fontsize=10,
        )

    fig.tight_layout(rect=(0, 0.12, 1, 0.92))

    path = output_path / f"convergence_all_datasets_{x_axis}.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return [path]


if __name__ == "__main__":
    for chart_path in plot_convergence():
        print(f"Saved {chart_path}")