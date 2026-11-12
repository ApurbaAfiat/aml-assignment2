"""Plot search progress from experiment result CSV files."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from experiments.results import load_run_history, load_summary


ALGORITHM_LABELS = {
    "RandomSearch": "Random Search",
    "RandomSample": "Random Sample",
    "BayesianOptimization": "Bayesian Optimisation",
    "SuccessiveHalving": "Successive Halving",
    "LearningCurve": "Learning Curve Extrapolation",
}


def plot_learning_curves(
    summary_path: str = "results/summary.csv",
    output_dir: str = "figures",
    x_axis: str = "simulated_time",
) -> List[Path]:
    """Plot mean best-so-far validation accuracy for each algorithm and dataset.

    Args:
        summary_path: Summary CSV used to identify the runs.
        output_dir: Directory where the figures are saved.
        x_axis: Either "simulated_time" or "evaluations".

    Returns:
        Paths to the generated figures.
    """
    if x_axis not in {"simulated_time", "evaluations"}:
        raise ValueError("x_axis must be 'simulated_time' or 'evaluations'.")

    summary_file = Path(summary_path)
    summary_rows = load_summary(str(summary_file))
    if not summary_rows:
        raise ValueError(f"No experiment results found in '{summary_path}'.")

    runs: Dict[Tuple[str, str], List[List[dict]]] = defaultdict(list)
    seen_runs = set()

    for row in summary_rows:
        dataset = row["dataset"]
        algorithm = row["algorithm"]
        seed = row["seed"]
        key = (dataset, algorithm, seed)

        if key in seen_runs:
            raise ValueError(
                f"Duplicate summary row for dataset/algorithm/seed {key}; "
                "the history filename does not distinguish multiple budgets."
            )
        seen_runs.add(key)

        history_path = summary_file.parent / (
            f"{algorithm}_{dataset}_seed{seed}.csv"
        )
        history = load_run_history(str(history_path))
        if not history:
            raise ValueError(f"Run history is empty: '{history_path}'.")

        runs[(dataset, algorithm)].append(history)

    datasets = sorted({dataset for dataset, _ in runs})
    algorithms = list(dict.fromkeys(algorithm for _, algorithm in runs))

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
    global_x_min = float("inf")
    global_x_max = 0.0

    for dataset_index, dataset in enumerate(datasets):
        ax = axes[0, dataset_index]

        for algorithm in algorithms:
            algorithm_runs = runs.get((dataset, algorithm))
            if not algorithm_runs:
                continue

            run_x = []
            run_y = []

            for history in algorithm_runs:
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

                valid = np.isfinite(x) & np.isfinite(y)
                x = x[valid]
                y = y[valid]

                if x_axis == "simulated_time":
                    positive = x > 0
                    x = x[positive]
                    y = y[positive]

                if len(x):
                    run_x.append(x)
                    run_y.append(y)

            if len(run_x) != len(algorithm_runs):
                raise ValueError(
                    f"Invalid or missing x-axis data for {algorithm} on {dataset}."
                )

            if x_axis == "evaluations":
                shared_end = min(len(values) for values in run_y)
                grid = np.arange(1, shared_end + 1, dtype=float)
                aligned = np.stack([values[:shared_end] for values in run_y])
            else:
                shared_start = max(values[0] for values in run_x)
                shared_end = min(values[-1] for values in run_x)
                if shared_end <= shared_start:
                    continue

                grid = np.linspace(shared_start, shared_end, 200)
                aligned = np.stack([
                    np.interp(grid, x, y)
                    for x, y in zip(run_x, run_y)
                ])

            global_x_min = min(global_x_min, float(grid[0]))
            global_x_max = max(global_x_max, float(grid[-1]))
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
            else "Cumulative simulated training time (s)",
            fontsize=12,
        )
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(axis="both", alpha=0.25, linestyle="--")

    for ax in axes[0]:
      ax.set_xscale("log")
      ax.set_xlim(global_x_min * 0.9, global_x_max * 1.05)

    axes[0, 0].set_ylabel("Best validation accuracy (%)", fontsize=12)
    fig.suptitle("Search progress by dataset", fontsize=16, fontweight="bold")

    if legend_items:
        fig.legend(
            [legend_items[algorithm] for algorithm in legend_items],
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

    figure_path = output_path / f"learning_curves_{x_axis}.png"
    fig.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return [figure_path]


if __name__ == "__main__":
    for path in plot_learning_curves():
        print(f"Saved {path}")