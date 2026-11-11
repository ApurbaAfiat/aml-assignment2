"""
Main CLI entry point for running NAS experiments on NATS-Bench TSS.

Examples:
    # Run full default suite (all algorithms, all 3 datasets, 10 seeds, budget=100)
    python run_experiments.py

    # Run specific algorithms on CIFAR-10 with 5 seeds and budget=50
    python run_experiments.py --datasets cifar10 --algorithms random_search bayesian_opt --num-seeds 5 --budget 50

    # Run with equal simulated compute time budget (e.g., 200,000 seconds)
    python run_experiments.py --budget-type simulated_time --budget 200000
"""

import argparse
from configs.experiment_config import ExperimentConfig
from experiments.runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Neural Architecture Search experiments on NATS-Bench TSS."
    )

    parser.add_argument(
        "--bench-path",
        type=str,
        default="NATS-tss-v1_0-3ffb9-simple",
        help="Path to the NATS-Bench benchmark directory or .pbz2 file.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10", "cifar100", "ImageNet16-120"],
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[
            "random_search",
            "random_sample",
            "bayesian_opt",
            "successive_halving",
            "learning_curve",
        ],
        choices=[
            "random_search",
            "random_sample",
            "bayesian_opt",
            "successive_halving",
            "learning_curve",
        ],
        help="NAS algorithms to run.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100,
        help="Computational budget limit (evaluations or simulated seconds).",
    )
    parser.add_argument(
        "--budget-type",
        type=str,
        default="evaluations",
        choices=["evaluations", "simulated_time"],
        help="How budget is measured.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of random seeds (uses seeds 0 .. num_seeds-1).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory where CSV results will be saved.",
    )
    parser.add_argument(
        "--bo-n-init",
        type=int,
        default=10,
        help="Number of random warm-up evaluations for Bayesian Optimization.",
    )
    parser.add_argument(
        "--sh-eta",
        type=int,
        default=3,
        help="Reduction factor (eta) for Successive Halving.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = ExperimentConfig(
        bench_path=args.bench_path,
        datasets=args.datasets,
        algorithms=args.algorithms,
        budget=args.budget,
        budget_type=args.budget_type,
        num_seeds=args.num_seeds,
        seeds=list(range(args.num_seeds)),
        results_dir=args.results_dir,
        bo_n_init=min(args.bo_n_init, max(1, args.budget - 1)),
        sh_reduction_factor=args.sh_eta,
    )

    runner = ExperimentRunner(config)
    runner.run_all(save_csv=True)


if __name__ == "__main__":
    main()