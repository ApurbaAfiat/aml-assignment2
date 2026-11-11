import csv
import os
from typing import List, Dict, Any

from utils.types import SearchResult


def save_run_history(result: SearchResult, output_dir: str = "results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{result.algorithm}_{result.dataset}_seed{result.seed}.csv"
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "eval_number",
        "arch_index",
        "hp",
        "iepoch",
        "val_accuracy",
        "test_accuracy",
        "train_time",
        "best_val_so_far",
        "best_test_so_far",
        "cumulative_cost",
    ]

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, record in enumerate(result.history):
            writer.writerow({
                "eval_number": record.eval_number,
                "arch_index": record.arch_index,
                "hp": record.hp,
                "iepoch": record.iepoch if record.iepoch is not None else "",
                "val_accuracy": f"{record.val_accuracy:.4f}",
                "test_accuracy": f"{record.test_accuracy:.4f}",
                "train_time": f"{record.train_time:.4f}",
                "best_val_so_far": f"{result.best_val_at_step[i]:.4f}",
                "best_test_so_far": f"{result.best_test_at_step[i]:.4f}",
                "cumulative_cost": f"{result.cumulative_cost[i]:.4f}",
            })

    return filepath


def save_summary(results: List[SearchResult], output_path: str = "results/summary.csv") -> str:
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    fieldnames = [
        "algorithm",
        "dataset",
        "seed",
        "budget",
        "budget_type",
        "num_evaluations",
        "best_arch_index",
        "best_arch_str",
        "best_val_accuracy",
        "best_test_accuracy",
        "total_simulated_cost",
        "wall_time",
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for res in results:
            total_cost = res.cumulative_cost[-1] if res.cumulative_cost else 0.0
            writer.writerow({
                "algorithm": res.algorithm,
                "dataset": res.dataset,
                "seed": res.seed,
                "budget": res.budget,
                "budget_type": res.budget_type,
                "num_evaluations": len(res.history),
                "best_arch_index": res.best.arch_index,
                "best_arch_str": res.best.arch_str,
                "best_val_accuracy": f"{res.best.val_accuracy:.4f}",
                "best_test_accuracy": f"{res.best.test_accuracy:.4f}",
                "total_simulated_cost": f"{total_cost:.4f}",
                "wall_time": f"{res.wall_time:.4f}",
            })

    return output_path


def load_run_history(filepath: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"History file not found: '{filepath}'")

    rows: List[Dict[str, Any]] = []
    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "eval_number": int(row["eval_number"]),
                "arch_index": int(row["arch_index"]),
                "hp": int(row["hp"]),
                "iepoch": int(row["iepoch"]) if row["iepoch"] != "" else None,
                "val_accuracy": float(row["val_accuracy"]),
                "test_accuracy": float(row["test_accuracy"]),
                "train_time": float(row["train_time"]),
                "best_val_so_far": float(row["best_val_so_far"]),
                "best_test_so_far": float(row["best_test_so_far"]),
                "cumulative_cost": float(row["cumulative_cost"]),
            })
    return rows


def load_summary(filepath: str = "results/summary.csv") -> List[Dict[str, Any]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Summary file not found: '{filepath}'")

    rows: List[Dict[str, Any]] = []
    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "algorithm": row["algorithm"],
                "dataset": row["dataset"],
                "seed": int(row["seed"]),
                "budget": int(row["budget"]),
                "budget_type": row["budget_type"],
                "num_evaluations": int(row["num_evaluations"]),
                "best_arch_index": int(row["best_arch_index"]),
                "best_arch_str": row["best_arch_str"],
                "best_val_accuracy": float(row["best_val_accuracy"]),
                "best_test_accuracy": float(row["best_test_accuracy"]),
                "total_simulated_cost": float(row["total_simulated_cost"]),
                "wall_time": float(row["wall_time"]),
            })
    return rows