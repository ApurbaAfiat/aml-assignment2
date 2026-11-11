"""Learning Curve Extrapolation for NATS-Bench TSS."""

import math
import time
from typing import List, Tuple

import numpy as np

from algorithms.base import BaseNAS
from utils.types import SearchResult


class LearningCurve(BaseNAS):
    """Extrapolate validation learning curves and stop unpromising candidates."""

    MAX_EPOCHS = 200
    MIN_EPOCHS = 5
    STOP_PROBABILITY = 0.05

    def _fit_predictive_models(
        self,
        accuracies: List[float],
    ) -> List[Tuple[float, float, float]]:
        """Return (model weight, predicted mean, predictive variance) tuples."""
        values = np.asarray(accuracies, dtype=float)
        epochs = np.arange(1, len(values) + 1, dtype=float)

        # Each basis represents a possible learning-curve shape. For each
        # exponent/rate, fit the two linear coefficients by least squares.
        bases = []
        for exponent in (0.25, 0.5, 1.0, 2.0):
            bases.append(
                (
                    f"power_{exponent}",
                    lambda x, p=exponent: -np.power(x, -p),
                )
            )
        for rate in (0.01, 0.03, 0.1, 0.3):
            bases.append(
                (
                    f"exponential_{rate}",
                    lambda x, r=rate: -np.exp(-r * x),
                )
            )
        bases.extend(
            [
                ("logarithmic", lambda x: np.log(x)),
                ("linear", lambda x: x / self.MAX_EPOCHS),
            ]
        )

        models = []
        for name, basis in bases:
            features = basis(epochs)
            design = np.column_stack((np.ones_like(features), features))

            coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
            residuals = values - design @ coefficients
            rss = float(residuals @ residuals)

            # BIC-based evidence approximation: models fitting the observations
            # well receive more weight, with a penalty for model complexity.
            variance = max(rss / len(values), 1e-6)
            log_evidence = -0.5 * len(values) * math.log(variance)
            log_evidence -= 0.5 * design.shape[1] * math.log(len(values))

            final_feature = basis(np.asarray([self.MAX_EPOCHS], dtype=float))[0]
            final_design = np.asarray([1.0, final_feature])
            predicted_mean = float(final_design @ coefficients)

            parameter_covariance = variance * np.linalg.pinv(design.T @ design)
            predictive_variance = variance + float(
                final_design @ parameter_covariance @ final_design.T
            )

            models.append(
                (name, log_evidence, predicted_mean, max(predictive_variance, 1e-6))
            )

        # Normalize the model evidence values into model weights.
        max_log_evidence = max(model[1] for model in models)
        unnormalized = [
            math.exp(model[1] - max_log_evidence) for model in models
        ]
        weight_total = sum(unnormalized)

        return [
            (weight / weight_total, mean, variance)
            for weight, (_, _, mean, variance) in zip(unnormalized, models)
        ]

    def _predict_final_accuracy(self, accuracies: List[float]) -> Tuple[float, float]:
        """Return the model-averaged final-accuracy mean and standard deviation."""
        models = self._fit_predictive_models(accuracies)
        mean = sum(weight * prediction for weight, prediction, _ in models)
        variance = sum(
            weight * (model_variance + (prediction - mean) ** 2)
            for weight, prediction, model_variance in models
        )
        return mean, math.sqrt(max(variance, 1e-6))

    def _probability_of_beating(
        self,
        accuracies: List[float],
        incumbent: float,
    ) -> float:
        """Estimate the probability that a candidate beats the incumbent."""
        models = self._fit_predictive_models(accuracies)
        probability = 0.0

        for weight, mean, variance in models:
            standard_deviation = math.sqrt(max(variance, 1e-6))
            z_score = (incumbent - mean) / standard_deviation
            probability_of_beating = 0.5 * (
                1.0 - math.erf(z_score / math.sqrt(2.0))
            )
            probability += weight * probability_of_beating

        return probability

    def run(self) -> SearchResult:
        start_time = time.time()
        total_archs = self.benchmark.num_architectures

        if self.budget_type not in {"evaluations", "simulated_time"}:
            raise ValueError(f"Unknown budget type: {self.budget_type}")
        if self.budget <= 0:
            raise ValueError("budget must be greater than zero")

        # A seeded generator makes the candidate order reproducible.
        candidate_order = self.rng.permutation(total_archs)
        num_iterations = 0
        incumbent_final_accuracy = float("-inf")

        for candidate_idx in candidate_order:
            candidate_idx = int(candidate_idx)
            candidate_curve: List[float] = []
            reached_final_epoch = False

            for epoch in range(self.MAX_EPOCHS):
                if self.budget_type == "evaluations":
                    if num_iterations >= self.budget:
                        break
                else:
                    current_cost = (
                        self.cumulative_cost[-1]
                        if self.cumulative_cost
                        else 0.0
                    )
                    if current_cost >= self.budget:
                        break

                record = self._evaluate(
                    arch_index=candidate_idx,
                    hp=self.MAX_EPOCHS,
                    iepoch=epoch,
                )
                num_iterations += 1
                candidate_curve.append(float(record.val_accuracy))

                if epoch == self.MAX_EPOCHS - 1:
                    reached_final_epoch = True

                # After enough observations, stop if the candidate is unlikely
                # to exceed the best completed candidate's final-epoch score.
                if (
                    len(candidate_curve) >= self.MIN_EPOCHS
                    and incumbent_final_accuracy > float("-inf")
                    and self._probability_of_beating(
                        candidate_curve,
                        incumbent_final_accuracy,
                    )
                    < self.STOP_PROBABILITY
                ):
                    break

            # Only a candidate evaluated through the final epoch can establish
            # or improve the incumbent final-epoch validation accuracy.
            if reached_final_epoch and candidate_curve:
                final_accuracy = candidate_curve[-1]
                incumbent_final_accuracy = max(
                    incumbent_final_accuracy,
                    final_accuracy,
                )

            if self.budget_type == "evaluations":
                if num_iterations >= self.budget:
                    break
            else:
                current_cost = (
                    self.cumulative_cost[-1] if self.cumulative_cost else 0.0
                )
                if current_cost >= self.budget:
                    break

        return self._create_search_result(
            algorithm_name="LearningCurve",
            start_wall_time=start_time,
            metadata={
                "num_iterations": num_iterations,
                "incumbent_final_val_accuracy": (
                    None
                    if incumbent_final_accuracy == float("-inf")
                    else incumbent_final_accuracy
                ),
            },
        )