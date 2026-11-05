from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ExperimentConfig:
    bench_path: str = "NATS-tss-v1_0-3ffb9-simple"
    search_space: str = "tss"
    fast_mode: bool = True

    ##Experimental design
    datasets: List[str] = field(default_factory=lambda: ["cifar10", "cifar100", "ImageNet16-120"])
    algorithms: List[str]= field(
        default_factory=lambda:[
            "random_search", "random_sample", "bayesian_opt", "successive_halving", "learning_curve"
        ]
    )
    budget: int = 100
    budget_type: str = "evaluations"
    hp_schedule: int = 200
    num_seeds: int = 10
    seeds: List[int] = field(default_factory=lambda: list(range(10)))

    results_dir: str ="results"
    figures_dir: str ="figures"

    #Algorithm specific hyperparameters
    #1. Bayesian optimisation

    bo_n_init: int = 10
    bo_exploration_weight: float = 0.01

    #2. Successive halving
    sh_reduction_factor: int = 3
    sh_min_epoch: int = 12
    sh_max_epoch: int = 200

    #3. Learning curve extrapolation
    lce_observed_epochs: int = 20
    lce_target_epoch: int = 199
    lce_pool_size: int = 100

    def __post_init__(self):
        if self.budget <=0:
            raise ValueError(f"budget must be positive, got {self.budget}")
        if self.budget_type not in ("evaluations", "simulated_time"):
                raise ValueError(f"budget must be 'evaluations' or 'simulated_time', got '{self.budget_type}'")
        if self.bo_n_init >= self.budget:
                    raise ValueError(f"bo_n_init ({self.bo_n_init}) must be less than budget ({self.budget})")
        if self.sh_reduction_factor <= 1:
                    raise ValueError(f"sh_reduction_factor must be > 1, got {self.sh_reduction_factor}")
        if self.lce_observed_epochs >= self.lce_target_epoch:
                    raise ValueError(f"lce_observed_epochs ({self.lce_observed_epochs}) must be < target ({self.lce_target_epoch})")
