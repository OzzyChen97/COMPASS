"""Serializable validation-driven learning-rate and early-stop control."""

from dataclasses import dataclass
from typing import Iterable


@dataclass
class PlateauDecision:
    improved: bool
    reduce_lr: bool
    stop: bool
    best_metric: float
    bad_evals: int
    current_lr: float


class PlateauController:
    """Track a validation metric without depending on a PyTorch scheduler."""

    def __init__(self, mode="min", factor=0.5, lr_patience=2,
                 stop_patience=6, min_index=0, min_lr=1e-6,
                 threshold=0.0):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if not 0.0 < factor < 1.0:
            raise ValueError("factor must be between 0 and 1")
        if lr_patience < 1 or stop_patience < 1:
            raise ValueError("patience values must be positive")
        self.mode = mode
        self.factor = float(factor)
        self.lr_patience = int(lr_patience)
        self.stop_patience = int(stop_patience)
        self.min_index = int(min_index)
        self.min_lr = float(min_lr)
        self.threshold = float(threshold)
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.bad_evals = 0
        self.bad_since_lr_drop = 0
        self.stop_bad_evals = 0

    def _improved(self, metric):
        if self.mode == "min":
            return metric < self.best_metric - self.threshold
        return metric > self.best_metric + self.threshold

    def step(self, metric, index, optimizers: Iterable):
        metric = float(metric)
        improved = self._improved(metric)
        if improved:
            self.best_metric = metric
            self.bad_evals = 0
            self.bad_since_lr_drop = 0
            self.stop_bad_evals = 0
        else:
            self.bad_evals += 1
            self.bad_since_lr_drop += 1
            if int(index) >= self.min_index:
                self.stop_bad_evals += 1
            else:
                self.stop_bad_evals = 0

        reduce_lr = self.bad_since_lr_drop >= self.lr_patience
        optimizers = list(optimizers)
        if reduce_lr:
            changed = False
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    old_lr = float(group["lr"])
                    group_min_lr = float(group.get("min_lr", self.min_lr))
                    new_lr = max(group_min_lr, old_lr * self.factor)
                    group["lr"] = new_lr
                    changed = changed or new_lr < old_lr
            reduce_lr = changed
            self.bad_since_lr_drop = 0

        current_lr = min(
            (float(group["lr"]) for optimizer in optimizers
             for group in optimizer.param_groups),
            default=0.0,
        )
        stop = self.stop_bad_evals >= self.stop_patience
        return PlateauDecision(
            improved=improved,
            reduce_lr=reduce_lr,
            stop=stop,
            best_metric=self.best_metric,
            bad_evals=self.bad_evals,
            current_lr=current_lr,
        )

    def state_dict(self):
        return {
            "mode": self.mode,
            "factor": self.factor,
            "lr_patience": self.lr_patience,
            "stop_patience": self.stop_patience,
            "min_index": self.min_index,
            "min_lr": self.min_lr,
            "threshold": self.threshold,
            "best_metric": self.best_metric,
            "bad_evals": self.bad_evals,
            "bad_since_lr_drop": self.bad_since_lr_drop,
            "stop_bad_evals": self.stop_bad_evals,
        }

    def load_state_dict(self, state):
        for key in self.state_dict():
            if key in state:
                setattr(self, key, state[key])


def add_plateau_args(parser, *, min_index_name, min_index_default=0,
                     stop_patience=6, min_lr=1e-6):
    parser.add_argument(
        "--lr-policy", choices=("fixed", "plateau"), default="fixed",
        help="Use legacy fixed milestones or validation-driven decay.",
    )
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--plateau-min-lr", type=float, default=min_lr)
    parser.add_argument("--early-stop-patience", type=int, default=stop_patience)
    parser.add_argument(
        f"--{min_index_name}", dest=min_index_name.replace("-", "_"),
        type=int, default=min_index_default,
    )
    return parser
