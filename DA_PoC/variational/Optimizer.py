from typing import Any, Callable

import numpy as np


class Optimizer:
    def __init__(self, state_dimension, bounds) -> None:
        self.state_dimension = state_dimension
        self.bounds = bounds
        assert len(self.bounds) == self.state_dimension

    def run_optimization(self, fun: Callable, x0: np.ndarray) -> Any:
        return self.optimizer(fun, x0, self.bounds)
