#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from ..solvers.solvers import integrate_step
from .abstractdynamicalmodel import DynamicalModel as Model


class LinearModel(Model):
    """Linear model of the form
    x_{k+1} = Mx_k + w_k
    y_k = Hx_k + v_k
    """

    dt = 1

    def __init__(self, M, H, Q, R):
        self.M = M
        self.H = H
        self.Q = Q
        self.R = R

    def apply_model(self, t, x):
        """Linear model"""
        return self.M @ x

    def apply_observation(self, t, x):
        return self.H @ x

    def step(self, f, t, x, dt):
        return self.M @ x + np.random.multivariate_normal(
            np.zeros(len(self.Q)), cov=self.Q
        )

    def integrate(self, t0, x0, Nsteps):
        return integrate_step(self.step, None, t0, x0, self.dt, Nsteps)


def main():
    dim = 2
    x0 = np.random.normal(0, 1, size=dim).reshape(dim, 1)
    M = np.random.randint(10, size=dim ** 2).reshape(dim, dim)
    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1, 0], [0, 1]])
    H = np.array([[1, 0]])
    lm = LinearModel(M, H, Q, R)
    print(lm.step(0, x0))


if __name__ == "__main__":
    main()
