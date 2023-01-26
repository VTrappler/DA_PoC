#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from ..solvers.solvers import integrate_step
from .abstractdynamicalmodel import DynamicalModel as Model


class NonLinearOscillatorModel(Model):
    dim = 2
    omega = 0.035
    lam = 3e-5
    xi = None
    dt = 1

    def __init__(self):
        pass

    @classmethod
    def step(cls, f: Callable, t: float, x: np.ndarray, dt: float) -> np.ndarray:
        """Computes the next state vector given the current one

        :param f: Callable, not used here
        :type f: Callable
        :param t: Current time (not used here)
        :type t: float
        :param x: Current state vector
        :type x: np.ndarray
        :param dt: time step
        :type dt: float
        :return: next state vector
        :rtype: np.ndarray
        """
        if cls.xi is None:
            xi = np.random.randn(1) * np.sqrt(0.0025)
        else:
            xi = cls.xi
        return np.array(
            [
                x[1],
                2 * x[1]
                - x[0]
                + (cls.omega ** 2) * x[1]
                - (cls.lam ** 2) * x[1] ** 3
                + xi[0],
            ]
        )

    @classmethod
    def integrate(cls, t0, x0, Nsteps):
        return integrate_step(cls.step, f=None, t0=t0, x0=x0, dt=cls.dt, Nsteps=Nsteps)


osci = NonLinearOscillatorModel()
osci.set_initial_state(0, np.array([0, 1]))
osci.forward(1000)
plt.plot(osci.state_vector[0, :])
plt.show()


def sample_observations(vector, spacing, shift):
    return vector[shift::spacing]


def main():
    N_iter = 5000
    oscillator = NonLinearOscillatorModel()
    oscillator.initial_state([0, 1])
    oscillator.step(N_iter)
    sigma2_obs = 49
    observations = oscillator.state_vector[:, 0] + np.random.randn(
        len(oscillator.state_vector)
    ) * np.sqrt(sigma2_obs)
    plt.plot(observations)
    plt.show()


if __name__ == "__main__":
    # main()
    pass
