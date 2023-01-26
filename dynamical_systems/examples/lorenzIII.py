#!/usr/bin/env python
# coding: utf-8

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from ..solvers.solvers import RK4_step, integrate_step
from .abstractdynamicalmodel import DynamicalModel as Model


def phi(v, pm):
    return np.roll(v, pm) * (np.roll(v, -pm) - np.roll(v, 2 * pm))


def xdot(
    x: np.ndarray, u: np.ndarray, F: float = 15, h: int = 1, c: int = 10, b: int = 10
):
    return (
        phi(x, 1)
        + F
        - (h * c / b) * np.fromiter(map(sum, np.split(u, len(x))), dtype="float")
    )


def udot(x, u, h=1, c=10, b=10):
    return (c / b) * phi(b * u, -1) + (h * c / b) * x[np.arange(len(u)) // 10]


def lorenz2scale_dot(t: float, xu: np.ndarray) -> np.ndarray:
    """Computes the dot function of the two scales Lorenz model

    :param t: current time
    :type t: float
    :param xu: State vector (Slow and fast dynamics)
    :type xu: np.ndarray
    :return: the dot function of the model
    :rtype: np.ndarray
    """
    c = 10  # Time scale ratio
    b = 10  # Space scale ratio
    h = 1  # Coupling
    F = 15  # Forcing
    Nx = 36
    x, u = xu[:Nx], xu[Nx:]
    dx = xdot(x, u, F=F, h=h, c=c, b=b)
    du = udot(x, u, h=h, c=c, b=b)
    return np.concatenate([dx, du])


class Lorenz2scalesModel(Model):
    dt = None
    solver = RK4_step

    @classmethod
    def dotfunction(cls, t: float, xu: np.ndarray) -> np.ndarray:
        return lorenz2scale_dot(t, xu)

    @classmethod
    def integrate(
        cls, t0: float, x0: np.ndarray, Nsteps: int, verbose=True
    ) -> Tuple[float, np.ndarray]:
        return integrate_step(
            cls.solver, cls.dotfunction, t0, x0, cls.dt, Nsteps, verbose=verbose
        )

    def __init__(self, Nx=36) -> None:
        self.Nx = Nx
        self.Nu = 10 * Nx
        self.dim = self.Nx + self.Nu

    @classmethod
    def set_observation_operator(cls, H: callable) -> None:
        cls.H = H

    @classmethod
    def set_observation_errors(cls, obserr):
        cls.vk = obserr

    def observe(self):
        vk = self.vk()
        return self.H(self.state_vector) + vk


if __name__ == "__main__":
    Nx, Nu = 36, 360
    initial_state = np.random.normal(0, 1, size=(Nx + Nu))
    lorenzIII = Lorenz2scalesModel()
    dt = 0.001
    Lorenz2scalesModel.dt = dt
    lorenzIII.set_initial_state(0, initial_state)
    lorenzIII.forward(10000)

    lorenzIII_ = Lorenz2scalesModel()
    initial_state[380] += 0.00001
    lorenzIII_.set_initial_state(0, initial_state)
    lorenzIII_.forward(10000)

    plt.imshow(
        lorenzIII.state_vector,
        aspect="auto",
    )
    plt.show()
