#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt

# from ..solvers.solvers import RK4_step, integrate_step, integration_RK4_tlm
from ..solvers.solvers import (
    RK4_step,
    integrate_step,
    integration_RK4_tlm,
)

# from .abstractdynamicalmodel import DynamicalModel as Model
from .abstractdynamicalmodel import (
    DynamicalModel as Model,
)


class Lorenz96Model(Model):
    dim = 40
    solver = RK4_step
    dt = 0.02

    @classmethod
    def set_dim(cls, dim):
        cls.dim = dim

    @classmethod
    def dotfunction(cls, t: float, x: np.ndarray) -> np.ndarray:
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        d = np.zeros(cls.dim)
        for i in range(cls.dim):
            d[i] = (x[(i + 1) % cls.dim] - x[i - 2]) * x[i - 1] - x[i] + 8
        return d

    @classmethod
    def TLM(cls, t: float, x: np.ndarray) -> np.ndarray:
        """Computes the TLM at a specific point

        :param t: time
        :type t: float
        :param x: current state vector
        :type x: np.ndarray
        :return: Tangent Linear Model
        :rtype: np.ndarray
        """
        jac = np.diag(-np.ones(cls.dim))  # Set the diagonal
        for i in range(cls.dim):
            jac[i, i - 2] = -x[i - 1]
            jac[i, i - 1] = x[(i + 1) % cls.dim] - x[i - 2]
            jac[i, i] = -1
            jac[i, (i + 1) % cls.dim] = x[i - 1]
        return jac

    @classmethod
    def integrate(cls, t0, x0, Nsteps):
        return integrate_step(cls.solver, cls.dotfunction, t0, x0, cls.dt, Nsteps)

    @classmethod
    def integrate_tlm(
        cls, t0: float, x0: np.ndarray, dx0: np.ndarray, Nsteps: int
    ) -> np.ndarray:
        t = np.arange(t0, cls.dt * (Nsteps + 1), cls.dt)
        x, dx = integration_RK4_tlm(cls.dotfunction, cls.TLM, t, x0, dx0, cls.dt)
        return t, x, dx

    @classmethod
    def construct_tlm_matrix(cls, t0: float, x0: np.ndarray, Nsteps: int) -> np.ndarray:
        """Construct the TL matrix of dimension

        :param t0: _description_
        :type t0: float
        :param x0: _description_
        :type x0: np.ndarray
        :param Nsteps: _description_
        :type Nsteps: int
        :return: _description_
        :rtype: np.ndarray
        """
        t = np.arange(t0, cls.dt * (Nsteps + 1), cls.dt)
        jacobian_matrix = np.empty((cls.dim, (Nsteps + 1), cls.dim))
        for i in range(cls.dim):
            ei = np.zeros(cls.dim)
            ei[i] = 1
            _, dei = integration_RK4_tlm(cls.dotfunction, cls.TLM, t, x0, ei, cls.dt)
            jacobian_matrix[:, :, i] = np.array(dei).T
        return jacobian_matrix


def main():
    Ndim = 200
    Lorenz96Model.dim = Ndim
    lorenz40 = Lorenz96Model()
    x0 = np.random.normal(0, 1, lorenz40.dim)
    lorenz40.set_initial_state(0, x0)
    lorenz40.forward(5000)
    plt.imshow(lorenz40.state_vector, aspect="auto")
    plt.show()



def test_tlm():
    Ndim = 50
    nobs = 10
    Lorenz96Model.dim = Ndim
    x0 = np.random.normal(0, 1, Ndim)

    def finite_diff(traj, x, dx, eps):
        _, traj2 = Lorenz96Model.integrate(0, x0=x + eps * dx, Nsteps=nobs)
        return (traj2 - traj) / eps


    t, traj = Lorenz96Model.integrate(0, x0=x0, Nsteps=nobs)
    dx0 = np.random.normal(0, 1, Ndim)
    t, x, dx = Lorenz96Model.integrate_tlm(0, x0=x0, dx0=dx0, Nsteps=nobs)
    dx = np.array(dx).T
    print(traj.shape)
    print(dx.shape)
    for eps in np.logspace(-1, -10, 10, base=10):
        fd = finite_diff(traj, x0, dx0, eps)
        print((dx**2).sum()/(fd**2).sum())


    jac = Lorenz96Model.construct_tlm_matrix(0, x0, Nsteps=nobs)


if __name__ == "__main__":
    main()
