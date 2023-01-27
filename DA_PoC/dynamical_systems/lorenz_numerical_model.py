import sys

# sys.path.append("../..")

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Callable
import tqdm
import scipy.sparse.linalg as sla
import scipy.optimize
from copy import deepcopy
import time

plt.style.use("seaborn-v0_8")
from .examples.lorenz96 import Lorenz96Model
from .examples.lorenz63 import Lorenz63Model
from DA_PoC.common.numerical_model import NumericalModel
from DA_PoC.common.observation_operator import (
    RandomObservationOperator,
    IdentityObservationOperator,
    ObservationOperator,
)


class LorenzWrapper:
    def __init__(self, state_dimension: int) -> None:
        self.state_dimension = state_dimension
        if state_dimension == 3:
            self.lorenz_model = Lorenz63Model
        else:
            self.lorenz_model = Lorenz96Model
            self.lorenz_model.dim = state_dimension
        self.period_assim = 1
        self.lorenz_model.dt = 0.02
        # 1 time unit = 120 hours
        # 0.05 time unit = 6 hours
        # 0.2 time unit = 24 hours
        self.background_error_cov_inv = None
        self.background = None

    def forward_model(self, x: np.ndarray, nsteps=None) -> np.ndarray:
        """Integrates the model over the whole assimilation window"""
        # return (
        #     self.H(self.lorenz_model.integrate(0, x, self.n_total_obs - 1)[1])
        # ).flatten()
        if nsteps is None:
            nsteps = self.n_total_obs
        return (self.H(self.lorenz_model.integrate(0, x, nsteps)[1])).flatten()

    def data_misfit(self, x: np.ndarray) -> np.ndarray:
        try:
            return self.forward_model(x) - self.H(self.obs).flatten()
        except Exception as e:
            raise RuntimeError(f"Observations not set, {e}")

    def background_cost(self, x: np.ndarray) -> float:
        try:
            return (
                (x - self.background).T
                @ self.background_error_cov_inv
                @ (x - self.background)
            )
        except NameError as e:
            raise RuntimeError(
                f"Background error covariance matrix or background value not set"
            )

    def cost_function(self, x: np.ndarray) -> float:
        """computes the cost functions with respect to the observations"""
        diff = self.data_misfit(x)
        if self.background_error_cov_inv is None:
            return 0.5 * diff.T @ diff
        else:
            return 0.5 * diff.T @ diff + 0.5 * self.background_cost(x)

    def gradient_fd(self, x: np.ndarray) -> np.ndarray:
        """Computes the gradient using finite differences"""
        grad = np.zeros(self.state_dimension)
        e = np.zeros_like(x)
        base = self.cost_function(x)
        for i in range(self.state_dimension):
            e[i] = self.eps
            grad[i] = (self.cost_function(x + e) - base) / self.eps
            e[i] = 0
        return grad

    def tangent_linear_operator(self, x: np.ndarray) -> sla.LinearOperator:
        """Returns the tangent linear as a LinearOperator

        :param x: Point where the linearization is performed
        :type x: np.ndarray
        :return: Tangent Linear model
        :rtype: LinearOperator
        """
        return sla.aslinearoperator(
            self.lorenz_model.construct_tlm_matrix(0, x, self.n_total_obs).reshape(
                -1, self.state_dimension
            )
        )

    def adjoint_operator(self, x: np.ndarray) -> sla.LinearOperator:
        """Returns the adjoint as a linear operator"""
        return self.tangent_linear_operator(x).adjoint()

    def gradient_adjoint(self, x: np.ndarray) -> np.ndarray:
        """Returns the gradient obtained using the gradient G^*(g(x) - y)"""
        if self.background_error_cov_inv is None:
            prior_lin = 0
        else:
            prior_lin = self.background_error_cov_inv @ (x - self.background)
        return self.adjoint_operator(x).matvec(self.data_misfit(x)) + prior_lin

    def gauss_newton_matrix(self, x: np.ndarray) -> np.ndarray:
        """Returns the Gauss Newton matrix G^*G"""
        tlm = self.lorenz_model.construct_tlm_matrix(0, x, self.n_total_obs).reshape(
            -1, self.state_dimension
        )
        if self.background_error_cov_inv is None:
            return tlm.T @ tlm
        else:
            return tlm.T @ tlm + self.background_error_cov_inv

    def set_observations(self, nobs: int = 10, burn: int = 500) -> None:
        x = np.random.normal(size=self.state_dimension) * 5
        self.n_total_obs = burn
        self.H = lambda x: x
        burn_in = self.forward_model(x).reshape(self.state_dimension, -1)
        x0_t = burn_in[:, -1]
        self.n_total_obs = nobs
        obs = self.forward_model(x0_t) + 1.0 * np.random.normal(
            size=(self.state_dimension * (1 + self.n_total_obs))
        )
        self.obs = obs


# n = 10
# nobs = 10


def burn_model(n, burn=500):
    x = np.random.normal(size=n) * 5
    lorenz = LorenzWrapper(n)
    lorenz.n_total_obs = burn
    lorenz.H = lambda x: x
    burn_in = lorenz.forward_model(x).reshape(n, -1)
    x0_t = burn_in[:, -1]
    return x0_t



def create_lorenz_model_observation(
    lorenz: LorenzWrapper,
    obs_operator: ObservationOperator,
    test_consistency=True,
) -> NumericalModel:
    """Create a NumericalModel instance implementing the Lorenz system with given observation operator

    :param lorenz: Lorenz system
    :type lorenz: LorenzWrapper
    :param obs_operator: Chosen observation operator
    :type obs_operator: ObservationOperator

    :param test_consistency: Should we perform the test to verify the TLM, adj etc, defaults to True
    :type test_consistency: bool, optional
    :return: The numerical model associated with the G = H o M
    :rtype: NumericalModel
    """
    # obs_operator = IdentityObservationOperator(m, m)
    # m = n * (nobs + 1)
    n = lorenz.state_dimension
    dim_observation = obs_operator.m
    # print(n)
    # print(f"{lorenz.obs.shape=}")
    # print(f"{lorenz.H=}")
    numerical_model_lorenz = NumericalModel(n, dim_observation)
    numerical_model_lorenz.background = lorenz.background
    numerical_model_lorenz.background_error_cov_inv = lorenz.background_error_cov_inv
    numerical_model_lorenz.set_observation_operator(obs_operator)
    numerical_model_lorenz.set_obs(obs_operator(lorenz.obs))
    numerical_model_lorenz.set_forward(lambda x: lorenz.forward_model(x))
    numerical_model_lorenz.nobs = lorenz.n_total_obs
    numerical_model_lorenz.set_tangent_linear(
        lambda x: lorenz.tangent_linear_operator(x).matmat(np.eye(n))
    )
    x0_t = np.random.normal(size=n)
    print(f"{lorenz.cost_function(x0_t)=}")
    print(f"{numerical_model_lorenz.cost_function(x0_t)=}")

    if test_consistency:
        numerical_model_lorenz.tests_consistency()
        x0 = np.zeros(n)
        print(f"Comparison of Scipy method and Gauss Newton/CG method\n")
        print(f"- scipy")
        before = time.time()
        sp_opt = scipy.optimize.minimize(numerical_model_lorenz.cost_function, x0)
        print(f" - time elapsed: {time.time() - before}s")
        sp_x, sp_fun = sp_opt.x, sp_opt.fun
        print(f"- GNmethod")
        before = time.time()
        (
            gn_x,
            gn_fun,
            n_iter,
            cost_outer,
            cost_inner,
            quad_error,
            inner_res,
        ) = numerical_model_lorenz.GNmethod(
            1 * np.random.normal(size=n),
            n_outer=10,
            n_inner=50,
            verbose=True,
            prec=None,
        )
        print(f" - time elapsed: {time.time() - before}s")
        print(f"{sp_fun=}, {gn_fun=}")
    return numerical_model_lorenz



def quad_function_plot(quad_error, cost_outer, color):
    last = 0
    plt.scatter(last, cost_outer[0], color=color)
    for i, inner_it in enumerate(quad_error):
        outer = np.arange(len(inner_it)) + last
        plt.plot(outer, inner_it, color=color)
        last = outer[-1]
        plt.scatter(last, cost_outer[i + 1], color=color)



def plot_quadratic_function(DA, n_cycle, title):
    for i, (quad_error, cost_outer) in enumerate(
        zip(
            DA["quad_errors"],
            DA["cost_outerloop"],
        )
    ):
        col = plt.get_cmap()(i / n_cycle)
        quad_function_plot(quad_error, cost_outer, col)
    plt.ylim(top=50000)
    plt.title(title)
