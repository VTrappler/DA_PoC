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

rng = np.random.default_rng()


def generate_observations(
    truth, i, state_dimension, sigsqobs, period_assim
) -> Tuple[float, np.ndarray]:
    """Generates the "truth" state vector. We perturbate them all

    :param i: dummy variable, indicating the time step
    :type i: int
    :return:
    :rtype: [type]
    """
    truth.forward(period_assim)
    y = truth.state_vector[:, -1] + rng.multivariate_normal(
        np.zeros(state_dimension), np.eye(state_dimension)
    ) * np.sqrt(sigsqobs)
    return truth.t[-1], y


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

    @property
    def background(self):
        return self._background

    @background.setter
    def background(self, value):
        assert (value is None) or (len(value) == self.state_dimension)
        self._background = value

    def create_and_burn_truth(self, burn=2000, x0=None):
        if x0 is None:
            x0 = np.zeros(self.state_dimension)
            x0[0] = 1
        self.truth = self.lorenz_model()
        self.truth.set_initial_state(-burn * self.truth.dt, x0)
        self.truth.forward(burn)

    def generate_obs(self, n_total_obs: int = 100, H: Callable = lambda x: x):
        """Generate n_total_obs observations

        :param n_total_obs: _description_, defaults to 100
        :type n_total_obs: int, optional
        :param H: _description_, defaults to lambdax:x
        :type H: _type_, optional
        """
        self.n_total_obs = n_total_obs
        obs = np.empty((self.state_dimension, self.n_total_obs))
        time_obs = np.empty(self.n_total_obs)
        self.initial_state = self.truth.state_vector[:, -1] + rng.multivariate_normal(
            np.zeros(self.state_dimension), cov=np.eye(self.state_dimension)
        )

        for i in range(self.n_total_obs):
            generated = generate_observations(
                self.truth, i, self.state_dimension, 1, self.period_assim
            )
            time_obs[i], obs[:, i] = generated
        self.obs = obs
        self.time_obs = time_obs
        self.H = H

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

    def forward_TLM(self, x: np.ndarray, return_base: bool = False):
        """Computes the forward and the tangent linear

        :param x: initial state vector
        :type x: np.ndarray
        :param return_base: Should the method return the forward, defaults to False
        :type return_base: bool, optional
        :return: Forward model (if return_bas) and Tangent linear
        :rtype: Tuple
        """
        tlm = self.lorenz_model.construct_tlm_matrix(0, x, self.n_total_obs - 1)
        if return_base:
            return self.forward_model(x), tlm
        else:
            return tlm

    def forward_steps(self, x: np.ndarray, nsteps: int):
        return self.lorenz_model.integrate(0, x, nsteps)[1]

    def gauss_newton_matrix(self, x: np.ndarray) -> np.ndarray:
        """Returns the Gauss Newton matrix G^*G"""
        tlm = self.lorenz_model.construct_tlm_matrix(0, x, self.n_total_obs).reshape(
            -1, self.state_dimension
        )
        if self.background_error_cov_inv is None:
            return tlm.T @ tlm
        else:
            return tlm.T @ tlm + self.background_error_cov_inv

    def get_next_observation(
        self, x_init: np.ndarray, model_error_sqrt: float, obs_error_sqrt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.state_dimension
        truth = np.empty((n, self.n_total_obs + 1))
        curr_state = x_init
        truth[:, 0] = curr_state
        for i in range(self.n_total_obs):
            curr_state = self.lorenz_model.integrate(0, curr_state, 1)[1][
                :, 1
            ] + model_error_sqrt * np.random.normal(size=(n))
            truth[:, i + 1] = curr_state
        obs = truth + obs_error_sqrt * np.random.normal(
            size=(n, (self.n_total_obs + 1))
        )
        x_t = truth[:, -1]
        return obs, x_t, truth

    def set_observations(
        self, nobs: int = 10, burn: int = 500, obs_error_sqrt: float = 1.0
    ) -> None:
        """Set instance attribute self.obs, by runnin the model with burn in time

        :param nobs: numbr of observed timesteps, defaults to 10
        :type nobs: int, optional
        :param burn: number of timesteps before recording, defaults to 500
        :type burn: int, optional
        :param obs_error_sqrt: std deviation of the observation error
        :type obs_error_sqrt: float, optional
        """
        x = np.random.normal(size=self.state_dimension) * 5
        self.n_total_obs = burn
        self.H = lambda x: x
        burn_in = self.forward_model(x).reshape(self.state_dimension, -1)
        x0_t = burn_in[:, -1]
        self.n_total_obs = nobs
        obs = self.forward_model(x0_t) + obs_error_sqrt * np.random.normal(
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
