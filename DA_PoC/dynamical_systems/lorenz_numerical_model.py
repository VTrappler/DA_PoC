import sys

# sys.path.append("../..")

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Callable
import tqdm
import scipy.sparse.linalg as sla
import scipy.optimize
from copy import deepcopy

plt.style.use("seaborn-v0_8")
from .examples.lorenz96 import Lorenz96Model
from .examples.lorenz63 import Lorenz63Model
from DA_PoC.common.numerical_model import NumericalModel
from DA_PoC.common.observation_operator import (
    RandomObservationOperator,
    IdentityObservationOperator,
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
        return (
            self.H(self.lorenz_model.integrate(0, x, nsteps)[1])
        ).flatten()

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


# x0_t = burn_model()


# GN = lorenz.gauss_newton_matrix(x)
# plt.imshow(GN)


def create_lorenz_model_observation(lorenz, m, obs_operator, test=True):
    # obs_operator = IdentityObservationOperator(m, m)
    # m = n * (nobs + 1)
    n = lorenz.state_dimension
    # print(n)
    # print(f"{lorenz.obs.shape=}")
    # print(f"{lorenz.H=}")
    l_model = NumericalModel(n, m)
    l_model.background = lorenz.background
    l_model.background_error_cov_inv = lorenz.background_error_cov_inv
    l_model.set_obs(obs_operator(lorenz.obs))
    l_model.set_forward(lambda x: obs_operator(lorenz.forward_model(x)))
    l_model.set_observation_operator(obs_operator)
    l_model.nobs = lorenz.n_total_obs
    l_model.set_tangent_linear(
        lambda x: lorenz.tangent_linear_operator(x).matmat(np.eye(n))
    )
    x0_t = np.random.normal(size=n)
    print(f"{lorenz.cost_function(x0_t)=}")
    print(f"{l_model.cost_function(x0_t)=}")

    if test:
        l_model.tests_consistency()

        x0 = np.zeros(n)

        sp_opt = scipy.optimize.minimize(l_model.cost_function, x0)
        sp_x, sp_fun = sp_opt.x, sp_opt.fun
        gn_x, gn_fun, n_iter, cost_outer, cost_inner, quad_error = l_model.GNmethod(
            5 * np.random.normal(size=n),
            n_outer=10,
            n_inner=50,
            verbose=True,
            prec=None,
        )

        print(f"{sp_fun=}, {gn_fun=}")
    return l_model


## Get last value ---------
# m = 2 * n

# random_obs_operator = RandomObservationOperator(n, m, 0.5)
# random_obs_operator = IdentityObservationOperator(n, m)

# # m = n * (nobs + 1)
# l_model = NumericalModel(n, m)
# l_model.set_obs(random_obs_operator(lorenz.obs.reshape(n, -1)[:, -1]))
# l_model.set_forward(
#     lambda x: random_obs_operator(lorenz.forward_model(x).reshape(n, -1)[:, -1])
# )
# l_model.set_observation_operator(random_obs_operator)

# l_model.set_tangent_linear(
#     lambda x: lorenz.tangent_linear_operator(x)
#     .matmat(np.eye(n))
#     .reshape(n, nobs + 1, n)[:, -1, :]
# )

# print(f"{lorenz.cost_function(x0_t)=}")
# print(f"{l_model.cost_function(x0_t)=}")


# l_model.tests_consistency()


## Alternative observation -----


def quad_function_plot(quad_error, cost_outer, color):
    last = 0
    plt.scatter(last, cost_outer[0], color=color)
    for i, inner_it in enumerate(quad_error):
        outer = np.arange(len(inner_it)) + last
        plt.plot(outer, inner_it, color=color)
        last = outer[-1]
        plt.scatter(last, cost_outer[i + 1], color=color)


# obs = obs.reshape(n, nobs)

# analys = l_model.forward(sp_x).reshape(n, nobs)
# for i in range(n):
#     plt.subplot(n, 1, i + 1)
#     plt.plot(obs[i, :], "r.")
#     plt.plot(analys[i, :])
# plt.show()


# plt.imshow(obs)


# def get_next_observations(x_init, sigsq=3, nobs=nobs):
#     lorenz.n_total_obs = nobs
#     forw = lorenz.forward_model(x_init)
#     obs = forw + sigsq * np.random.normal(size=(n * (nobs + 1)))
#     truth = forw.reshape(n, nobs + 1)
#     x_t = truth[:, -1]
#     return obs.reshape(n, nobs + 1), x_t, truth

# def get_next_observations(x_init, modsigsq=0.5, obssigsq=3, nobs=nobs):
#     lorenz.n_total_obs = nobs
#     truth = np.empty((n, nobs + 1))
#     curr_state = x_init
#     truth[:, 0] = curr_state
#     for i in range(nobs):
#         curr_state = lorenz.lorenz_model.integrate(0, curr_state, 1)[1][
#             :, 1
#         ] + modsigsq * np.random.normal(size=(n))
#         truth[:, i + 1] = curr_state
#     obs = truth + obssigsq * np.random.normal(size=(n, (nobs + 1)))
#     x_t = truth[:, -1]
#     return obs, x_t, truth





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
    # np.asarray(quad_error).flatten() / quad_error[-1][-1]


# plot_quadratic_function(DA_vanilla, n_cycle, "vanilla")


# DA_jacobi = data_assimilation(
#     l_model_randobs,
#     random_obs_operator,
#     n_cycle,
#     n_outer,
#     n_inner,
#     prec="jacobi",
#     plot=False,
# )


# diagnostic_plots(DA_jacobi, "jacobi")
# plot_quadratic_function(DA_jacobi, n_cycle, "jacobi")

# # div = [2, 4, 8]
# # l_model.r = n // 4
# DA_LMP = {}
# for r_ in [30, 20, 10, 5]:
#     l_model_randobs.r = r_
#     DA_LMP[r_] = data_assimilation(
#         l_model_randobs,
#         random_obs_operator,
#         n_cycle,
#         n_outer,
#         n_inner,
#         prec="spectralLMP",
#         plot=False,
#     )
#     diagnostic_plots(DA_LMP[r_], f"spectralLMP, r={r_}")
#     plot_quadratic_function(DA_LMP[r_], n_cycle, f"spectralLMP, r={r_}")


# obs = DA_vanilla["obs_full"]
# tru = DA_vanilla["truth_full"]

# # U, S, VT = scipy.linalg.svd(obs)

# # plt.plot(1 - S ** 2 / sum(S ** 2))
# # plt.axhline(0.9)
# # plt.axhline(0.95)
# # m_ = np.array(DA_vanilla["n_iter_innerloop"]).T.mean(1)
# # s_ = np.array(DA_vanilla["n_iter_innerloop"]).T.std(1)
# # max_ = np.array(DA_vanilla["n_iter_innerloop"]).T.max(1)
# # min_ = np.array(DA_vanilla["n_iter_innerloop"]).T.min(1)




# plot_innerloopiter(DA_vanilla, "blue", "vanilla")
# plot_innerloopiter(DA_jacobi, "red", "jacobi")
# for r_, c_ in zip(
#     [30, 20, 10, 5],
#     ["turquoise", "magenta", "orange", "green", "black"],
# ):
#     plot_innerloopiter(DA_LMP[r_], c_, f"LMP, r={r_}")
# plt.legend()
# plt.grid()
# plt.show()


# for r_ in [30, 20, 10, 5]:
#     plt.figure()
#     plot_quadratic_function(DA_LMP[r_], n_cycle, r_)


# def plot_optim_gap(DA):
#     # plt.subplot(1, 2, 1)
#     plt.plot(np.asarray(DA["cost_outerloop"]) - np.asarray(DA["sp_optimisation"]))
#     plt.show()
if __name__ == "__main__":
    ## Identity observation -----

    n = 100
    nobs = 10
    m = n * (nobs + 1)
    lorenz = LorenzWrapper(n)
    lorenz.H = lambda x: x
    lorenz.n_total_obs = nobs
    lorenz.set_observations(nobs)
    B_inv_half = np.random.normal(size=(n, n))
    B_inv = B_inv_half.T @ B_inv_half
    lorenz.background_error_cov_inv = B_inv
    lorenz.background = np.random.normal(size=n)

    id_obs_operator = IdentityObservationOperator(m, m)
    l_model_id = create_lorenz_model_observation(lorenz, m, id_obs_operator, test=False)
    l_model_id.background = lorenz.background
    l_model_id.background_error_cov_inv = lorenz.background_error_cov_inv
    l_model_id.tests_consistency()
