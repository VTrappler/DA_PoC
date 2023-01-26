import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Callable
import tqdm
import scipy.sparse.linalg as sla
import scipy.optimize
from copy import deepcopy

plt.style.use("seaborn-notebook")
from DA_PoC.dynamical_systems.examples.lorenz96 import Lorenz96Model
from DA_PoC.dynamical_systems.examples.lorenz63 import Lorenz63Model

# Let us set a rng for reproducibility
rng = np.random.default_rng(seed=93)


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

    def forward_model(self, x: np.ndarray) -> np.ndarray:
        """Integrates the model over the whole assimilation window"""
        # return (
        #     self.H(self.lorenz_model.integrate(0, x, self.n_total_obs - 1)[1])
        # ).flatten()
        return (
            self.H(self.lorenz_model.integrate(0, x, self.n_total_obs)[1])
        ).flatten()

    def data_misfit(self, x: np.ndarray) -> np.ndarray:
        try:
            return self.forward_model(x) - self.H(self.obs).flatten()
        except:
            raise RuntimeError("Observations not set")

    def cost_function(self, x: np.ndarray) -> float:
        """computes the cost functions with respect to the observations"""
        diff = self.data_misfit(x)
        return 0.5 * diff.T @ diff

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
        return self.adjoint_operator(x).matvec(self.data_misfit(x))

    def gauss_newton_matrix(self, x: np.ndarray) -> np.ndarray:
        """Returns the Gauss Newton matrix G^*G"""
        tlm = self.lorenz_model.construct_tlm_matrix(0, x, self.n_total_obs).reshape(
            -1, self.state_dimension
        )
        return tlm.T @ tlm


n = 100


def burn_model(burn=500):
    x = np.random.normal(size=n) * 5
    lorenz = LorenzWrapper(n)
    lorenz.n_total_obs = burn
    lorenz.H = lambda x: x
    burn_in = lorenz.forward_model(x).reshape(n, -1)
    x0_t = burn_in[:, -1]
    return x0_t


x0_t = burn_model()

lorenz = LorenzWrapper(n)
lorenz.H = lambda x: x
nobs = 10
lorenz.n_total_obs = nobs
obs = lorenz.forward_model(x0_t) + 1.0 * np.random.normal(
    size=(n * (1 + lorenz.n_total_obs))
)
lorenz.obs = obs

# GN = lorenz.gauss_newton_matrix(x)
# plt.imshow(GN)


from DA_PoC.common.numericalmodel import NumericalModel
from DA_PoC.common.observation_operator import (
    RandomObservationOperator,
    IdentityObservationOperator,
)


def create_lorenz_model_observation(n, m, obs_operator, test=True):
    # obs_operator = IdentityObservationOperator(m, m)
    # m = n * (nobs + 1)
    l_model = NumericalModel(n, m)
    l_model.set_obs(obs_operator(lorenz.obs))
    l_model.set_forward(lambda x: obs_operator(lorenz.forward_model(x)))
    l_model.set_observation_operator(obs_operator)

    l_model.set_tangent_linear(
        lambda x: lorenz.tangent_linear_operator(x).matmat(np.eye(n))
    )
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

## Identity observation -----

# print(f"Identity observation operator ---------------")
# m = n * (nobs + 1)

# id_obs_operator = IdentityObservationOperator(m, m)
# l_model_id = create_lorenz_model_observation(n, m, id_obs_operator)


## Alternative observation -----
print(f"Random observation operator   ---------------")
m = n * (nobs + 1)

random_obs_operator = RandomObservationOperator(m, m, 0.8, 1 / 440.0)
plt.imshow(random_obs_operator.H)
# m = n * (nobs + 1)
l_model_randobs = create_lorenz_model_observation(n, m, random_obs_operator, test=False)


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


def get_next_observations(x_init, sigsq=3, nobs=nobs):
    lorenz.n_total_obs = nobs
    forw = lorenz.forward_model(x_init)
    obs = forw + sigsq * np.random.normal(size=(n * (nobs + 1)))
    truth = forw.reshape(n, nobs + 1)
    x_t = truth[:, -1]
    return obs.reshape(n, nobs + 1), x_t, truth


def get_next_observations(x_init, modsigsq=0.5, obssigsq=3, nobs=nobs):
    lorenz.n_total_obs = nobs
    truth = np.empty((n, nobs + 1))
    curr_state = x_init
    truth[:, 0] = curr_state
    for i in range(nobs):
        curr_state = lorenz.lorenz_model.integrate(0, curr_state, 1)[1][
            :, 1
        ] + modsigsq * np.random.normal(size=(n))
        truth[:, i + 1] = curr_state
    obs = truth + obssigsq * np.random.normal(size=(n, (nobs + 1)))
    x_t = truth[:, -1]
    return obs, x_t, truth


def data_assimilation(
    l_model, obs_operator, n_cycle, n_outer, n_inner=n, prec=None, plot=True
):
    x0_t = burn_model()
    x_t = x0_t
    t = np.arange(nobs)
    truth_full = np.empty((n, nobs * n_cycle))
    analysis_full = np.empty((n, nobs * n_cycle))
    obs_full = np.empty((n, nobs * n_cycle))
    n_iter_innerloop = []

    cost_outerloop = []
    sp_optimisation = []
    x0_optim = x_t + np.random.normal(size=n)
    quad_errors = []
    if plot:
        plt.figure(figsize=(10, 2 * n))
    for i_cycle in range(n_cycle):
        obs, x_t, truth = get_next_observations(x_t)
        truth = truth[:, 1:]

        l_model.set_obs(obs_operator(obs.reshape(-1)))
        (
            gn_x,
            gn_fun,
            n_iter_inner,
            cost_outer,
            cost_inner,
            quad_error,
        ) = l_model.GNmethod(
            x0_optim, n_outer=n_outer, n_inner=n_inner, verbose=True, prec=prec
        )
        n_iter_innerloop.append(n_iter_inner)
        cost_outerloop.append(cost_outer)
        # sp_optim = scipy.optimize.minimize(l_model.cost_function, x0=x0_optim)
        # sp_optimisation.append(sp_optim.fun)
        quad_errors.append(quad_error)
        analysis = l_model.forward(gn_x).reshape(n, -1)[:, 1:]
        # print(f"{analysis.shape=}")
        x0_optim = analysis[:, -1]
        t_cycle = t + i_cycle * nobs
        analysis_full[:, t_cycle] = analysis
        truth_full[:, t_cycle] = truth
        obs_full[:, t_cycle] = obs[:, 1:]
        if plot:
            for i in range(n):
                plt.subplot(n, 1, i + 1)
                plt.plot(t_cycle, obs[i, 1:], "r.")
                plt.plot(t_cycle, analysis[i, :], color="green")
                plt.plot(t_cycle, truth[i, :], color="black")
                plt.scatter(t_cycle[-1], x_t[i])
        # plt.scatter(t_cycle[0], x0_t[i])
    if plot:
        plt.tight_layout()
        plt.show()
    return {
        "truth_full": truth_full,
        "analysis_full": analysis_full,
        "obs_full": obs_full,
        "n_iter_innerloop": n_iter_innerloop,
        "cost_outerloop": cost_outerloop,
        "sp_optimisation": sp_optimisation,
        "quad_errors": quad_errors,
        "l_model": l_model,
    }


def diagnostic_plots(DA, title):
    truth_full, obs_full, analysis_full = (
        DA["truth_full"],
        DA["obs_full"],
        DA["analysis_full"],
    )
    plt.subplot(1, 3, 1)
    plt.plot(
        ((truth_full - obs_full) ** 2).mean(0), label="Observation error (wrt obs)"
    )
    plt.plot(
        ((analysis_full - truth_full) ** 2).mean(0), label="Analysis error (wrt truth)"
    )
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.hist(((obs_full - truth_full) ** 2).mean(0), alpha=0.5, density=True)
    plt.hist(((analysis_full - truth_full) ** 2).mean(0), alpha=0.5, density=True)

    iter_CG = np.array(DA["n_iter_innerloop"]).T
    plt.subplot(1, 3, 3)
    plt.plot(np.arange(n_outer) + 1, iter_CG, color="gray", alpha=0.1)
    plt.boxplot(iter_CG.T)
    plt.xlabel(r"# of outer loop")
    plt.title(f"Number of CG iterations")
    # plt.axhline(n, color="red")
    plt.suptitle(f"{title}")
    plt.show()


n_cycle = 5
n_outer = 5
n_inner = 50
# np.random.set_state(7071522)
DA_vanilla = data_assimilation(
    l_model_randobs,
    random_obs_operator,
    n_cycle,
    n_outer,
    n_inner,
    prec=None,
    plot=False,
)
diagnostic_plots(DA_vanilla, "vanilla")


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


plot_quadratic_function(DA_vanilla, n_cycle, "vanilla")


DA_jacobi = data_assimilation(
    l_model_randobs,
    random_obs_operator,
    n_cycle,
    n_outer,
    n_inner,
    prec="jacobi",
    plot=False,
)
diagnostic_plots(DA_jacobi, "jacobi")
plot_quadratic_function(DA_jacobi, n_cycle, "jacobi")

# div = [2, 4, 8]
# l_model.r = n // 4
DA_LMP = {}
for r_ in [30, 20, 10, 5]:
    l_model_randobs.r = r_
    DA_LMP[r_] = data_assimilation(
        l_model_randobs,
        random_obs_operator,
        n_cycle,
        n_outer,
        n_inner,
        prec="spectralLMP",
        plot=False,
    )
    diagnostic_plots(DA_LMP[r_], f"spectralLMP, r={r_}")
    plot_quadratic_function(DA_LMP[r_], n_cycle, f"spectralLMP, r={r_}")


obs = DA_vanilla["obs_full"]
tru = DA_vanilla["truth_full"]

# U, S, VT = scipy.linalg.svd(obs)

# plt.plot(1 - S ** 2 / sum(S ** 2))
# plt.axhline(0.9)
# plt.axhline(0.95)
# m_ = np.array(DA_vanilla["n_iter_innerloop"]).T.mean(1)
# s_ = np.array(DA_vanilla["n_iter_innerloop"]).T.std(1)
# max_ = np.array(DA_vanilla["n_iter_innerloop"]).T.max(1)
# min_ = np.array(DA_vanilla["n_iter_innerloop"]).T.min(1)


def plot_innerloopiter(DA_dict, color, label):
    ninnerloop = DA_dict["n_iter_innerloop"]
    m_, s_, max_, min_ = (
        func(np.asarray(ninnerloop).T, 1) for func in [np.mean, np.std, np.max, np.min]
    )
    x_ = np.arange(n_outer) + 1
    plt.plot(x_, m_, color=color, alpha=1, label=label)
    plt.fill_between(x_, m_ + s_, m_ - s_, alpha=0.15, color=color)
    plt.plot(x_, max_, color=color, ls=":")
    plt.plot(x_, min_, color=color, ls=":")


plot_innerloopiter(DA_vanilla, "blue", "vanilla")
plot_innerloopiter(DA_jacobi, "red", "jacobi")
for r_, c_ in zip(
    [30, 20, 10, 5],
    ["turquoise", "magenta", "orange", "green", "black"],
):
    plot_innerloopiter(DA_LMP[r_], c_, f"LMP, r={r_}")
plt.legend()
plt.grid()
plt.show()


for r_ in [30, 20, 10, 5]:
    plt.figure()
    plot_quadratic_function(DA_LMP[r_], n_cycle, r_)


def plot_optim_gap(DA):
    # plt.subplot(1, 2, 1)
    plt.plot(np.asarray(DA["cost_outerloop"]) - np.asarray(DA["sp_optimisation"]))
    plt.show()
