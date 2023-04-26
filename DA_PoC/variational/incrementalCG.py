from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from ..common.numerical_model import NumericalModel
from ..common.observation_operator import ObservationOperator
from .VariationalMethod import VariationalMethod
from typing import Callable


def pad_ragged(list_of_arr):
    max_len = 0
    for arr in list_of_arr:
        if len(arr) > max_len:
            max_len = len(arr)
    array = np.empty((len(list_of_arr), max_len))
    for i, arr in enumerate(list_of_arr):
        array[i, :] = np.pad(
            arr, (0, max_len - len(arr)), "constant", constant_values=(np.nan)
        )
    return array


class Incremental4DVarCG(VariationalMethod):
    def __init__(
        self,
        state_dimension: int,
        bounds: np.ndarray,
        numerical_model: NumericalModel,
        observation_operator: ObservationOperator,
        x0_run: np.ndarray,
        x0_analysis: np.ndarray,
        get_next_observations: Callable,
        n_cycle: int,
        n_outer: int,
        n_inner: int,
        prec: dict = None,
        plot: bool = False,
        log_append: bool = False,
        save_all: bool = False,
    ) -> None:
        """Instantiate an Incremental4DVar method

        :param state_dimension: state dimension (n)
        :type state_dimension: int
        :param bounds: bounds of the state vector
        :type bounds: np.ndarray
        :param numerical_model: Numerical Model implementing all useful methods
        :type numerical_model: NumericalModel
        :param observation_operator: Observation operator instance
        :type observation_operator: ObservationOperator
        :param x0_run: Initial state for the DA cycles (which generates the truth/observations )
        :type x0_run: np.ndarray
        :param x0_analysis: Initial state for the optimization procedure
        :type x0_analysis: np.ndarray
        :param get_next_observations: Function which generates the next observations
        :type get_next_observations: Callable
        :param n_cycle: Number of DA cycle to perform
        :type n_cycle: int
        :param n_outer: Number of Linearization to perform
        :type n_outer: int
        :param n_inner: Number of iterations of CG to make
        :type n_inner: int
        :param prec: Preconditioner
        :type prec: dict
        :param plot: Plot some diagnostics
        :type plot: bool
        :param log_append: Append log to file
        :type log_append: bool
        """
        super().__init__(state_dimension, bounds)
        self.numerical_model = numerical_model
        self.observation_operator = observation_operator
        self.x0_run = x0_run
        self.x0_analysis = x0_analysis
        self.get_next_observations = get_next_observations
        self.n_cycle = n_cycle
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.preconditioner = prec
        self.plot = plot
        self.run_summary = None
        self.GNlog_file = None
        self.exp_name = "default"
        self.save_all = save_all

    def run(self, n_cycle: int = None, verbose=False) -> dict:
        if n_cycle is not None:
            print("Attribute n_cycle modified")
            self.n_cycle = n_cycle
        nobs = self.numerical_model.nobs
        x_t = self.x0_run
        t = np.arange(nobs)
        # Initialize arrays
        if self.save_all:
            truth_full = np.empty((self.state_dimension, nobs * self.n_cycle))
            analysis_full = np.empty((self.state_dimension, nobs * self.n_cycle))
            obs_full = np.empty((self.state_dimension, nobs * self.n_cycle))
        else:
            truth_full = None
            analysis_full = None
            obs_full = None

        n_iter_innerloop = []
        cost_outerloop = []
        sp_optimisation = []
        if self.x0_analysis is None:
            x0_optim = x_t + np.random.normal(size=self.state_dimension)
        else:
            x0_optim = self.x0_analysis
        quad_errors = []
        innerloop_residual_cycle = []

        for i_cycle in range(self.n_cycle):
            obs, x_t, truth = self.get_next_observations(x_t)
            truth = truth[:, 1:]

            self.numerical_model.set_obs(self.observation_operator(obs.reshape(-1)))
            gauss_newton_dict = self.numerical_model.GNmethod(
                x0_optim,
                n_outer=self.n_outer,
                n_inner=self.n_inner,
                verbose=True,
                prec=self.preconditioner,
                log_file=self.GNlog_file,
                exp_name=self.exp_name,
                i_cycle=i_cycle,
            )
            gn_x = gauss_newton_dict["gn_x"]
            # gn_fun = gauss_newton_dict["gn_fun"]
            # cost_inner = gauss_newton_dict["cost_inner"]
            quad_errors.append(gauss_newton_dict["quad_error"])
            innerloop_residual_cycle.append(gauss_newton_dict["inner_residual"])
            n_iter_innerloop.append(gauss_newton_dict["niter_inner"])
            cost_outerloop.append(gauss_newton_dict["cost_outer"])
            # sp_optim = scipy.optimize.minimize(self.numerical_model.cost_function, x0=x0_optim)
            # sp_optimisation.append(sp_optim.fun)
            if self.save_all:
                analysis = self.numerical_model.forward_no_obs(gn_x).reshape(
                    self.state_dimension, -1
                )[:, 1:]
                x0_optim = analysis[:, -1]
            else:
                analysis = self.numerical_model.forward_no_obs(gn_x)
                x0_optim = analysis
            if verbose:
                print(f"{self.numerical_model.forward_no_obs(gn_x).shape=}")
                print(f"{analysis.shape=}")
            t_cycle = t + i_cycle * nobs
            if verbose:
                print(f"{analysis.shape=}")
                print(f"{truth.shape=}")
                if self.save_all:
                    print(f"{analysis_full.shape=}")
                    print(f"{truth_full.shape=}")

            if self.save_all:
                analysis_full[:, t_cycle] = analysis
                truth_full[:, t_cycle] = truth
                obs_full[:, t_cycle] = obs[:, 1:]
            if self.plot:
                for i in range(self.state_dimension):
                    plt.subplot(self.state_dimension, 1, i + 1)
                    plt.plot(t_cycle, obs[i, 1:], "r.", label="obs")
                    plt.plot(t_cycle, analysis[i, :], color="green", label="analysis")
                    plt.plot(t_cycle, truth[i, :], color="black", label="truth")
                    plt.scatter(t_cycle[-1], x_t[i])
        if self.plot:
            plt.tight_layout()
            plt.show()

        self.run_summary = {
            "truth_full": truth_full,
            "analysis_full": analysis_full,
            "obs_full": obs_full,
            "n_iter_innerloop": n_iter_innerloop,
            "cost_outerloop": cost_outerloop,
            "sp_optimisation": sp_optimisation,
            "quad_errors": quad_errors,
            "l_model": self.numerical_model,
            "inner_res_cycle": innerloop_residual_cycle,
        }
        return self.run_summary

    def diagnostic_plots(self, title: str = ""):
        if self.run_summary is None:
            raise RuntimeError("No stored data")
        truth_full, obs_full, analysis_full = (
            self.run_summary["truth_full"],
            self.run_summary["obs_full"],
            self.run_summary["analysis_full"],
        )
        plt.subplot(1, 3, 1)
        plt.plot(
            ((truth_full - obs_full) ** 2).mean(0), label="Observation error (wrt obs)"
        )
        plt.plot(
            ((analysis_full - truth_full) ** 2).mean(0),
            label="Analysis error (wrt truth)",
        )
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.hist(((obs_full - truth_full) ** 2).mean(0), alpha=0.5, density=True)
        plt.hist(((analysis_full - truth_full) ** 2).mean(0), alpha=0.5, density=True)

        iter_CG = np.array(self.run_summary["n_iter_innerloop"]).T
        n_outer = len(iter_CG[:, 0])
        plt.subplot(1, 3, 3)
        plt.plot(np.arange(n_outer) + 1, iter_CG, color="gray", alpha=0.1)
        plt.boxplot(iter_CG.T)
        plt.xlabel(r"# of outer loop")
        plt.title(f"Number of CG iterations")
        # plt.axhline(n, color="red")
        plt.suptitle(f"{title}")
        plt.show()

    def plot_innerloopiter(self, color: str = "blue", label: str = "") -> None:
        ninnerloop = self.run_summary["n_iter_innerloop"]
        n_outer = len(ninnerloop[0])
        m_, s_, max_, min_ = (
            func(np.asarray(ninnerloop).T, 1)
            for func in [np.mean, np.std, np.max, np.min]
        )
        x_ = np.arange(n_outer) + 1
        plt.plot(x_, m_, color=color, alpha=1, label=label)
        plt.fill_between(x_, m_ + s_, m_ - s_, alpha=0.15, color=color)
        plt.plot(x_, max_, color=color, ls=":")
        plt.plot(x_, min_, color=color, ls=":")

    def plot_residuals_inner_loop(
        self, color: str = "blue", label: str = "", nostats=False, cumulative=False
    ) -> np.ndarray:
        """Plot the norm of the residuals along the inner loop iterations

        :param color: color in which to plot the norm, defaults to "blue"
        :type color: str, optional
        :param label: label, defaults to ""
        :type label: str, optional
        :param nostats: Should some statistical study be performed, defaults to False
        :type nostats: bool, optional
        :param cumulative: plot the cumulative fraction of finished iterations, defaults to False
        :type cumulative: bool, optional
        :raises RuntimeError: no stored data (run not performed)
        :return: array of residuals
        :rtype: np.ndarray
        """
        if self.run_summary is None:
            raise RuntimeError("No stored data")
        residuals = []
        for cy in self.run_summary["inner_res_cycle"]:
            for out in cy:
                residuals.append(np.sqrt((np.array(out["residuals"]) ** 2).sum(1)))
            plt.yscale("log")
        residuals = pad_ragged(residuals)
        iters = np.arange(residuals.shape[1])
        if nostats:
            for i, res in enumerate(residuals):
                plt.plot(res, color=plt.get_cmap("viridis")(i / len(residuals)))
        else:
            import scipy.stats

            plt.plot(residuals.T, color=color, alpha=0.1)
            geom_mean = np.exp(np.nanmean(np.log(residuals), 0))
            sd = np.nanstd(np.log(residuals), 0)
            # plt.plot(residuals.T, color=color, alpha=0.1)
            plt.plot(geom_mean, lw=5, color=color, label=label)
            plt.plot(np.exp(np.log(geom_mean) - sd), ":", color=color)
            plt.plot(np.exp(np.log(geom_mean) + sd), ":", color=color)

            lregress = scipy.stats.linregress(iters, np.log(geom_mean))
            print(
                f"{label}: log(residuals) = {lregress.intercept:.2f} +  {lregress.slope:.2f} * i"
            )
            lregress = scipy.stats.linregress(iters[:20], np.log(geom_mean[:20]))
            print(
                f"First 20 it, {label}: log(residuals) = {lregress.intercept:.2f} +  {lregress.slope:.2f} * i"
            )
            plt.plot(
                iters[:20],
                np.exp(lregress.intercept + lregress.slope * iters[:20]),
                color="white",
                label="fitted line",
                linestyle="dashed",
                linewidth=0.75,
            )
            if cumulative:
                ax2 = plt.gca().twinx()
                ax2.plot(np.isnan(residuals).mean(0), color=color, linestyle=":")
                ax2.grid(None)
                ax2.set_ylim([0, 1])
        return residuals

    def extract_condition_niter(self) -> Tuple:
        cond = []
        ite = []
        for ncy in range(self.n_cycle):
            for out in range(self.n_outer):
                cond.append(self.run_summary["inner_res_cycle"][ncy][out]["cond"])
                ite.append(self.run_summary["inner_res_cycle"][ncy][out]["niter"])
        return cond, ite


## Backward compatibility ?


def __data_assimilation(
    l_model,
    obs_operator,
    x0_t,
    get_next_observations,
    n_cycle,
    n_outer,
    n_inner,
    prec=None,
    plot=True,
):
    n = l_model.n
    nobs = l_model.nobs
    # x0_t = l_model.burn_model(n)
    x_t = x0_t
    t = np.arange(nobs)
    # print(f"{n=}, {nobs=}")
    truth_full = np.empty((n, nobs * n_cycle))
    analysis_full = np.empty((n, nobs * n_cycle))
    obs_full = np.empty((obs_operator.m, nobs * n_cycle))
    n_iter_innerloop = []

    cost_outerloop = []
    sp_optimisation = []
    x0_optim = x_t + np.random.normal(size=n)
    quad_errors = []
    inner_res_cycle = []

    if plot:
        plt.figure(figsize=(10, 2 * n))
    for i_cycle in range(n_cycle):
        obs, x_t, truth = get_next_observations(x_t)
        # print(f"before: {truth.shape=}")
        truth = truth[:, 1:]

        l_model.set_obs(obs_operator(obs.reshape(-1)))
        (
            gn_x,
            gn_fun,
            n_iter_inner,
            cost_outer,
            cost_inner,
            quad_error,
            inner_res,
        ) = l_model.GNmethod(
            x0_optim, n_outer=n_outer, n_inner=n_inner, verbose=True, prec=prec
        )
        inner_res_cycle.append(inner_res)
        n_iter_innerloop.append(n_iter_inner)
        cost_outerloop.append(cost_outer)
        # sp_optim = scipy.optimize.minimize(l_model.cost_function, x0=x0_optim)
        # sp_optimisation.append(sp_optim.fun)
        quad_errors.append(quad_error)
        analysis = l_model.forward_no_obs(gn_x).reshape(n, -1)[:, 1:]
        # print(f"{l_model.forward_no_obs(gn_x).shape=}")
        # print(f"{analysis.shape=}")
        x0_optim = analysis[:, -1]
        t_cycle = t + i_cycle * nobs
        # print(f"{analysis.shape=}")
        # print(f"{analysis_full.shape=}")
        # print(f"{truth.shape=}")
        # print(f"{truth_full.shape=}")
        analysis_full[:, t_cycle] = analysis
        truth_full[:, t_cycle] = truth
        # obs_full[:, t_cycle] = obs[:, 1:]
        if plot:
            for i in range(n):
                plt.subplot(n, 1, i + 1)
                plt.plot(t_cycle, obs[i, 1:], "r.", label="obs")
                plt.plot(t_cycle, analysis[i, :], color="green", label="analysis")
                plt.plot(t_cycle, truth[i, :], color="black", label="truth")
                plt.scatter(t_cycle[-1], x_t[i])
                # if i % 5 == 0:
                #     plt.legend()
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
        "inner_res_cycle": inner_res_cycle,
    }


def __diagnostic_plots(DA, title):
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
    n_outer = len(iter_CG[:, 0])
    plt.subplot(1, 3, 3)
    plt.plot(np.arange(n_outer) + 1, iter_CG, color="gray", alpha=0.1)
    plt.boxplot(iter_CG.T)
    plt.xlabel(r"# of outer loop")
    plt.title(f"Number of CG iterations")
    # plt.axhline(n, color="red")
    plt.suptitle(f"{title}")
    plt.show()


def __plot_innerloopiter(DA_dict, color, label):
    ninnerloop = DA_dict["n_iter_innerloop"]
    n_outer = len(ninnerloop[0])
    m_, s_, max_, min_ = (
        func(np.asarray(ninnerloop).T, 1) for func in [np.mean, np.std, np.max, np.min]
    )
    x_ = np.arange(n_outer) + 1
    plt.plot(x_, m_, color=color, alpha=1, label=label)
    plt.fill_between(x_, m_ + s_, m_ - s_, alpha=0.15, color=color)
    plt.plot(x_, max_, color=color, ls=":")
    plt.plot(x_, min_, color=color, ls=":")
