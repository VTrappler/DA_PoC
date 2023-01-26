import numpy as np
import matplotlib.pyplot as plt

def data_assimilation(
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
            inner_res
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
        "inner_res_cycle": inner_res_cycle
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
    n_outer = len(iter_CG[:, 0])
    plt.subplot(1, 3, 3)
    plt.plot(np.arange(n_outer) + 1, iter_CG, color="gray", alpha=0.1)
    plt.boxplot(iter_CG.T)
    plt.xlabel(r"# of outer loop")
    plt.title(f"Number of CG iterations")
    # plt.axhline(n, color="red")
    plt.suptitle(f"{title}")
    plt.show()



def plot_innerloopiter(DA_dict, color, label):
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