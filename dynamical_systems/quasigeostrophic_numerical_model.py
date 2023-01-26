import sys

sys.path.append("../..")

from copy import deepcopy
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.sparse.linalg as sla
import tqdm

plt.style.use("seaborn-notebook")
sys.path.append("..")

import warnings
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
from DA_PoC.common.numerical_model import NumericalModel
from DA_PoC.common.observation_operator import (
    IdentityObservationOperator,
    RandomObservationOperator,
)

# Let us set a rng for reproducibility
rng = np.random.default_rng(seed=93)


def create_QG_model(qg_model, obs_operator, test=True, background=None, gnparams=None):
    n = qg_model.spectral_dim

    model = NumericalModel(n, n)
    model.set_obs(qg_model.obs)
    model.set_forward(lambda x: qg_model.forward(x)[1][:, -1])
    model.set_observation_operator(obs_operator)
    model.nobs = 1
    model.set_tangent_linear(lambda x: qg_model.forward_jacobian(x=x)[-1][..., -1])
    if background is not None:
        model.background, model.background_error_cov_inv = background

    if test:
        model.tests_consistency()
        x0 = np.zeros(n)
        sp_opt = scipy.optimize.minimize(model.cost_function, x0)
        sp_x, sp_fun = sp_opt.x, sp_opt.fun
        if gnparams is None:
            n_inner, n_outer, sigma = 50, 30, 1
        else:
            n_inner, n_outer, sigma = gnparams
        gn_x, gn_fun, n_iter, cost_outer, cost_inner, quad_error = model.GNmethod(
            sigma * np.random.normal(size=n),
            n_outer=n_outer,
            n_inner=n_inner,
            verbose=True,
            prec=None,
        )

        print(f"{sp_fun=}, {gn_fun=}")
        plt.plot(sp_x, label="scipy")
        plt.plot(gn_x, label="GN")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return model, sp_x, gn_fun
    else:
        return model, None, None


def create_QG_model_grid(
    qg_model, obs_operator, test=True, background=None, gnparams=None
):
    n = qg_model.spectral_dim
    n_grid = qg_model.grid_x * qg_model.grid_y

    model = NumericalModel(n, 2 * n_grid)

    def forward_grid(x):
        return np.moveaxis(
            np.asarray(qg_model.forward_grid(x)[2])[:, -1, :, :], -1, -2
        ).flatten()

    model.set_observation_operator(obs_operator)
    model.set_obs(obs_operator(qg_model.obs))
    model.set_forward(forward_grid)
    model.nobs = 1

    model.qg = qg_model

    def jac_grid(x):
        return np.asarray(qg_model.forward_jacobian_grid(x))[..., -1].reshape(-1, n)

    model.set_tangent_linear(jac_grid)

    if background is not None:
        (
            model.background,
            model.background_error_cov_inv,
            model.background_error_sqrt,
        ) = background

    if test:
        model.tests_consistency()
        x0 = np.zeros(n)
        sp_opt = scipy.optimize.minimize(model.cost_function, x0)
        sp_x, sp_fun = sp_opt.x, sp_opt.fun
        if gnparams is None:
            n_inner, n_outer, sigma = 50, 30, 1
        else:
            n_inner, n_outer, sigma = gnparams
        gn_x, gn_fun, n_iter, cost_outer, cost_inner, quad_error = model.GNmethod(
            sigma * np.random.normal(size=n),
            n_outer=n_outer,
            n_inner=n_inner,
            verbose=True,
            prec=None,
        )

        print(f"{sp_fun=}, {gn_fun=}")
        plt.plot(sp_x, label="scipy")
        plt.plot(gn_x, label="GN")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return model, sp_x, gn_fun
    else:
        return model, None, None
