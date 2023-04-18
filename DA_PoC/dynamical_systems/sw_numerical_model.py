import jax
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Callable
import tqdm
import scipy.sparse.linalg as sla
import scipy.optimize
from copy import deepcopy
import time

plt.style.use("seaborn-v0_8")
from .examples.sw import SWModelJax
from DA_PoC.common.numerical_model import NumericalModel

from DA_PoC.common.observation_operator import (
    ObservationOperator,
)

rng = np.random.default_rng()


def create_lorenz_model_observation(
    swmodel: SWModelJax,
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
    # m = n * (nobs + 1)
    # n = lorenz.state_dimension
    # dim_observation = obs_operator.m
    # print(n)
    state_dimension = 3 * swmodel.state_variable_length
    obs_dimension = swmodel.state_variable_length
    H_obs = np.hstack(
        [
            np.zeros(swmodel.grid_shape),
            np.zeros(swmodel.grid_shape),
            np.eye(swmodel.state_variable_length),
        ]
    )

    obs_operator = ObservationOperator(H_obs=H_obs)
    window = 15
    # print(f"{lorenz.obs.shape=}")
    # print(f"{lorenz.H=}")
    numerical_model_sw = NumericalModel(state_dimension, obs_dimension)
    background = np.zeros(state_dimension)
    background[swmodel.slice_h] = swmodel.depth
    numerical_model_sw.background = background
    numerical_model_sw.background_error_cov_inv = np.eye(state_dimension)
    numerical_model_sw.set_observation_operator(obs_operator)
    # numerical_model_sw.set_obs(obs_operator(lorenz.obs))
    numerical_model_sw.set_forward(lambda x: swmodel.forward(x, n_steps=window))
    numerical_model_sw.set_tangent_linear(
        lambda x: swmodel.forward_TLM_h(x, n_steps=window)
    )
    x0_t = 100 + np.random.normal(size=state_dimension)
    print(f"{numerical_model_sw.cost_function(x0_t)=}")

    if test_consistency:
        numerical_model_sw.tests_consistency()
        x0 = np.zeros(n)
        print(f"Comparison of Scipy method and Gauss Newton/CG method\n")
        print(f"- scipy")
        before = time.time()
        sp_opt = scipy.optimize.minimize(numerical_model_sw.cost_function, x0)
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
        ) = numerical_model_sw.GNmethod(
            1 * np.random.normal(size=n),
            n_outer=10,
            n_inner=50,
            verbose=True,
            prec=None,
        )
        print(f" - time elapsed: {time.time() - before}s")
        print(f"{sp_fun=}, {gn_fun=}")
    return numerical_model_sw
