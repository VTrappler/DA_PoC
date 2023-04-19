import sys

sys.path.append("..")
import matplotlib.pyplot as plt

# from DA_PoC.dynamical_systems.sw_numerical_model
from DA_PoC.dynamical_systems.examples.sw import SWModelJax
from DA_PoC.common.numerical_model import NumericalModel

from DA_PoC.common.observation_operator import LinearObservationOperator
import numpy as np

n_x = 32
dx = 5e3
n_y = 32
dy = 5e3
swmodel = SWModelJax(n_x, dx, n_y, dy, periodic_x=True)


state_dimension = 3 * swmodel.state_variable_length
obs_dimension = swmodel.state_variable_length
h_no_nan = np.eye(swmodel.state_variable_length)
for i in range(swmodel.state_variable_length):
    if i % 32 == 0:
        h_no_nan[i, i] = 0
        h_no_nan[i - 1, i - 1] = 0

H_obs = np.hstack(
    [
        np.zeros((swmodel.state_variable_length, swmodel.state_variable_length)),
        np.zeros((swmodel.state_variable_length, swmodel.state_variable_length)),
        h_no_nan,
    ]
)
obs_operator = LinearObservationOperator(Hmatrix=H_obs)
window = 15
numerical_model_sw = NumericalModel(state_dimension, obs_dimension)
background = np.zeros(state_dimension)
background[swmodel.slice_h] = swmodel.depth
numerical_model_sw.background = background
numerical_model_sw.background_error_cov_inv = np.eye(state_dimension)
numerical_model_sw.set_observation_operator(obs_operator)


def remove_nan(array):
    return np.where(np.isnan(array), 0, array)


numerical_model_sw.set_forward(lambda x: remove_nan(swmodel.forward(x, n_steps=window)))
numerical_model_sw.set_tangent_linear(
    lambda x: np.asarray(swmodel.forward_TLM(x, n_steps=window)[0])
)
x0_t = background
x0_t[swmodel.slice_h] += 3 * np.random.randn(swmodel.state_variable_length)


# new_state = numerical_model_sw.forward_no_obs(x0_t)
def generate_obs_no_noise(x0, n_steps):
    return remove_nan(swmodel.forward(x0, n_steps=n_steps)[swmodel.slice_h])


# Generate observations
true_obs = generate_obs_no_noise(x0_t, window)
noisy_obs = true_obs + 0.01 * np.random.normal(loc=0, scale=0.1, size=true_obs.shape)

numerical_model_sw.set_obs(noisy_obs)
numerical_model_sw.cost_function(x0_t)


numerical_model_sw.tests_consistency(figname="/DA_PoC/notebooks/consistency_sw.png")
# res=  swmodel.forward_TLM_h(x0_t, n_steps=window)
