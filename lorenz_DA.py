#%%
import sys

sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-notebook")
# from DA_PoC.dynamical_systems.examples.lorenz96 import Lorenz96Model
# from DA_PoC.dynamical_systems.examples.lorenz63 import Lorenz63Model
# from DA_PoC.common.numerical_model import NumericalModel
from DA_PoC.common.observation_operator import (
    RandomObservationOperator,
    IdentityObservationOperator,
)
from DA_PoC.dynamical_systems.lorenz_numerical_model import (
    LorenzWrapper,
    create_lorenz_model_observation,
    data_assimilation,
    diagnostic_plots,
    plot_innerloopiter,
)

#%%
rng = np.random.default_rng(seed=93)

n = 100
nobs = 15

#%%
lorenz = LorenzWrapper(n)
lorenz.H = lambda x: x
lorenz.set_observations(nobs=nobs, burn=500)
lorenz.n_total_obs = nobs


def get_next_observations(x_init, lorenz=lorenz, modsigsq=0.5, obssigsq=3, nobs=nobs):
    lorenz.n_total_obs = nobs
    n = lorenz.state_dimension
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


#%%
print(f"Random observation operator   ---------------")
m = n * (nobs + 1)

random_obs_operator = IdentityObservationOperator(m, m)  # , 0.9, 0)
random_obs_operator = RandomObservationOperator(m, m, type="rect", p=1, p_offdiag=0)

# plt.imshow(random_obs_operator.H)
# m = n * (nobs + 1)
l_model_randobs = create_lorenz_model_observation(
    lorenz, m, random_obs_operator, test=False
)

#%%
n_cycle = 2
n_outer = 10
n_inner = 50
# np.random.set_state(7071522)
DA_vanilla = data_assimilation(
    l_model_randobs,
    random_obs_operator,
    get_next_observations,
    n_cycle,
    n_outer,
    n_inner,
    prec=None,
    plot=False,
)
diagnostic_plots(DA_vanilla, "vanilla")

#%%
l_model_randobs.r = n // 10
DA_LMP_10 = data_assimilation(
    l_model_randobs,
    random_obs_operator,
    get_next_observations,
    n_cycle,
    n_outer,
    n_inner,
    prec="spectralLMP",
    plot=False,
)
diagnostic_plots(DA_LMP_10, "LMP 10")
#%%
l_model_randobs.r = n // 20
DA_LMP_5 = data_assimilation(
    l_model_randobs,
    random_obs_operator,
    get_next_observations,
    n_cycle,
    n_outer,
    n_inner,
    prec="spectralLMP",
    plot=False,
)
diagnostic_plots(DA_LMP_5, "LMP 5")
#%%
l_model_randobs.r = n // 50
DA_LMP_2 = data_assimilation(
    l_model_randobs,
    random_obs_operator,
    get_next_observations,
    n_cycle,
    n_outer,
    n_inner,
    prec="spectralLMP",
    plot=False,
)
diagnostic_plots(DA_LMP_2, "LMP 2")

#%%
plot_innerloopiter(DA_vanilla, "blue", "vanilla")
plot_innerloopiter(DA_LMP_10, "red", "LMP 10")
plot_innerloopiter(DA_LMP_5, "magenta", "LMP 5")
plot_innerloopiter(DA_LMP_2, "grey", "LMP 2")
plt.legend()
