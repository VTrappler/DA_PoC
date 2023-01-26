import argparse
import pickle
import sys
import os
sys.path.append("..")
sys.path.append("../..")
sys.path.append("/da_dev/qgs/")

import warnings
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import tqdm
from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
# m = n * (nobs + 1)
from common.numerical_model import NumericalModel
from common.observation_operator import IdentityObservationOperator, RandomObservationOperator
from dynamical_systems.quasigeostrophic_numerical_model import create_QG_model, create_QG_model_grid

plt.style.use("seaborn") #type: ignore
plt.set_cmap('magma')
from qgs.QG import QGwrapper #type: ignore

# Let us set a rng for reproducibility
rng = np.random.default_rng(seed=93)


def generate_model(wave_x, wave_y, days=1):
    qg_model = QGwrapper(wavenumbers=(wave_x, wave_y), tsteps=0.05, write_steps=1, dt=0.01)
    qg_model.configure(tangent_linear=True)
    qg_model.burn_model(burn=20_000)


    tsteps = qg_model.tstep_1day * days
    qg_model.change_settings(tsteps=tsteps, write_steps=1, dt=None)
    print(f"{(tsteps / qg_model.tstep_1day) * 24:.2f} hours")
    # obs_operator = RandomObservationOperator(2 * qg_model.grid_x * qg_model.grid_y, 2 * qg_model.grid_x * qg_model.grid_y, type='square', p=0.5, p_offdiag=0.01)
    obs_operator = IdentityObservationOperator(2 * qg_model.grid_x * qg_model.grid_y, 2 * qg_model.grid_x * qg_model.grid_y)


    truth = np.moveaxis(np.array(qg_model.forward_grid(qg_model.initial_conditions)[2])[:, -1, :, :], -1, -2).flatten()
    obs = truth + np.random.normal(size=truth.shape)
    qg_model.obs = obs

    Binv = np.load(f"quasi_geostrophic_Binv/Binv_{qg_model.x}_{qg_model.y}.npy")
    L = np.linalg.cholesky(np.linalg.inv(Binv))

    background = np.zeros(qg_model.spectral_dim), Binv, L
    model, _, _ = create_QG_model_grid(qg_model, obs_operator=obs_operator, background=background, test=False, gnparams=(10, 10, 0.1))
    return model, qg_model

def get_GN(x0, qg_model, model):
    H_GN = model.gauss_newton_hessian_matrix(x0)
    t, traj = qg_model.forward(x0)
    return x0, traj[:, -1], H_GN


def generate_training_dataset(qg_model, model, x0=None, N=100):
    if x0 is None:
        x0 = qg_model.initial_conditions
    train = []
    x = x0
    for _ in tqdm.trange(N):
        out = get_GN(x, qg_model, model)
        train.append(out)
        x = out[0]
    return train

def save_data(data, path):
    pickle.dump(data, open(f"{path}", "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the training data for the QG model"
    )
    parser.add_argument("wavex", type=int, help="x-wave numbers")
    parser.add_argument("wavey", type=int, help="y-wave numbers")

    parser.add_argument("N", type=int, help="Number of training data to save")
    parser.add_argument("target", type=str, help="target file")
    # parser.add_argument("--dummy", action="store_true")
    # parser.add_argument("--vector", action="store_true", help="Use G^T G dx")
    # parser.add_argument("--nbatch", type=int, help='Number of perturbations dx to evaluate', default=1)
    args = parser.parse_args()

    print(f"Training tuples to save: {args.N}")
    print(f"into {args.target}")
  
    model, qg_model = generate_model(args.wavex, args.wavey, days=1.0)
    # def H_nl(x, gamma=4):
    #     return 2 * x + 1
    training_data = generate_training_dataset(qg_model, model, N=args.N)
    save_data(training_data, args.target)
    print(f"Done")


