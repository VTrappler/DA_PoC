import sys

sys.path.append("/home/vtrappler/")
sys.path.append("/home/vtrappler/qgs/")

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Callable
import tqdm

plt.style.use("seaborn-notebook")
from qgs.QG import QGwrapper

# Let us set a rng for reproducibility
rng = np.random.default_rng(seed=93)

qg_model = QGwrapper(wavenumbers=(3, 3), tsteps=0.05, write_steps=1, dt=0.01)
qg_model.configure(tangent_linear=True)
qg_model.burn_model(burn=20_000)

for mul in [5, 20, 100, 200, 500]:
    qg_model.change_settings(tsteps=0.01 * mul, write_steps=1, dt=None)
    GN, (time, _, _) = qg_model.GaussNewtonMatrix(qg_model.initial_conditions)
    slogdet = np.linalg.slogdet(GN)
    print(mul, slogdet, np.linalg.cond(GN))

plt.subplot(1, 2, 1)
plt.title(np.linalg.slogdet(GN))
plt.imshow(GN)
plt.subplot(1, 2, 2)
plt.plot(np.linalg.eigvalsh(GN))


tstep_1day = qg_model.model_parameters.dimensional_time ** (-1)
## Run model for 1 day
qg_model.change_settings(tsteps=1 * tstep_1day, write_steps=1, dt=None)
GN, (time, _, _) = qg_model.GaussNewtonMatrix(qg_model.initial_conditions)
slogdet = np.linalg.slogdet(GN)
print(slogdet, np.linalg.cond(GN))
plt.subplot(1, 2, 1)
plt.title(np.linalg.slogdet(GN))
plt.imshow(GN)
plt.subplot(1, 2, 2)
plt.title(np.linalg.cond(GN))
plt.plot(np.linalg.eigvalsh(GN))
plt.yscale("log")


t, traj, jacobian_ = qg_model.forward_jacobian(x=qg_model.initial_conditions)
jacobian = jacobian_[:, :, -1]

y = qg_model.forward(x=2 * qg_model.initial_conditions)[1][:, -1]



