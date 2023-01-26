import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Callable
import tqdm

plt.style.use("seaborn-notebook")
from DA_PoC.dynamical_systems.examples.lorenz96 import Lorenz96Model
from DA_PoC.dynamical_systems.examples.lorenz63 import Lorenz63Model

# Let us set a rng for reproducibility
rng = np.random.default_rng(seed=93)


def generate_observations(
    truth, i, state_dimension, sigsqobs, period_assim
) -> Tuple[float, np.ndarray]:
    """Generates the "truth" state vector. We perturbate them all

    :param i: dummy variable, indicating the time step
    :type i: int
    :return:
    :rtype: [type]
    """
    truth.forward(period_assim)
    y = truth.state_vector[:, -1] + rng.multivariate_normal(
        np.zeros(state_dimension), np.eye(state_dimension)
    ) * np.sqrt(sigsqobs)
    return truth.t[-1], y


class LorenzWrapper:
    def __init__(self, state_dimension: int):
        self.state_dimension = state_dimension
        if state_dimension == 3:
            self.lorenz_model = Lorenz63Model
        else:
            self.lorenz_model = Lorenz96Model
            self.lorenz_model.dim = state_dimension
        self.period_assim = 1
        self.lorenz_model.dt = 0.02

    def create_and_burn_truth(self, burn: int = 2000):
        """Initiate the state vector with a burn period of burn timesteps

        :param burn: number of timesteps before reaching t=0, defaults to 2000
        :type burn: int, optional
        """
        x0 = np.zeros(self.state_dimension)
        x0[0] = 1
        self.truth = self.lorenz_model()
        self.truth.set_initial_state(-burn * self.truth.dt, x0)
        self.truth.forward(burn)

    def generate_obs(self, n_total_obs: int = 100, H: Callable = lambda x: x):
        """Generate n_total_obs observations

        :param n_total_obs: _description_, defaults to 100
        :type n_total_obs: int, optional
        :param H: _description_, defaults to lambdax:x
        :type H: _type_, optional
        """
        self.n_total_obs = n_total_obs
        obs = np.empty((self.state_dimension, self.n_total_obs))
        time_obs = np.empty(self.n_total_obs)
        self.initial_state = self.truth.state_vector[:, -1] + rng.multivariate_normal(
            np.zeros(self.state_dimension), cov=np.eye(self.state_dimension)
        )

        for i in range(self.n_total_obs):
            generated = generate_observations(
                self.truth, i, self.state_dimension, 1, self.period_assim
            )
            time_obs[i], obs[:, i] = generated
        self.obs = obs
        self.time_obs = time_obs
        self.H = H

    def forward_model(self, x: np.ndarray):
        """Integrates the model over the whole assimilation window"""
        return (
            self.H(self.lorenz_model.integrate(0, x, self.n_total_obs - 1)[1])
        ).flatten()

    def cost_function(self, x: np.ndarray):
        """computes the cost functions with respect to the observations"""
        diff = self.forward_model(x) - self.H(self.obs).flatten()
        return 0.5 * diff.T @ diff

    def gradient(self, x: np.ndarray):
        """Computes the gradient using finite differences"""
        grad = np.zeros(self.state_dimension)
        e = np.zeros_like(x)
        base = self.cost_function(x)
        for i in range(self.state_dimension):
            e[i] = self.eps
            grad[i] = (self.cost_function(x + e) - base) / self.eps
            e[i] = 0
        return grad

    def forward_TLM_fd(self, x: np.ndarray, return_base: bool = False):
        """Computes the forward model and its TLM using finite difference"""
        e = np.zeros_like(x)
        tlm = np.empty(((self.state_dimension * self.n_total_obs, len(x))))
        forbase = self.forward_model(x).flatten()
        for i in range(self.state_dimension):
            e[i] = self.eps
            tlm[..., i] = (self.forward_model(x + e).flatten() - forbase) / self.eps
            e[i] = 0
        if return_base:
            return forbase, tlm
        else:
            return tlm

    def forward_TLM(self, x: np.ndarray, return_base: bool = False):
        """Computes the forward and the tangent linear

        :param x: initial state vector
        :type x: np.ndarray
        :param return_base: Should the method return the forward, defaults to False
        :type return_base: bool, optional
        :return: Forward model (if return_bas) and Tangent linear
        :rtype: Tuple
        """
        tlm = self.lorenz_model.construct_tlm_matrix(0, x, self.n_total_obs - 1)
        if return_base:
            return self.forward_model(x), tlm
        else:
            return tlm

    def grad_TLM(self, x: np.ndarray):
        """Computes the TLM and the gradient"""
        tlm = self.forward_TLM(x, self.eps, return_base=False)
        return tlm, tlm.T @ (self.forward_model(x) - self.H(self.obs).flatten())

    def get_quantities_for_optim(self, x: np.ndarray):
        forward = self.forward_model(x)
        diff = forward - self.H(self.obs).flatten()
        cost = 0.5 * diff.T @ diff
        grad = np.zeros(len(x))
        e = np.zeros_like(x)
        tlm = np.empty(((self.state_dimension * self.n_total_obs, len(x))))
        for i in range(len(x)):
            e[i] = self.eps
            forward_e = self.forward_model(x + e)
            diff_e = forward_e - self.H(self.obs).flatten()
            cost_e = 0.5 * diff_e.T @ diff_e
            grad[i] = (cost_e - cost) / self.eps
            tlm[..., i] = (forward_e.flatten() - forward) / self.eps
            e[i] = 0
        return forward, cost, grad, tlm

    def get_quantities_for_optim_ML(self, x: np.ndarray):
        forward = self.forward_model(x)
        diff = forward - self.H(self.obs).flatten()
        cost = 0.5 * diff.T @ diff
        grad = np.zeros(len(x))
        e = np.zeros_like(x)
        for i in range(len(x)):
            e[i] = self.eps
            grad[i] = (self.cost_function(x + e) - cost) / self.eps
            e[i] = 0
        return forward, cost, grad, []


if __name__ == "__main__":
    lorenz = LorenzWrapper(state_dimension=3)
    lorenz.create_and_burn_truth()
    plt.imshow(lorenz.truth.state_vector, aspect="auto")
    plt.title("Burn period for the Lorenz96 model")
    lorenz.generate_obs(n_total_obs=100)
    plt.figure(figsize=(16, 5))
    plt.imshow(lorenz.obs)
    plt.title(r"Perturbated truth vector, observations are above the horizontal line")
    plt.show()

