from functools import partial
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import scipy.linalg as la
from tqdm.auto import tqdm

from ..common.utils import DataAssimilationMethod


def matrix_induced_norm(x: np.ndarray, Sigma: np.ndarray, mode="inv") -> float:
    """Computes the squared norm of a vector with a matrix (x^T Σ x) = ||x||_Σ

    :param x: vector whose norm to compute
    :type x: np.ndarray
    :param Sigma: Matrix to use to weight the norm
    :type Sigma: np.ndarray
    :param mode: if "inv", use directly Sigma to compute the norm, if "chol",
    Sigma is the cholesky decomposition, defaults to "inv"
    :type mode: str, optional
    :return: squared matrix norm
    :rtype: float
    """
    x = np.atleast_2d(x).T
    if mode == "inv":
        # return nla.multi_dot([x.T, Sigma, x])
        return x.T @ Sigma @ x
    elif mode == "chol":
        return la.norm(np.matmul(Sigma, x)) ** 2


class VariationalMethod(DataAssimilationMethod):
    def __init__(self, state_dimension: int, bounds: np.ndarray) -> None:
        super().__init__(state_dimension=state_dimension)
        self.bounds = bounds

    def set_background_gaussian(self, xb: np.ndarray, B: np.ndarray) -> None:
        """Set the background information using a Gaussian prior

        :param xb: background value
        :type xb: np.ndarray
        :param B: Background covariance error
        :type B: np.ndarray
        """
        self.xb = xb
        self.B = B
        self.Binv = la.inv(B)
        self.Bchol = la.cholesky(B)

    def cost_background(self, x: np.ndarray) -> float:
        """Computes the error associated with the background misfit

        :param x: state vector
        :type x: np.ndarray
        :return: half the squared norm of the background difference
        :rtype: float
        """
        bck_misfit = x - self.xb
        return 0.5 * matrix_induced_norm(bck_misfit, self.Binv)

    def set_optimizer(
        self, optimizer: Callable, opt_params: Optional[Dict] = {}
    ) -> None:
        """Set an optimizer to use in order to perform the analysis

        :param optimizer: Callable that returns the minimizer and the minimum
        :type optimizer: Callable
        :param opt_params: args to pass to the optimizer, defaults to {}
        :type opt_params: Optional[Dict], optional
        """
        self.optimizer = partial(optimizer, **opt_params)

    def vectorized_fun_to_optim(
        self, y: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        """returns the function to minimize given the observation y

        :param y: observation
        :type y: np.ndarray
        :return: a "vectorized" function which computes the loss associated with its argument
        :rtype: Callable[[np.ndarray], np.ndarray]
        """

        def fto(x):
            x = np.atleast_2d(x)
            J = np.empty(len(x))
            for i, x_ in enumerate(x):
                J[i] = self.cost_function_element(y, x_)
            return J

        return fto

    def optimization(self, y: np.ndarray, x0: np.ndarray) -> Tuple[np.ndarray, float]:
        """Performs the optimization given the observation y and the starting point x0

        :param y: observation
        :type y: np.ndarray
        :param x0: initial guess
        :type x0: np.ndarray
        :return: the optimizer and the minimum associated
        :rtype: Tuple[np.ndarray, float]
        """
        optim = self.optimizer(fun=self.vectorized_fun_to_optim(y), x0=x0)
        return optim.x, optim.fun

    def run(
        self,
        Nsteps: int,
        x0: np.ndarray,
        get_obs: Callable[[int], Tuple[float, np.ndarray]],
        full_obs: bool = True,
    ) -> dict:
        """Run the Variational Data assimilation method

        :param Nsteps: Number of assimilations steps, ie cycle
        :type Nsteps: int
        :param x0: starting point for the optimizations
        :type x0: np.ndarray
        :param get_obs: Function to call in order to get a tuple of time, observation
        :type get_obs: Callable[[int], Tuple[float, np.ndarray]]
        :param full_obs: Is there a need to apply the observation operator when analysing those, defaults to True
        :type full_obs: bool, optional
        :return: Dictionary composed of the analysed states, observations, minimums and time
        :rtype: dict
        """
        observations = []
        xa = []
        J = []
        time = []
        for i in tqdm(range(Nsteps)):
            # self.forecast_ensemble()
            # ensemble_f.append(self.xf_ensemble)
            t, y = get_obs(i)
            observations.append(y)
            time.append(t)
            if full_obs:
                y = self.H(y)
            x, Jx = self.optimization(y, x0)
            xa.append(x)
            x0 = xa[-1]
            J.append(Jx)
        return {
            "observations": observations,
            "xa": xa,
            "J": J,
            "time": time,
        }
