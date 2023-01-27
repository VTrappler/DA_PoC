from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
import scipy.optimize
import tqdm.auto as tqdm

from .VariationalMethod import VariationalMethod, matrix_induced_norm


class FourDimensionalVar(VariationalMethod):
    """Class which implements the 4DVar (weak and strong formulation) for sequential Data Assimilation"""

    def __init__(
        self,
        state_dimension: int,
        R: np.ndarray,
        bounds: np.ndarray,
        assimilation_window: int,
        mode: str = "strong",
    ) -> None:
        super().__init__(state_dimension, bounds)
        self.R = R
        self.assimilation_window = assimilation_window
        self.mode = mode

    def set_model_error(self, Q: np.ndarray) -> None:
        self.Q = Q
        self.Qinv = la.inv(Q)

    def cost_model_weak(self, x: np.ndarray) -> float:
        """Computes the error in the propagation of the model, when the constraint is weak

        :param x: flattened state vector
        :type x: np.ndarray
        :return: Cumulative sum of the norms
        :rtype: float
        """
        x_iterator = self.iterator_input(x)
        acc = 0
        for xim1, xi in zip(x_iterator[:-1], x_iterator[1:]):
            diff = self.forward(xim1) - xi
            acc += matrix_induced_norm(diff, self.Qinv)
        return acc * 0.5

    def cost_model_strong(self, x: np.ndarray, y: List[np.ndarray]) -> float:
        """Compute the data misfit

        :param x: state vector corresponding to the first observation
        :type x: np.ndarray
        :param y:
        :type y: List[np.ndarray]
        :return: Cumulative sum of the norm of the data misfits over the assimilation window
        :rtype: float
        """
        acc = 0
        xi = x
        for yi in y:
            misfit = yi - self.H(xi)
            acc += matrix_induced_norm(misfit, self.Rinv)
            xi = self.forward(xi)  # Strong constraint on the model
        return 0.5 * acc

    def cost_observation_weak(self, x: np.ndarray, y: List[np.ndarray]) -> float:
        """Computes the data misfit in the weak constraint case

        :param x: Consecutive values of the state vector over the assimilation window, flattened
        :type x: np.ndarray
        :param y:
        :type y: np.ndarray
        :return: List[np.ndarray]
        :rtype: float
        """
        x_iterator = self.iterator_input(x)
        acc = 0
        for xi, yi in zip(x_iterator, y):
            misfit = yi - self.H(xi)
            acc += matrix_induced_norm(misfit, self.Rinv)
        return acc * 0.5

    def cost_function_element(self, y: List[np.ndarray], x: np.ndarray) -> float:
        """Evaluate the cost function given the observations y on the assimilation window, and the state vector x

        :param y: observations on the assimilation window
        :type y: List[np.ndarray]
        :param x: state vector
        :type x: np.ndarray
        :return: Cost function evaluated at pair x, y
        :rtype: float
        """
        if self.mode == "weak":
            j_obs = self.cost_observation_weak(x, y)
            j_obs += self.cost_model_weak(x)
            j_bck = self.cost_background(self.iterator_input(x)[0])
        elif self.mode == "strong":
            j_obs = self.cost_model_strong(x, y)
            j_bck = self.cost_background(x)
        return j_obs + j_bck

    def flatten_input(self, x: List[np.ndarray]) -> np.ndarray:
        """Flatten the list of state vector

        :param x: list of state vectors, of len = assimilation_window
        :type x: List[np.ndarray]
        :return: flattened state vector of dimension (state_dimension * assimilation_window)
        :rtype: np.ndarray
        """
        return np.array(x).reshape(-1)

    def iterator_input(self, x: np.ndarray) -> List[np.ndarray]:
        """Transform the flattened state vector into list of state vectors

        :param x: flattened state vector
        :type x: np.ndarray
        :return: list of state vectors
        :rtype: List[np.ndarray]
        """
        return np.split(x, self.assimilation_window)

    def optimization(
        self,
        fun: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs the optimization of a function, and returns the minimizer and minimum

        :param fun: Function to minimize
        :type fun: Callable[[np.ndarray], np.ndarray]
        :param x0: starting point
        :type x0: np.ndarray
        :return: Tuple formed of the minimizer and the minimum
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        if self.optimizer is None:
            self.optimizer = scipy.optimize.minimize
        optim = self.optimizer(fun=fun, x0=x0)
        return optim.x, optim.fun

    def run(
        self,
        Nsteps: int,
        x0: np.ndarray,
        get_obs: Callable[[int], Tuple[float, np.ndarray]],
        full_obs: bool = True,
        mode: Optional[str] = None,
    ) -> dict:
        """Run the 4DVar

        :param Nsteps: Number of assimilations steps, ie cycle
        :type Nsteps: int
        :param x0: starting point for the optimizations
        :type x0: np.ndarray
        :param get_obs: Function to call in order to get a tuple of time, observation
        :type get_obs: Callable[[int], Tuple[float, np.ndarray]]
        :param full_obs: Is there a need to apply the observation operator when analysing those, defaults to True
        :type full_obs: bool, optional
        :param mode: weak or strong, defaults to None
        :type mode: string, optional
        :return: Dictionary composed of the analysed states, observations, minimums and time
        :rtype: dict
        """

        if mode is not None:
            self.mode = mode
        observations = []
        xa = []
        J = []
        time = []
        for i in tqdm.trange(Nsteps // self.assimilation_window):
            y_window = []
            t_window = []
            for k in range(
                self.assimilation_window
            ):  # Gather the observation of the window
                t, y = get_obs(i * self.assimilation_window + k)
                t_window.append(t)
                time.append(t)
                observations.append(y)
                if full_obs:
                    y_window.append(self.H(y))
                else:
                    y_window.append(y)
            fto = self.vectorized_fun_to_optim(y_window)
            if self.mode == "weak":
                x_start = self.flatten_input([x0] * self.assimilation_window)
            elif self.mode == "strong":
                x_start = x0
            x, Jx = self.optimization(fto, x_start)

            if self.mode == "strong":
                x_str = [x]
                for j in range(self.assimilation_window - 1):
                    x_str.append(self.forward(x_str[j]))
                x0 = x_str[-1]
                xa.append(np.asarray(x_str))
            elif self.mode == "weak":
                xit = self.iterator_input(x)
                x0 = xit[-1]
                xa.append(xit)
            J.append(Jx)

        return {
            "observations": observations,
            "x": xa,
            "J": J,
            "time": time,
        }
