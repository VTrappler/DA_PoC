import numpy as np
import scipy.linalg as la

from .VariationalMethod import VariationalMethod, matrix_induced_norm


class ThreeDimensionalVar(VariationalMethod):
    def __init__(self, state_dimension: int, R: np.ndarray, bounds) -> None:
        super().__init__(state_dimension, bounds)
        self.R = R

    def cost_observation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Computes the data misfit
        :param x: state vector
        :type x: np.ndarray
        :param y: observation
        :type y: np.ndarray
        :return: the observation term of the cost function
        :rtype: float
        """
        misfit = y - self.H(x)
        acc = matrix_induced_norm(misfit, self.Rinv)
        return acc * 0.5

    def cost_function_element(self, y: np.ndarray, x: np.ndarray) -> float:
        """Computes the cost function associated with the observation y and the state vector x

        :param y: observation
        :type y: np.ndarray
        :param x: state vector
        :type x: np.ndarray
        :return: cost function J(x)
        :rtype: float
        """
        j_obs = self.cost_observation(x, y)
        j_bck = self.cost_background(x)
        return j_obs + j_bck
