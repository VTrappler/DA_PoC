from typing import Callable, Type, Union
import warnings
import numpy as np
import scipy.linalg as la


def Kalman_gain(H: np.ndarray, Pf: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Computes the Kalman Gain Given the observation matrix, the prior covariance matrix and the error covariance matrix error R
    :param H: Linearized observation operator
    :type H: np.ndarray
    :param Pf: Covariance matrix of the prior error
    :type Pf: np.ndarray
    :param R: Covariance matrix of the observation errors
    :type R: np.ndarray
    :return: Kalman Gain
    :rtype: np.ndarray
    """
    return np.linalg.multi_dot(
        [
            Pf,
            H.T,
            np.linalg.inv(np.linalg.multi_dot([H, Pf, H.T]) + R),
        ]
    )


class DataAssimilationMethod:
    """General class for Data Assimilation methods"""

    def __init__(self, state_dimension: int) -> None:
        self.state_dimension = state_dimension

    @property
    def R(self) -> np.ndarray:
        """Covariance matrix of observation error

        :return: covariance matirx of obs errors.
        :rtype: np.ndarray
        """
        return self._R

    @R.setter
    def R(self, value: np.ndarray) -> None:
        self._R = value
        self.Rinv = la.inv(value)
        self.Rchol = la.cholesky(value)

    @property
    def H(self) -> None:
        """Observation operator

        :return: Observation operator which maps the state space to the observation space
        :rtype: [type]
        """
        return self._H

    @H.setter
    def H(self, value: np.ndarray) -> None:
        if callable(value):
            self._H = value
            self.linearH = None
        elif isinstance(value, np.ndarray):
            self._H = lambda x: value @ x
            self.linearH = value

    def set_forwardmodel(self, model: Callable) -> None:
        """Set the forward operator of tracked dynamical system

        :param model: function which maps the state of the system from x_n to x_n+1
        :type model: Callable
        """
        self.forward = model
