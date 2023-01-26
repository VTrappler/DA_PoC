"""Module implementing the Kalman Filter (possibly extended if the linearizations are provided"""

import warnings
from logging import warning
from typing import Callable, Tuple, Union

import numpy as np
import scipy.linalg as la
from tqdm.auto import tqdm

from ..common.utils import Kalman_gain


class KalmanFilter:
    def __init__(self, state_dimension: int) -> None:
        self.xf = np.empty((state_dimension, 0))
        self.xa = np.empty((state_dimension, 0))
        self.state_dimension = state_dimension

    @property
    def R(self):
        """Observation error covariance matrix"""
        return self._R

    @R.setter
    def R(self, value):
        self._R = value
        self.Rinv = la.inv(value)
        self.Rchol = la.cholesky(value)

    @property
    def Q(self):
        """Model error covariance matrix"""
        return self._Q

    @Q.setter
    def Q(self, value):
        self._Q = value
        self.Qinv = la.inv(value)
        self.Qchol = la.cholesky(value)

    @property
    def H(self):
        """Observation operator or matrix"""
        return self._H

    @H.setter
    def H(self, value):
        if callable(value):
            warnings.warn(
                "The observation operator has been set with an arbitrary function. Set a linearization of H in self.linearH"
            )
            self._H = value
            self.linearH = None
        elif isinstance(value, np.ndarray):
            self._H = lambda x: value @ x
            self.linearH = lambda x: value

    def set_forwardmodel(self, M):
        self.M = M

    @property
    def M(self):
        """Model operator or matrix"""
        return self._M

    @M.setter
    def M(self, value):
        if callable(value):
            warnings.warn(
                "The forward model has been set with an arbitrary function. Set a linearization of M in self.linearM"
            )
            self._M = value
            self.linearM = None
        elif isinstance(value, np.ndarray):
            self._M = lambda x: value @ x
            self.linearM = lambda x: value

    def set_x0(self, x0, P0):
        self.xf = x0
        self.xa = x0
        self.Pf = P0
        self.Pa = P0

    def analysis(self, obs: Union[np.ndarray, None]) -> None:
        if obs is None:
            self.xa = self.xf
            self.Pa = self.Pf
        else:
            innovation_vector = obs - self.H(self.xf)
            Kstar = Kalman_gain(self.linearH(self.xf), self.Pf, self.R)
            self.xa = self.xf + Kstar @ innovation_vector
            self.Pa = (
                np.eye(self.state_dimension) - Kstar @ self.linearH(self.xf)
            ) @ self.Pf

    def forecast(self) -> None:
        try:
            self.xf = self.M(self.xa)
            self.Pf = self.linearM(self.xf) @ self.Pa @ self.linearM(self.xf).T + self.Q
        except:
            self.xf = self.M(self.xf)
            self.Pf = self.linearM(self.xf) @ self.Pf @ self.linearM(self.xf).T + self.Q

    def run(
        self,
        Nsteps: int,
        get_obs: Callable[[int], Tuple[float, np.ndarray]],
    ) -> dict:
        observations = []
        xf = []
        xa = []
        time = []
        Pa = []
        Pf = []
        for i in tqdm(range(Nsteps)):
            self.forecast()
            xf.append(self.xf)
            Pf.append(self.Pf)
            t, y = get_obs(i)
            observations.append(y)
            time.append(t)
            self.analysis(self.H(y))
            xa.append(self.xa)
            Pa.append(self.Pa)
        return {
            "observations": observations,
            "xf": xf,
            "Pf": Pf,
            "Pa": Pa,
            "xa": xa,
            "time": time,
        }
