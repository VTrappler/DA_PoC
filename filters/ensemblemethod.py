"""This module implements the base class for Ensemble methods"""


from typing import Callable, Union

import numpy as np
import scipy.linalg as la


class EnsembleMethod:
    def __init__(self, state_dimension: int, Nensemble: int) -> None:
        self.state_dimension = state_dimension
        self.Nensemble = Nensemble

    @property
    def R(self):
        """Observation error covariance matrix"""
        return self._R

    @R.setter
    def R(self, value: np.ndarray) -> None:
        self._R = value
        self.Rinv = la.inv(value)
        try:
            self.Rchol = la.cholesky(value)
        except la.LinAlgError:
            pass

    @property
    def H(self) -> None:
        """Observation operator or matrix"""
        return self._H

    @H.setter
    def H(self, value: Union[Callable, np.ndarray]) -> None:
        if callable(value):
            self._H = value
            self.linearH = None
        elif isinstance(value, np.ndarray):
            self._H = lambda x: value @ x
            self.linearH = value

    # Methods for manipulating ensembles ---
    @property
    def xf_ensemble(self) -> np.ndarray:
        return self._xf_ensemble

    @xf_ensemble.setter
    def xf_ensemble(self, xf_i: np.ndarray):
        self._xf_ensemble = xf_i
        self.Pf = np.cov(xf_i)

    @property
    def xa_ensemble(self) -> np.ndarray:
        return self._xa_ensemble

    @xa_ensemble.setter
    def xa_ensemble(self, xa_i: np.ndarray) -> None:
        self._xa_ensemble = xa_i
        self.Pa = np.cov(xa_i)

    def set_forwardmodel(self, model: Callable) -> None:
        self.forward = model

    def generate_ensemble(
        self, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Generation of the ensemble members, using a multivariate normal rv

        :param mean: mean of the ensemble members
        :type mean: np.ndarray
        :param cov: Covariance matrix
        :type cov: np.ndarray
        """
        self.xf_ensemble = rng.multivariate_normal(
            mean=mean, cov=cov, size=self.Nensemble
        ).T
        self.xf_ensemble_total = self.xf_ensemble[:, :, np.newaxis]

    def set_localization_inflation(self, localization_inflation):
        self.localization_inflation = localization_inflation


def rho_localization(d: np.ndarray, lamb: float) -> np.ndarray:
    a = np.sqrt(10.0 / 3.0) * lamb
    r = d / a
    rho = np.zeros_like(d)
    rho = np.where(
        r < 1,
        -(r ** 5) / 4.0 + (r ** 4) / 2.0 + (r ** 3) * 5 / 8.0 - (r ** 2) * 5 / 3.0 + 1,
        (r ** 5) / 12
        - (r ** 4) / 2
        + (r ** 3) * 5 / 8.0
        + r ** 2 * 5 / 3.0
        - 5 * r
        + 4
        - (2 / 3.0) * r ** (-1),
    )
    rho = np.where(r >= 2.0, 0, rho)
    return rho


def distance_periodic(n: int) -> np.ndarray:
    dist = np.zeros((n, n))
    for i in range(n - 1):
        cst = np.amin([i + 1, n - (i + 1)])
        dist += np.diag(cst * np.ones(n - i - 1), k=i + 1)
        dist += np.diag(cst * np.ones(n - i - 1), k=-i - 1)
    return dist


class LocalizationInflation:
    def __init__(self, inflation, state_dimension, lamb) -> None:
        self.inflation = inflation
        self.state_dimension = state_dimension
        self.lamb = lamb
        self.dist_mat = distance_periodic(self.state_dimension)
        self.rho_loc = rho_localization(self.dist_mat, self.lamb)
