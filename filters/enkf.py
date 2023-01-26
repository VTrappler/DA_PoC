"""This module implements the Ensemble Kalman Filter (Stochastic perturbations of the observation)
"""

from typing import Callable, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from ..common.utils import Kalman_gain
from .ensemblemethod import EnsembleMethod

"""
x_{k+1} = M_k(x_k)  + w_k
y_k = H_k(x_k) + v_k

where Cov[v_k] = R_k, Cov[w_k] = Q_k
"""


class EnKF(EnsembleMethod):
    """Wrapper class for running an EnKF"""

    def __init__(
        self,
        state_dimension: int,
        Nensemble: int,
        R: Optional[np.ndarray] = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        """Initializes an instance of the Ensemble Kalman Filter (Stochastic)

        :param state_dimension: Dimension of the state vector
        :type state_dimension: int
        :param Nensemble: ensemble size
        :type Nensemble: int
        :param R: Covariance of the observation errors, defaults to None
        :type R: Optional[np.ndarray], optional
        :param rng: rng for reproducibility, defaults to np.random.default_rng()
        :type rng: np.random.Generator, optional
        """
        super().__init__(state_dimension, Nensemble)
        if R is not None:
            self.R = R
        self.rng = rng
        self.localization_inflation = None

    # EnKF parameters ---

    def analysis(self, y: np.ndarray, stochastic: bool = True) -> None:
        """Performs the analysis step given the observation

        :param y: Observation to be assimilated
        :type y: np.ndarray
        :param stochastic: Perform Stochastic EnKF, ie perturbates observations, defaults to True
        :type stochastic: bool, optional
        """
        if stochastic:
            # Perturbation of the observed data
            u = self.rng.multivariate_normal(
                mean=np.zeros_like(y), cov=self.R, size=self.Nensemble
            ).T
            # u[:, -1] = -u[:, :-1].sum(1)  # Ensure unbiased sample
            y = y[:, np.newaxis] + u
            self.empirical_R = np.atleast_2d(
                np.cov(u)
            )  # Compute empirical covariance matrix

        if self.localization_inflation is not None:
            self.Pf = (
                self.localization_inflation.rho_loc
                * self.localization_inflation.inflation
                * self.Pf
            )

        Kstar = Kalman_gain(self.linearH, self.Pf, self.empirical_R)
        anomalies_vector = y - self.H(self.xf_ensemble)
        self.xa_ensemble = self.xf_ensemble + Kstar @ anomalies_vector
        try:
            self.xa_ensemble_total = np.concatenate(
                [self.xa_ensemble_total, self.xa_ensemble[:, :, np.newaxis]], 2
            )
        except AttributeError:
            self.xa_ensemble_total = self.xa_ensemble[:, :, np.newaxis]

    def forecast_ensemble(self) -> None:
        """Propagates the ensemble members through the model"""
        try:
            self.xf_ensemble = np.apply_along_axis(
                self.forward,
                axis=0,
                arr=self.xa_ensemble,
            )
        except AttributeError:
            self.xf_ensemble = np.apply_along_axis(
                self.forward,
                axis=0,
                arr=self.xf_ensemble,
            )

        self.xf_ensemble_total = np.concatenate(
            [self.xf_ensemble_total, self.xf_ensemble[:, :, np.newaxis]], 2
        )

    def run(
        self,
        Nsteps: int,
        get_obs: Callable[[int], Tuple[float, np.ndarray]],
        full_obs: bool = True,
    ) -> dict:

        observations = []
        ensemble_f = []
        ensemble_a = []
        time = []
        for i in tqdm(range(Nsteps)):
            self.forecast_ensemble()
            ensemble_f.append(self.xf_ensemble)
            t, y = get_obs(i)
            observations.append(y)
            time.append(t)
            if full_obs:
                self.analysis(self.H(y))
            else:
                self.analysis(y)
            ensemble_a.append(self.xa_ensemble)
        return {
            "observations": observations,
            "ensemble_f": ensemble_f,
            "ensemble_a": ensemble_a,
            "time": time,
        }
