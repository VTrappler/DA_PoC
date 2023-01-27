"""This module implements the Ensemble Transform Kalman Filter, but also the ETKF-Q
"""
from typing import Callable, Tuple

import numpy as np
import scipy.linalg as la
from numpy.linalg import multi_dot
from tqdm.auto import tqdm

from .ensemblemethod import EnsembleMethod


class ETKF(EnsembleMethod):
    """Wrapper class for running an Ensemble Transform Kalman Filter"""

    def __init__(
        self,
        state_dimension: int,
        Nensemble: int,
        R: np.ndarray,
        inflation_factor: float = 1.0,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(state_dimension, Nensemble)
        if R is not None:
            self.R = R
        self.inflation_factor = inflation_factor

    def state_anomalies(self) -> np.ndarray:
        """Computes the state normalised anomalies x - xbar/ sqrt(N-1)

        :return: matrix of state anomalies X of dim (state_dim * Nensemble)
        :rtype: np.ndarray
        """
        return (self.xf_ensemble - self.xf_ensemble.mean(1, keepdims=True)) / np.sqrt(
            self.Nensemble - 1
        )

    def observation_anomalies(self) -> np.ndarray:
        """Computes the observation normalised anomalies H(x) - Hbar(x) / sqrt(N-1)

        :return: matrix of state anomalies Y of dim (obs_dim * Nensemble)
        :rtype: np.ndarray
        """
        Hxf = self.H(self.xf_ensemble)
        Yf = (Hxf - Hxf.mean(1, keepdims=True)) / np.sqrt(self.Nensemble - 1)
        precYf = la.solve_triangular(self.Rchol, Yf)
        return Yf, precYf

    def compute_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the transform matrix using "naive" method

        :return: transform matrix T and its square
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        Yf, _ = self.observation_anomalies()
        Tsqm1 = np.eye(self.Nensemble) + Yf.T @ self.Rinv @ Yf
        print(f"{la.det(Tsqm1)=}")
        Tsq = la.inv(Tsqm1)
        return la.sqrtm(Tsq), Tsq

    def compute_transform_SVD(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the transform matrix using SVD

        :return: Transform matrix, and the SVD of the preconditionned Yf
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        _, Yfhat = self.observation_anomalies()
        U, Sigma, VT = la.svd(Yfhat.T)
        if self.linearH.shape[0] < self.Nensemble:
            Lm = np.concatenate(
                [
                    1 / np.sqrt(1 + Sigma ** 2),
                    np.ones(self.Nensemble - self.linearH.shape[0]),
                ]
            )
            # Sigma = np.concatenate(
            #     [Sigma, np.zeros(self.Nensemble - self.linearH.shape[0])]
            # )
        else:
            Lm = 1 / np.sqrt(1 + Sigma ** 2)
        T = (U * Lm) @ U.T
        return T, U, Sigma, VT

    def analysis(self, obs: np.ndarray) -> None:
        """Performs the analysis step given the observation

        :param obs: Observation to be assimilated
        :type obs: np.ndarray
        """
        innovation_vector = obs - (self.H(self.xf_ensemble)).mean(1)
        # Yf, Yfhat = self.observation_anomalies()
        transform_matrix, U, Sigma, VT = self.compute_transform_SVD()
        omega = transform_matrix @ transform_matrix.T
        # wa = T2 @ Yf.T @ la.solve(self.R, innovation_vector, sym_pos=True)
        # wa = omega @ Yf.T @ self.Rinv @ innovation_vector
        xfbar = self.xf_ensemble.mean(1, keepdims=True)
        Xa = np.sqrt(self.Nensemble - 1) * self.state_anomalies() @ transform_matrix
        # print(f"{transform_matrix.shape=}, {wa.shape=}")

        wa = multi_dot(
            [
                U,
                la.diagsvd(Sigma, self.Nensemble, self.linearH.shape[0]),
                np.diag(1 / (1 + Sigma ** 2)),
                VT,
                la.solve_triangular(self.Rchol, innovation_vector),
            ]
        )

        # print(f"{wa.shape=}")
        # print(f"{xfbar.shape=}")
        # print(f"{self.state_anomalies().shape=}")
        # print(f"{Xa.shape=}")
        xabar = xfbar + self.state_anomalies() @ wa[:, np.newaxis]
        # self.xa_ensemble = xfbar + self.inflation_factor * self.state_anomalies() @ (
        #     wa + np.sqrt(self.Nensemble - 1) * transform_matrix
        # )
        self.xa_ensemble = xabar + Xa
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
        verbose: bool = True,
    ) -> dict:
        """Run the filter

        :param Nsteps: Number of assimilation steps to perform
        :type Nsteps: int
        :param get_obs: Function which provides the (time, observation) tuple
        :type get_obs: Callable[[int], Tuple[float, np.ndarray]]
        :param full_obs: Does the obs operator needs to be applied before the analysis, defaults to True
        :type full_obs: bool, optional
        :param verbose: tqdm progress bar, defaults to True
        :type verbose: bool, optional
        :return: Dictionary containing the ensemble members, analised or not
        :rtype: dict
        """
        observations = []
        ensemble_f = []
        ensemble_a = []
        time = []
        if not verbose:
            iterator = range(Nsteps)
        else:
            iterator = tqdm(range(Nsteps))
        for i in iterator:
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


class ETKFQ(ETKF):
    """Wrapper class for running an ETKF-Q, meaning that we assume error in the model, with covariance matrix Q"""

    def __init__(
        self,
        state_dimension: int,
        Nensemble: int,
        R: np.ndarray,
        Q: np.ndarray,
        inflation_factor: float = 1,
    ) -> None:
        if state_dimension < Nensemble:
            raise ValueError(
                "State dimension should be larger than the ensemble number"
            )
        super().__init__(state_dimension, Nensemble, R, inflation_factor)
        self.Q = Q
        self.U, self.Uinv = self.constructU(self.Nensemble)

    @classmethod
    def constructU(cls, m: int) -> Tuple[np.ndarray, np.ndarray]:
        Um = np.zeros((m, m - 1))
        for i in range(m - 1):
            Um[: (i + 1), i] = 1
            Um[i + 1, i] = -(i + 1)
            Um[(i + 2) :, i] = 0
        Um = Um / np.sqrt((Um ** 2).sum(0, keepdims=True))
        U = np.concatenate([np.ones((m, 1)) / m, Um / np.sqrt(m - 1)], axis=1)
        Uinv = np.concatenate([np.ones((m, 1)), Um * np.sqrt(m - 1)], axis=1)
        return U, Uinv

    def update_ensemble(self) -> np.ndarray:
        """Update ensemble using the deviation matrix taking into account the model error

        :return: updated ensemble
        :rtype: np.ndarray
        """

        print(f"{self.U.shape=}")
        m = self.Nensemble
        print(f"{m=}")
        tmp = self.xf_ensemble @ self.U
        xkbar, deviation_mat = tmp[:, 0], tmp[:, 1:]
        eivals, Vk = np.linalg.eigh(deviation_mat @ deviation_mat.T + self.Q)
        # print(f"{eivals.shape=}")
        # print(f"{Vk.shape=}")
        # print(f"{eivals=}")
        truncated_eigvals = eivals[:-(m):-1]
        print(f"{truncated_eigvals=}")
        Lambdak = np.diag(truncated_eigvals)  # Keep m-1 largest eigenvalues
        # print(f"{Lambdak.shape=}")
        Vk = Vk[:, :-(m):-1]
        # print(f"{Vk.shape=}")
        new_deviation_matrix = Vk @ np.sqrt(Lambdak)
        # print(f"{xkbar.shape=}")
        # print(f"{deviation_mat.shape=}")
        # print(f"{new_deviation_matrix.shape=}")
        # print(f"{xkbar[:, np.newaxis].shape=}")

        Ek = (
            np.concatenate([xkbar[:, np.newaxis], new_deviation_matrix], axis=1)
            @ self.Uinv
        )
        return Ek

    def run(
        self,
        Nsteps: int,
        get_obs: Callable[[int], Tuple[float, np.ndarray]],
        full_obs: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Run the filter

        :param Nsteps: Number of assimilation steps to perform
        :type Nsteps: int
        :param get_obs: Function which provides the (time, observation) tuple
        :type get_obs: Callable[[int], Tuple[float, np.ndarray]]
        :param full_obs: Does the obs operator needs to be applied before the analysis, defaults to True
        :type full_obs: bool, optional
        :param verbose: tqdm progress bar, defaults to True
        :type verbose: bool, optional
        :return: Dictionary containing the ensemble members, analised or not
        :rtype: dict
        """
        if verbose:
            iterator = tqdm(range(Nsteps))
        else:
            iterator = range(Nsteps)

        observations = []
        ensemble_f = []
        ensemble_a = []
        time = []
        for i in iterator:
            self.forecast_ensemble()
            self.xf_ensemble = self.update_ensemble()
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
