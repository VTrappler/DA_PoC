"""This module implements the base class for particle filters
"""


from typing import Callable, Tuple, Union

import numpy as np


class BaseParticleFilter:
    """Implementation of the standard particle filter, where the prior is chosen as sampling density,
    and the likelihood is used to reweight"""

    def __init__(
        self,
        state_dimension: int,
        Nparticles: int,
        R: np.ndarray,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        self._state_dimension = state_dimension
        self._Nparticles = Nparticles
        self._R = R
        self.rng = rng

    @property
    def Nparticles(self) -> int:
        return self._Nparticles

    @property
    def state_dimension(self) -> int:
        return self._state_dimension

    @property
    def R(self) -> np.ndarray:
        """Observation error covariance matrix"""
        return self._R

    @property
    def H(self):
        """Observation operator or matrix"""
        return self._H

    @H.setter
    def H(self, value: Union[np.ndarray, callable]):
        if callable(value):
            self._H = value
        elif isinstance(value, np.ndarray):
            self._H = lambda x: value @ x
            self.linearH = value

    @classmethod
    def weighted_moments(
        cls, X: np.ndarray, weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and variance using the weights

        :param X: Particles
        :type X: np.ndarray
        :param weights: associated weights
        :type weights: np.ndarray
        :return: weighted average and variance
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        mean = np.average(X, weights=weights, axis=1)
        var = np.average(
            (X - mean.reshape(-1, 1)) ** 2,
            weights=weights,
            axis=1,
        )
        return mean, var

    def generate_particles(self, mean: np.ndarray, cov: np.ndarray) -> None:
        """Generate the particles according to a MVN distribution

        :param mean: mean of the particle to sample
        :type mean: np.ndarray
        :param cov: covariance matrix from which to draw
        :type cov: np.ndarray
        """
        self.particles = self.rng.multivariate_normal(
            mean=mean,
            cov=cov,
            size=self.Nparticles,
        ).T
        self.weights = np.ones(self.Nparticles) / self.Nparticles

    def set_forwardmodel(self, model: Callable) -> None:
        self.forward = model

    def normalize_weights(self) -> None:
        self.weights = self.weights / sum(self.weights)

    def propagate_particles(self) -> None:
        """Apply the forward model to the particles"""
        self.particles = np.apply_along_axis(
            self.forward,
            axis=0,
            arr=self.particles,
        )

    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the estimates based on the particles"""
        return self.weighted_moments(
            self.particles,
            self.weights,
        )

    def get_ESS(self) -> float:
        """Compute the Effective Sample Size of the particles"""
        return 1.0 / ((self.weights ** 2).sum())

    def resample_particles(self) -> None:
        """Resample particles, straightforward algorithm"""
        new_particles = self.rng.choice(
            a=self.Nparticles, size=self.Nparticles, p=self.weights, replace=True
        )
        self.particles = self.particles[:, new_particles]
        self.weights = np.ones(self.Nparticles) / self.Nparticles

    def resample_particles_systematic(self) -> None:
        """Resample the particles using systematic resampling"""
        cumulative_weights = np.cumsum(self.weights)
        uj = (
            np.arange(self.Nparticles) / self.Nparticles
            + self.rng.uniform() / self.Nparticles
        )
        new_particles = np.digitize(uj, cumulative_weights)
        self.particles = self.particles[:, new_particles]
        self.weights = np.ones(self.Nparticles) / self.Nparticles

    def resample_particles_RSR(self) -> None:
        """Resample the particles using RSR"""
        dU0 = self.rng.uniform() / self.Nparticles
        new_particles = np.empty(self.Nparticles, dtype="int")
        dUprev = dU0
        for m, w in enumerate(self.weights):
            new_particles[m] = (np.floor((w - dUprev) * self.Nparticles) + 1).astype(
                "int"
            )
            dUprev = dUprev + (new_particles[m] / self.Nparticles) - w
        print(new_particles.min(), new_particles.max())
        self.particles = self.particles[:, new_particles]
        self.weights = np.ones(self.Nparticles) / self.Nparticles
