"""Implements the standard particle filter, where the weights update is based on the likelihood"""

from typing import Callable, Optional, Tuple

import numpy as np
import scipy.stats
from tqdm.auto import tqdm

from .baseparticlefilter import BaseParticleFilter

# import tqdm


class BootstrapPF(BaseParticleFilter):
    """Implementation of the standard particle filter,
    where the prior is chosen as sampling density, and the likelihood is used to reweights"""

    def __init__(
        self,
        state_dimension: int,
        Nparticles: int,
        R: np.ndarray,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(state_dimension, Nparticles, R, rng)

    def update_weights(self, y: np.ndarray) -> None:
        """Update the weights using the likelihood (Standard PF/Bootstrap Bayesian Filtering)

        :param y: observation
        :type y: np.ndarray
        """
        lik = np.ones(self.Nparticles)
        for i, part in enumerate(self.particles.T):
            dist = y - self.H(part)
            lik[i] = scipy.stats.multivariate_normal(np.zeros(len(dist)), self.R).pdf(
                dist
            )
        self.weights = self.weights * lik

    def run(
        self,
        Nsteps: int,
        get_obs: Callable[[int], Tuple[float, np.ndarray]],
        full_obs: bool = True,
        ESS_lim: Optional[float] = None,
    ) -> dict:
        """Run the standard PF

        :param Nsteps: Assimilation steps
        :type Nsteps: int
        :param get_obs: callable which gives the observation at i
        :type get_obs: Callable[[int], Tuple[float, np.ndarray]]
        :param full_obs: Observation operator needs to be applied, defaults to True
        :type full_obs: bool, optional
        :param ESS_lim: Given threshold for resampling, defaults to None
        :type ESS_lim: Optional[float], optional
        :return: Dictionary containing all the filtered data
        :rtype: dict
        """
        if ESS_lim is None:
            ESS_lim = 0.6 * self.Nparticles
        observations = []
        estimates = []
        particles = []
        time = []
        weights = []
        for i in tqdm(range(Nsteps)):
            particles.append(self.particles)
            weights.append(self.weights)
            self.propagate_particles()

            # Get observations
            t, obs = get_obs(i)
            observations.append(obs)
            time.append(t)
            if full_obs:
                self.update_weights(self.H(obs))
            else:
                self.update_weights(obs)
            self.normalize_weights()

            estimates.append(self.estimate())
            if self.get_ESS() < ESS_lim:
                self.resample_particles_systematic()
        return {
            "observations": observations,
            "estimates": estimates,
            "particles": particles,
            "time": time,
            "weights": weights,
        }
