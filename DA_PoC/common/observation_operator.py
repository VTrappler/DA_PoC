from __future__ import nested_scopes
import abc
from typing import Any, Callable, List, Optional
import numpy as np
import scipy.sparse.linalg as sla


class AbstractObsOperator(abc.ABC):
    def H(self, x: np.ndarray, i: int) -> np.ndarray:
        """Observation operator, call as H(x, i) = y

        :param x: state vector
        :type x: np.ndarray
        :param i: index of time step
        :type i: int
        :return: observed vector
        :rtype: np.ndarray
        """
        pass

    def linearH(self, x: np.ndarray, i: int) -> np.ndarray:
        """Linearized Observation Operator: linearH(x,i)

        :param x: state vector
        :type x: np.ndarray
        :param i: time index
        :type i: int
        :return: linearized observation operator
        :rtype: np.ndarray
        """
        pass


def create_projector(n: int, m: int, indices: Optional[Any] = None) -> np.ndarray:
    """Create a projector from a space of n dimension to m, where the identity is applied to the indices

    :param n: input dimension
    :type n: int
    :param m: output dimension
    :type m: int
    :param indices: indices to keep
    :type indices: Iterable
    :return: Projection matrix
    :rtype: np.ndarray
    """
    if indices is None:
        indices = np.arange(m)
    H = np.zeros((m, n))
    for i in indices:
        H[i, i] = 1
    return H


def create_projector_random(
    n: int, m: int, n_samples: int, rng: np.random.Generator = np.random.default_rng()
) -> np.ndarray:
    """Create a projector which chooses randomly n_samples coordinates to keep

    :param n: input dimension
    :type n: int
    :param m: output dimension
    :type m: int
    :param n_samples: number of components to keep
    :type n_samples: int
    :param rng: random number generator to use, defaults to np.random.default_rng()
    :type rng: np.random.Generator, optional
    :return: random projector
    :rtype: np.ndarray
    """
    rnd_indices = rng.choice(np.arange(m), size=n_samples, replace=False)
    return create_projector(n, m, rnd_indices)


def H_nl(x: np.ndarray, i: int) -> np.ndarray:
    """Example of non-linear Observation operator

    :param x: State vector
    :type x: np.ndarray
    :param i: index of time step
    :type i: int
    :return: observed state vector
    :rtype: np.ndarray
    """
    return np.array([(x[0] + x[1] - i) ** 2, x[1] * x[2]])


def H_nl_linear(x: np.ndarray, i: int) -> np.ndarray:
    """Linearized Observation operator

    :param x: State vector
    :type x: np.ndarray
    :param i: index of time step
    :type i: int
    :return: Jacobian matrix of the observation operator
    :rtype: np.ndarray
    """
    return np.array(
        [
            [2 * (x[0] + x[1] - i), 2 * (x[0] + x[1]), 0],
            [0, x[2], x[1]],
        ]
    )


# class ObservationOperator(AbstractObsOperator):
#     def __init__(self, n: int, m: int) -> None:
#         """Initialize an observation operator, which maps the state space of dimension n to the observation space of dimension m
#             The Observation operator is a function of the form H(x, i) -> y, where i indicates the time step
#         :param n: state vector dimension
#         :type n: int
#         :param m: observation dimension
#         :type m: int
#         """

#         self.n = n
#         self.m = m

#     def set_H(
#         self, observation_operator: Callable[[np.ndarray, int], np.ndarray]
#     ) -> None:
#         self.H = observation_operator
#         self.H.__doc__ = super().H.__doc__

#     def set_linearizedH(
#         self, linearized_observation_operator: Callable[[np.ndarray, int], np.ndarray]
#     ) -> None:
#         self.linearH = linearized_observation_operator
#         self.linearH.__doc__ = super().linearH.__doc__


class ObservationOperator:
    def __init__(self, n: int, m: int) -> None:
        self.n = n
        self.m = m

    def set_operator(self, operator: Callable[[np.ndarray], np.ndarray]) -> None:
        self.operator = operator

    def set_linearized(self, linfunc: Callable[[np.ndarray], np.ndarray]) -> None:
        self.linearized_observation = linfunc

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.operator(x)

    def linearized_operator(self, x: np.ndarray) -> sla.LinearOperator:
        return sla.aslinearoperator(self.linearized_observation(x))

    def test_obs_and_linearization(self) -> None:
        pass


class LinearObservationOperator(ObservationOperator):
    def __init__(self, Hmatrix: np.ndarray) -> None:
        """
        :param n: input dimension
        :type n: int
        :param m: output dimension
        :type m: int
        """
        m, n = Hmatrix.shape
        super().__init__(n, m)
        self.n = n
        self.m = m
        self.H = Hmatrix

    @property
    def H(self) -> np.ndarray:
        return self._H

    @H.setter
    def H(self, value: np.ndarray) -> None:
        assert value.shape == (self.m, self.n)
        self._H = value
        self.set_operator(sla.aslinearoperator(self._H))

    def linearized_operator(self, x: np.ndarray) -> sla.LinearOperator:
        return sla.aslinearoperator(self._H)


class RandomObservationOperator(ObservationOperator):
    def __init__(
        self, n: int, m: int, type: str, p: float, p_offdiag: Optional[float] = None
    ) -> None:
        super().__init__(n, m)
        self.n = n
        self.m = m
        self.prob = p
        self.p_offdiag = p_offdiag
        self.type = type
        self.generate_H(self.type)

    # def generate_H(self) -> None:
    #     self.H = np.random.binomial(n=1, p=self.prob, size=self.n * self.m).reshape(self.m, self.n)
    #     self.H = 100 * self.H / np.sum(self.H)

    def generate_H(self, type: str) -> None:
        if type == "square":
            self.H = np.diag(np.random.binomial(n=1, p=self.prob, size=self.m))
            if self.p_offdiag is not None:
                self.H = self.H + np.random.binomial(
                    n=1, p=self.p_offdiag, size=self.m * self.m
                ).reshape(self.m, self.m)
        elif type == "rect":
            self.H = np.zeros((self.m, self.n))
            for i in range(self.m):
                self.H[i, i] = np.random.binomial(n=1, p=self.prob, size=1)
        elif type == "identity":
            self.H = np.eye(self.m)

    @property
    def H(self) -> np.ndarray:
        return self._H

    @H.setter
    def H(self, value: np.ndarray) -> None:
        assert value.shape == (self.m, self.n)
        self._H = value
        self.set_operator(sla.aslinearoperator(self._H))

    def linearized_operator(self, x: np.ndarray) -> sla.LinearOperator:
        return sla.aslinearoperator(self._H)


class IdentityObservationOperator(LinearObservationOperator):
    def __init__(self, dim: int) -> None:
        super().__init__(np.eye(dim))


class TimeStateIndicesObservationOperator(LinearObservationOperator):
    def __init__(
        self,
        state_dimension: int,
        window: int,
        idx_observed_states: List[int],
        idx_observed_timesteps: List[int],
    ) -> None:
        """Create observation matrix of dimension (len(obs_states) * len(obs_tsteps)) x dim * (window+1)

        :param state_dimension: State vector dimension
        :type state_dimension: int
        :param window: length of the window (observation of length window + 1)
        :type window: int
        :param idx_observed_states: indices of observed states
        :type idx_observed_states: List[int]
        :param idx_observed_timesteps: indices of observed time steps
        :type idx_observed_timesteps: List[int]
        """
        self.state_dimension = state_dimension
        self.window = window
        self.idx_observed_states = idx_observed_states
        self.idx_observed_timesteps = idx_observed_timesteps

        obs_state = np.zeros((len(self.idx_observed_states), self.state_dimension))
        for i, idx in enumerate(self.idx_observed_states):
            obs_state[i, idx] = 1.0

        obs_steps = np.zeros((len(self.idx_observed_timesteps), (self.window + 1)))
        for i, idx in enumerate(idx_observed_timesteps):
            obs_steps[i, idx] = 1.0

        obs_H = np.kron(obs_state, obs_steps)

        super().__init__(Hmatrix=obs_H)


# def main():
#     n, m = 10, 6
#     H = create_projector_random(n, m, n_samples=4)
#     linear_observator = LinearObservationOperator(n, m, H)
#     # print(linear_observator.linearH(np.arange(3), 0))
#     # print(linear_observator.Hmat)
#     print(help(linear_observator.linearH))

#     nnlin_observator = ObservationOperator(3, 2)
#     nnlin_observator.set_H(H_nl)
#     nnlin_observator.set_linearizedH(H_nl_linear)
#     x = np.arange(3)
#     print(f"{nnlin_observator.H(x, 0)=}")
#     print(f"{nnlin_observator.linearH(x, 0)=}")
#     print(f"{nnlin_observator.H(x, 1)=}")
#     print(f"{nnlin_observator.linearH(x, 1)=}")
#     print(help(nnlin_observator.linearH))
#     print(help(nnlin_observator.H))

#     nn_smooth = ObservationOperator(3, 2)

#     def H_nnsmooth(x: np.ndarray):
#         delta = 1e-4
#         x0 = 2.0
#         if x[0] < x0:
#             y0 = x[0] ** 3 / x0 ** 2
#         else:
#             y0 = x[0] ** 2 / x0
#         if x[1] >= 0:
#             y1 = np.log(x[1] + delta)
#         else:
#             y1 = np.log(-x[1] + delta)
#         return np.array([y0, y1])

#     def H_nnsmooth_linearized(x: np.ndarray):
#         delta = 1e-4
#         x0 = 2.0
#         if x[0] < x0:
#             y0 = 3 * x[0] ** 2 / x0 ** 2
#         else:
#             y0 = 2 * x[0] / x0
#         if x[1] >= 0:
#             y1 = 1 / (x[1] + delta)
#         else:
#             y1 = -1 / (-x[1] + delta)
