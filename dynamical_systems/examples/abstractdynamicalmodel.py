from abc import ABC, abstractmethod
import numpy as np


class DynamicalModel(ABC):
    """
    Abstract class implementing a dynamical model
    A dynamical model is defined as a forward operator, which maps the x_n and x_n+1

    """

    def set_initial_state(
        self, t0: float, x0: np.ndarray, force_reset: bool = False
    ) -> None:
        """Set state vector and time to specific value

        :param t0: initial time
        :type t0: float
        :param x0: initial state vector
        :type x0: np.ndarray
        :param force_reset: erase current state vector if it exists, defaults to False
        :type force_reset: bool, optional
        :raises Exception:
        """
        if hasattr(self, "state_vector") and not force_reset:
            raise Exception(
                "This model has already been forwarded. Set 'force_reset' to True in order to overwrite"
            )
        else:
            self.state_vector = x0.reshape(self.dim, 1)
            self.t = np.array(t0).reshape(1)

    def forward(self, Nsteps: int) -> None:
        """Run the forward model a prescribed number of time steps

        :param Nsteps: Number of time steps
        :type Nsteps: int
        """
        t0 = self.t[-1]
        x0 = self.state_vector[:, -1]
        t_, x_ = self.integrate(t0, x0, Nsteps)
        self.t = np.concatenate([self.t, t_[1:]])
        self.state_vector = np.concatenate([self.state_vector, x_[:, 1:]], axis=1)

    @abstractmethod
    def integrate(cls, t0: float, x0: np.ndarray, Nsteps: int) -> np.ndarray:
        """Integrate the state equation

        :param t0: Initial time
        :type t0: float
        :param x0: Initial state vector
        :type x0: np.ndarray
        :param Nsteps: Number of time steps of integration
        :type Nsteps: int
        :return: The evolution of the state vector
        :rtype: np.ndarray
        """

        pass

    def integrate_tlm(cls, t0: float, x0: np.ndarray, Nsteps:int) -> np.ndarray:
        pass
