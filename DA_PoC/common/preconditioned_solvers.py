import numpy as np
from .linearsolver import conjGrad
import warnings


class PreconditionedSolver:
    def __init__(self, tol: float = 1e-8, maxiter: int = 100):
        self.tol = tol
        self.maxiter = maxiter

    def __call__(
        self, A: np.ndarray, b: np.ndarray, x: np.ndarray, maxiter: int = None
    ):
        warnings.warn("Subclass to create preconditioner, no preconditioning with this")
        if maxiter is not None:
            self.maxiter = maxiter
        return conjGrad(A, x, b, tol=self.tol, maxiter=self.maxiter)


class PseudoInverseStart(PreconditionedSolver):
    def __init__(self, rank: int = 5, tol: float = 1e-8, maxiter: int = 100):
        super().__init__(tol, maxiter)
        self.rank = rank

    def __call__(self, A, b, x, maxiter):
        U, S, Vt = np.linalg.svd(A)
        A_dagger = U[:, : self.rank] @ (S[: self.rank] ** (-1) * U[:, : self.rank]).T
        return conjGrad(
            A, A_dagger @ b, b, tol=self.tol, maxiter=self.maxiter, verbose=False
        )


class BalancingPrec(PreconditionedSolver):
    def __init__(self, rank: int = 5, tol: float = 1e-8, maxiter: int = 100):
        super().__init__(tol, maxiter)
        self.rank = rank

    def __call__(self, A, b, x, maxiter):
        U, S, Vt = np.linalg.svd(A)
        A_dagger = U[:, : self.rank] @ (
            (S[: self.rank] ** (-1) - 1) * U[:, : self.rank]
        ).T + np.eye(S.shape[0])
        return conjGrad(
            A_dagger @ A,
            0 * b,
            A_dagger @ b,
            tol=self.tol,
            maxiter=self.maxiter,
            verbose=False,
        )

    def get_prec(self, A):
        U, S, Vt = np.linalg.svd(A)
        return U[:, : self.rank] @ (
            (S[: self.rank] ** (-1) - 1) * U[:, : self.rank]
        ).T + np.eye(S.shape[0])

    def power_inv(self, A, power):
        U, S, Vt = np.linalg.svd(A)
        return U[:, : self.rank] @ (
            (S[: self.rank] ** (power) - 1) * U[:, : self.rank]
        ).T + np.eye(S.shape[0])


class SplitPrec(PreconditionedSolver):
    def __init__(self, rank: int = 5, tol: float = 1e-8, maxiter: int = 100):
        super().__init__(tol, maxiter)
        self.rank = rank

    def power_prec(self, A, power):
        U, S, Vt = np.linalg.svd(A)
        return U[:, : self.rank] @ (
            (S[: self.rank] ** (power) - 1) * U[:, : self.rank]
        ).T + np.eye(S.shape[0])

    def __call__(self, A, b, x, maxiter):
        L = self.power_prec(A, power=-0.5)
        # Linv = self.power_prec(A, power=0.5)
        x_hat, res_dict = conjGrad(
            L.T @ A @ L,
            0 * b,
            L.T @ b,
            tol=self.tol,
            maxiter=self.maxiter,
            verbose=False,
        )
        return L @ x_hat, res_dict


if __name__ == "__main__":
    A_ = np.random.normal(size=(20, 20))
    A = A_.T @ A_
    x = np.arange(20)
    SplitPrec()(A, A @ x, None, None)
