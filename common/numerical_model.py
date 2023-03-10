import scipy.sparse.linalg as sla
import numpy as np

from .observation_operator import ObservationOperator
from .linearsolver import solve_cg, solve_cg_jacobi, solve_cg_LMP
from typing import Callable, Tuple, Optional, List, Union


class NumericalModel:
    """Class wrapping Numerical models
    with TLM, Adjoint etc
    """

    def __init__(self, n: int, m: int) -> None:
        """Initialize a Numerical Model instance"""
        self.n = n
        self.m = m
        self.obs_operator = None
        self.background_error_cov_inv = None
        self.background_error_sqrt = None
        self.background = None
        # Type hint
        self.tangent_linear: Callable[[np.ndarray], sla.LinearOperator]
        self.adjoint: Callable[[np.ndarray], sla.LinearOperator]

    def set_obs(self, y: np.ndarray) -> None:
        """Set the observations

        :param y: np array that is to be considered as observations
        :type y: np.ndarray
        """
        self.obs = y

    def set_observation_operator(self, obs_operator: ObservationOperator) -> None:
        """Set observation operator

        :param obs_operator: Observation operator to use
        :type obs_operator: ObservationOperator
        """
        self.obs_operator = obs_operator

    def set_forward(self, forw: Callable[[np.ndarray], np.ndarray]) -> None:
        """Set forward model operator (M) as self.forward and the forward model operator observed (H \circ M) as self.forward_no_obs

        :param forw: forward model as function
        :type forw: Callable[[np.ndarray], np.ndarray]
        """
        if self.obs_operator is None:
            self.forward = forw
            self.forward_no_obs = forw
        else:
            self.forward_no_obs = forw
            self.forward = lambda x: self.obs_operator(forw(x))

    def set_forward_no_obs(self, forw: Callable[[np.ndarray], np.ndarray]) -> None:
        self.forward_no_obs = forw

    def set_tangent_linear(self, tlm: Callable) -> None:
        """Set Tangent Linear operator, and adjoint as side effect. If observation operator is set, compose with tangent linear of the observation operator

        :param tlm: Function which is the jacobian of the argument
        :type tlm: Callable
        """
        if self.obs_operator is None:
            self.tangent_linear = lambda x: sla.aslinearoperator(tlm(x))
            self.adjoint = lambda x: sla.aslinearoperator(tlm(x)).adjoint()
        else:
            self.tangent_linear = lambda x: self.obs_operator.linearized_operator(
                x
            ) @ sla.aslinearoperator(tlm(x))
            self.adjoint = (
                lambda x: sla.aslinearoperator(tlm(x)).adjoint()
                @ self.obs_operator.linearized_operator(x).adjoint()
            )

    def jac_mat(self, x: np.ndarray) -> np.ndarray:
        """Returns the jacobian matrix of the model at x

        :param x: Linearization point
        :type x: np.ndarray
        :return: Jacobian matrix
        :rtype: np.ndarray
        """
        return self.tangent_linear(x).matmat(np.eye(self.n))

    def adj_mat(self, x: np.ndarray) -> np.ndarray:
        """Returns the adjoint of the jacobian matrix of the model at x

        :param x: Linearization point
        :type x: np.ndarray
        :return: Adjoint of the jacobian matrix
        :rtype: np.ndarray
        """
        return self.tangent_linear(x).rmatmat(np.eye(self.m))

    def set_adjoint(self, adj: Callable) -> None:
        """Set adjoint

        :param adj: Set the adjoint of the model as operator
        :type adj: Callable
        """

        if self.obs_operator is None:
            self.adjoint = lambda x: sla.aslinearoperator(adj(x))
        else:
            self.adjoint = (
                lambda x: sla.aslinearoperator(adj(x))
                @ self.obs_operator.linearized_operator(x).adjoint()
            )

    def data_misfit(self, x: np.ndarray) -> np.ndarray:
        """Returns the data misfit M(x) - y

        :param x: Point to consider
        :type x: np.ndarray
        :return: M(x) - y
        :rtype: np.ndarray
        """

        return self.forward(x) - self.obs

    def background_cost(self, x: np.ndarray) -> float:
        """Computes 2 * Jb = |x-xb|**2 according to the norm defined by B^{-1}

        :param x: Point at which we evaluate this cost
        :type x: np.ndarray
        :raises RuntimeError: Background informations are not set
        :return: Cost associated with background error
        :rtype: float
        """
        try:
            return (
                (x - self.background).T
                @ self.background_error_cov_inv
                @ (x - self.background)
            )
        except NameError as e:
            raise RuntimeError(
                f"Background error covariance matrix or background value not set"
            )

    def cost_function(self, x: np.ndarray) -> np.ndarray:
        """Computes the 4DVar cost function

        :param x: Points to evaluate
        :type x: np.ndarray
        :return: objective associated with the 4DVar cost function
        :rtype: np.ndarray
        """
        # return 0.5 * ((data_misfit(x))**2).sum()
        diff = self.data_misfit(x)
        if self.background_error_cov_inv is None:
            return 0.5 * diff.T @ diff
        else:
            return 0.5 * diff.T @ diff + 0.5 * self.background_cost(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        if self.background_error_cov_inv is None:
            prior_jac = 0
        else:
            prior_jac = self.background_error_cov_inv @ (x - self.background)
        return self.adjoint(x) @ self.data_misfit(x) + prior_jac

    def gauss_newton_hessian_matrix(
        self, x: np.ndarray, bck_prec: bool = False
    ) -> Union[np.ndarray, sla.LinearOperator]:
        H_GN = self.adjoint(x) @ (self.jac_mat(x))
        if (self.background_error_cov_inv is None) or (bck_prec):
            return H_GN
        else:
            return self.background_error_cov_inv + H_GN

    def test_forw_tlm_alpha(self, alpha: float) -> float:
        """Test consistency between TLM and forward model
        computes |G(x + alpha dx) - G(x)| / |alpha * TLM(x, dx)|
        """
        x = np.random.normal(size=self.n)
        dx = np.random.normal(size=self.n)
        fd = (self.forward(x + alpha * dx) - self.forward(x)) / alpha
        tlm = self.tangent_linear(x).matvec(dx)
        return np.linalg.norm(fd) / (np.linalg.norm(tlm))

    def test_consistency_tlm_forward(self, plot=True) -> Tuple[np.ndarray, List]:
        """
        "Test Forw/TLM
        |Forward(x + a*dx) - Forward(x)| / | (a*TLM(x, dx)) | - 1 ->  0 as a -> 0
        """
        print("Test Forw/TLM")
        rat = []
        alpha = np.logspace(0, -14, 15, base=10)
        # print(f"     a, ratio")
        for al in alpha:
            ratio = np.abs(self.test_forw_tlm_alpha(al) - 1)
            rat.append(ratio)
            # print(f"{al:<6}, {ratio:>8}")
        if plot:
            import matplotlib.pyplot as plt

            plt.plot(alpha, rat)
            plt.title("Forward/TLM consistency")
            plt.yscale("log")
            plt.xscale("log")
            plt.show()
        return alpha, rat

    def test_consistency_tlm_adjoint(self, n_repet: int = 1) -> List:
        """Test TLM/ADJ
        <Mdx, y> / <dx, M*y> = 1
        """
        print("Test TLM/ADJ")
        ips = []
        for i in range(n_repet):
            x = np.random.normal(size=self.n)
            dx = np.random.normal(size=self.n)
            y = np.random.normal(size=self.m)
            ip1 = self.tangent_linear(x).matvec(dx).T @ y
            ip2 = dx.T @ self.adjoint(x).matvec(y)
            print(f"<Mdx, y> / <dx, M*y> = {ip1 / ip2} should be 1.0")
            ips.append((ip1, ip2))
        return ips

    def test_consistency_forward_adjoint(
        self, plot: bool = True
    ) -> Tuple[np.ndarray, List]:
        print("Test Forward/Gradient Adjoint")
        x = np.random.normal(size=self.n)
        dx = np.random.normal(size=self.n)
        alpha = np.logspace(0, -14, 15, base=10)  ## modified
        diff = []
        for al in alpha:
            fd = np.empty(self.n)
            for i in range(self.n):
                dx = np.zeros(self.n)
                dx[i] = 1
                # print(self.cost_function(x + al * dx))
                fd[i] = (self.cost_function(x + al * dx) - self.cost_function(x)) / al
            diff.append(((fd - self.gradient(x)) ** 2).sum())
        if plot:
            import matplotlib.pyplot as plt

            plt.plot(alpha, diff)
            plt.title("Forward/Adjoint Gradient consistency")
            plt.yscale("log")
            plt.xscale("log")
            plt.show()
        return alpha, diff

    def tests_consistency(self) -> None:
        """Performs the consistency test between forward, TLM and adjoint based gradient"""
        alpha, rat = self.test_consistency_tlm_forward(plot=False)
        import matplotlib.pyplot as plt
        import scipy

        plt.subplot(1, 2, 1)
        plt.plot(alpha, rat)
        plt.title("Forward/TLM consistency\nV shape expected")
        plt.yscale("log")
        plt.xscale("log")
        plt.subplot(1, 2, 2)
        alpha, diff = self.test_consistency_forward_adjoint(plot=False)
        plt.plot(alpha, diff)
        plt.title("Forward/Adjoint Gradient consistency\nV shape expected")
        plt.yscale("log")
        plt.xscale("log")
        plt.show()
        n_repet = 5
        ips = self.test_consistency_tlm_adjoint(n_repet)
        ratios_ips = np.fromiter(map(lambda x: x[0] / x[1], ips), dtype=float)
        print(
            f"All ratios equal to 1: {np.allclose(ratios_ips, np.ones_like(ratios_ips))}\n"
        )
        print("Scipy Check Gradient:")
        for _ in range(5):
            print(
                f"{scipy.optimize.check_grad(self.cost_function, self.gradient, np.random.normal(size=self.n))} close to 0 ?"
            )

    def solve_innerloop(
        self,
        x: np.ndarray,
        iter_inner: int = 10,
        prec: Optional[Union[str, Callable]] = None,
        side: str = "left",
    ) -> Tuple[np.ndarray, dict]:

        GtG = self.gauss_newton_hessian_matrix(x, bck_prec=(prec == "bck"))
        try:
            GtG = GtG.matmat(np.eye(self.n))
        except AttributeError:
            pass
        # slogdet, cond = np.linalg.slogdet(GtG), np.linalg.cond(GtG) ## TODO: caution to rm
        # if cond > 1e3:   # TODO: SAME
        #     prec = None # TODO: SAME
        if prec is None:
            return solve_cg(GtG, -self.gradient(x), maxiter=iter_inner)
        elif prec == "jacobi":
            return solve_cg_jacobi(GtG, -self.gradient(x), maxiter=iter_inner)
        elif prec == "spectralLMP":
            return solve_cg_LMP(GtG, -self.gradient(x), r=self.r, maxiter=iter_inner)
        elif callable(prec):
            if side == "left":
                H = prec(x)
                return solve_cg(H @ GtG, - H @ self.gradient(x), maxiter=iter_inner)
            elif side == "right":
                H_R = prec(x)
                cg_solution = solve_cg(GtG @ H_R, -self.gradient(x), maxiter=iter_inner)
                return H_R @ cg_solution[0], cg_solution[1]
        elif prec == "bck":
            # B = UUT
            prec_mat = (
                np.eye(self.n)
                + self.background_error_sqrt.T @ GtG @ self.background_error_sqrt
            )
            cg_solution = solve_cg(
                prec_mat, -self.background_error_sqrt.T @ self.gradient(x)
            )
            return self.background_error_sqrt @ cg_solution[0], cg_solution[1]

    def GNmethod(
        self,
        x0: np.ndarray,
        n_outer: int = 3,
        n_inner: int = 10,
        verbose: bool = False,
        prec = None,
    ) -> Tuple:
        x_curr = x0
        colnames = ["niter", "f(x)", "CG iter", "log-det", "cond"]
        print(
            f"{colnames[0].rjust(5)}, {colnames[1].rjust(8)}, {colnames[2].rjust(8)}, {colnames[3].rjust(6)}, {colnames[4].rjust(6)}"
        )
        n_iter = np.empty(n_outer)
        fun = np.empty(n_outer + 1)
        fun[0] = self.cost_function(x_curr)
        if isinstance(prec, str):
            prec_name = prec
            side = "left"
        elif isinstance(prec, dict):
            prec_name = prec['prec_name']
            side = prec['side']
        elif prec is None:
            prec_name = None
            side = None
        cost_inner = []
        quad_error = []
        inner_res = []
        for i_outer in range(n_outer):
            dx, res = self.solve_innerloop(x_curr, n_inner, prec_name, side=side)
            inner_res.append(res)
            GtG = sla.aslinearoperator(self.gauss_newton_hessian_matrix(x_curr)).matmat(
                np.eye(self.n)
            )
            dx_star = np.linalg.solve(GtG, -self.gradient(x_curr))
            delta_x = np.asarray(res["x_list"]) - dx_star
            sq_err = 0.5 * np.diag(delta_x @ GtG @ delta_x.T) + self.cost_function(
                x_curr + dx_star
            )
            quad_error.append(sq_err)
            prev_n_inner_loop = res["niter"]
            cost_inner.append(res["cost_inner"])
            x_curr += dx
            cfun = self.cost_function(x_curr)
            if verbose:
                GtG = self.gauss_newton_hessian_matrix(x_curr, bck_prec=(prec_name == "bck"))
                if prec_name == "bck":
                    GtG = (
                        np.eye(self.n)
                        + self.background_error_sqrt.T
                        @ GtG
                        @ self.background_error_sqrt
                    )
                try:
                    slogdet, cond = np.linalg.slogdet(GtG), np.linalg.cond(GtG)
                except np.linalg.LinAlgError:
                    slogdet = 0, 0
                    cond = 0
                print(
                    f"{i_outer:>5}, {cfun:>8.4f}, {prev_n_inner_loop:>8d}, {slogdet[1]:>6.2f}, {cond:>6.4e}"
                )
            n_iter[i_outer] = prev_n_inner_loop
            fun[i_outer + 1] = cfun
        return x_curr, self.cost_function(x_curr), n_iter, fun, cost_inner, quad_error, inner_res


def generate_marginal_obs_operator(
    H: np.ndarray, marginal,_func: Callable, linearized_marginal_func: Callable
) -> ObservationOperator:
    # H = np.random.binomial(1, p, np.prod(shape)).reshape(shape)
    obs_op = ObservationOperator(H.shape[0], H.shape[1])
    obs_op.set_operator(lambda x: H @ marginal_func(x))
    obs_op.set_linearized(lambda x: H @ linearized_marginal_func(x))
    return obs_op


def f(x):
    return 1 / (x ** 2 + 1)


def fprime(x):
    return np.diag(-(2 * x) / (x ** 2 + 1) ** 2)


if __name__ == "__main__":
    shape = (2, 3)
    p = 0.5
    H = np.random.binomial(1, p, np.prod(shape)).reshape(shape)
    print(H)
    operator = generate_marginal_obs_operator(H, f, fprime)
    operator.linearized_operator(1 + np.arange(3)).matmat(np.eye(3))

    n = 2
    m = 2
    A = np.random.uniform(size=n * m).reshape(m, n)

    def G(x):
        return A @ x

    y = np.random.normal(size=m)

    def data_misfit(x):
        return G(x) - y

    def cost_function(x):
        return 0.5 * np.linalg.norm(data_misfit(x)) ** 2

    def TLM(x):
        return A

    def adjoint(x):
        return A.T

    def GtG(x):
        return adjoint(x) @ TLM(x)

    def grad(x):
        return adjoint(x) @ data_misfit(x)

    linsys = NumericalModel(n=n, m=m)
    linsys.set_forward(G)
    linsys.set_tangent_linear(TLM)
    linsys.test_consistency_tlm_forward()
    linsys.set_adjoint(adjoint)
    linsys.test_consistency_tlm_adjoint()

    x = np.random.uniform(size=n)
    obs = G(x) + np.random.normal(size=m)
    linsys.set_obs(obs)
    linsys.test_consistency_forward_adjoint()

    x0 = np.zeros(n)
    import scipy.optimize

    print(
        f"x scipy error: {((scipy.optimize.minimize(linsys.cost_function, x0=x0).x - x)**2).sum()}"
    )
    x_star, *_ = linsys.GNmethod(x0)
    print(f"x GN    error: {((x_star - x)**2).sum()}")

    nnlinsys = NumericalModel(n=3, m=3)

    def forw(x):
        return np.array([x[0] * x[1], x[1] * x[2], x[2] * x[0]])

    def tlm(inp):
        x, y, z = inp[0], inp[1], inp[2]
        return np.array([[y, x, 0], [0, z, y], [z, 0, x]])

    nnlinsys.set_forward(forw)
    nnlinsys.set_tangent_linear(tlm)
    x = np.random.uniform(size=nnlinsys.n)
    obs = forw(x)  # + np.random.normal(size=nnlinsys.m)
    nnlinsys.set_obs(obs)
    nnlinsys.test_forward_tlm_adjoint()
