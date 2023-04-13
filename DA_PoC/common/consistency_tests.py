import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import scipy.sparse.linalg as sla


def test_forw_tlm_alpha(
    forward: Callable[[np.ndarray], np.ndarray],
    tangent_linear: Callable[[np.ndarray], sla.LinearOperator],
    n: int,
    alpha: float,
) -> float:
    """Test consistency between TLM and forward model
    computes |G(x + alpha dx) - G(x)| / |alpha * TLM(x, dx)|
    """
    x = np.random.normal(size=n)
    dx = np.random.normal(size=n)
    fd = (forward(x + alpha * dx) - forward(x)) / alpha
    tlm = tangent_linear(x).matvec(dx)
    return np.linalg.norm(fd) / (np.linalg.norm(tlm))


def test_consistency_tlm_forward(
    forward: Callable[[np.ndarray], np.ndarray],
    tangent_linear: Callable[[np.ndarray], sla.LinearOperator],
    n: int,
    plot=True,
) -> Tuple[np.ndarray, List]:
    """
    "Test Forw/TLM
    |Forward(x + a*dx) - Forward(x)| / | (a*TLM(x, dx)) | - 1 ->  0 as a -> 0
    """
    print("Test Forw/TLM")
    rat = []
    alpha = np.logspace(0, -14, 15, base=10)
    # print(f"     a, ratio")
    for al in alpha:
        ratio = np.abs(test_forw_tlm_alpha(forward, tangent_linear, n, al) - 1)
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


def test_consistency_tlm_adjoint(
    tangent_linear: Callable[[np.ndarray], sla.LinearOperator],
    adjoint: Callable[[np.ndarray], sla.LinearOperator],
    n: int,
    m: int,
    n_repet: int = 1,
) -> List:
    """Test TLM/ADJ
    <Mdx, y> / <dx, M*y> = 1
    """
    print("Test TLM/ADJ")
    ips = []
    for i in range(n_repet):
        x = np.random.normal(size=n)
        dx = np.random.normal(size=n)
        y = np.random.normal(size=m)
        ip1 = tangent_linear(x).matvec(dx).T @ y
        ip2 = dx.T @ adjoint(x).matvec(y)
        print(f"<Mdx, y> / <dx, M*y> = {ip1 / ip2} should be 1.0")
        ips.append((ip1, ip2))
    return ips


def test_consistency_forward_adjoint(
    cost_function: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    n: int,
    plot: bool = True,
) -> Tuple[np.ndarray, List]:
    print("Test Forward/Gradient Adjoint")
    x = np.random.normal(size=n)
    dx = np.random.normal(size=n)
    alpha = np.logspace(0, -14, 15, base=10)  ## modified
    diff = []
    for al in alpha:
        fd = np.empty(n)
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = 1
            # print(cost_function(x + al * dx))
            fd[i] = (cost_function(x + al * dx) - cost_function(x)) / al
        diff.append(((fd - gradient(x)) ** 2).sum())
    if plot:
        import matplotlib.pyplot as plt

        plt.plot(alpha, diff)
        plt.title("Forward/Adjoint Gradient consistency")
        plt.yscale("log")
        plt.xscale("log")
        plt.show()
    return alpha, diff
