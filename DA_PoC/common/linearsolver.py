import numpy as np

# import torch
class OrthogonalProjector:
    def __init__(self, A: np.ndarray, W: np.ndarray):
        self.W = W
        self.A = A
        self.normalizing_matrix = np.linalg.inv(self.W.T @ self.A @ self.W)
        self.projection_matrix = self.W @ self.normalizing_matrix @ self.W.T


def conjGrad(
    A: np.ndarray, x: np.ndarray, b: np.ndarray, tol: float, maxiter: int, verbose=False
):
    """Solves Ax = b with Conjugate Gradient method

    :param A: Matrix to inverse
    :type A: np.ndarray
    :param x: First guess of the solution
    :type x: np.ndarray
    :param b: Expected solution of Ax
    :type b: np.ndarray
    :param tol: tolerance
    :type tol: float
    :param maxiter: maximum number of iterations
    :type maxiter: int
    :param verbose: _description_, defaults to False
    :type verbose: bool, optional
    :return: _description_
    :rtype: _type_
    """
    r = b - A.dot(x)
    p = r.copy()
    residuals = [r]
    x_list = [x]
    norm_res = []
    it = 0
    cost_inner = [np.sum(r * x)]
    try:
        cond = np.linalg.cond(A)
    except np.linalg.LinAlgError:
        cond = np.inf
    if verbose:
        print(f"Condition number of A: {cond}")
    while (np.sqrt(np.sum((r**2))) >= tol) and (it < maxiter):
        Ap = A.dot(p)
        alpha = np.dot(p, r) / np.dot(p, Ap)
        x = x + alpha * p
        r = b - A.dot(x)
        beta = -np.dot(r, Ap) / np.dot(p, Ap)
        p = r + beta * p
        it += 1
        norm_res_ = np.sqrt(np.sum((r**2)))
        norm_res.append(norm_res_)
        cost_inner.append(np.sum(r * x))
        x_list.append(x)
        residuals.append(r)
        if verbose and it % 20 == 0:
            print(f"It: {it:>5}, ||r|| = {norm_res_}")
    if verbose:
        print(f"It: {it:>5}, ||r|| = {norm_res_}")
    result_dict = {
        "success": it != maxiter,
        "niter": it,
        "residuals": residuals,
        "norm_res": norm_res,
        "cost_inner": cost_inner,
        "x_list": x_list,
        "cond": cond,
    }
    return x, result_dict


def deflated_conjGrad(
    A: np.ndarray,
    x: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    tol: float,
    maxiter: int,
    verbose=False,
):
    """Solves Ax = b with Deflated Conjugate Gradient method

    :param A: Matrix to inverse
    :type A: np.ndarray
    :param x: First guess of the solution
    :type x: np.ndarray
    :param b: Expected solution of Ax
    :type b: np.ndarray
    :param W: full column rank matrix
    :type W: np.ndarray
    :param tol: tolerance
    :type tol: float
    :param maxiter: maximum number of iterations
    :type maxiter: int
    :param verbose: _description_, defaults to False
    :type verbose: bool, optional
    :return: _description_
    :rtype: _type_
    """
    projection = W.T @ A @ W
    inv_projection = np.linalg.inv(projection)
    r = b - A.dot(x)
    x = x + W @ inv_projection @ W.T @ r
    p = r.copy()
    residuals = [r]
    x_list = [x]
    norm_res = []
    it = 0
    cost_inner = [np.sum(r * x)]
    try:
        cond = np.linalg.cond(A)
    except np.linalg.LinAlgError:
        cond = np.inf
    if verbose:
        print(f"Condition number of A: {cond}")
    while (np.sqrt(np.sum((r**2))) >= tol) and (it < maxiter):
        Ap = A.dot(p)
        alpha = np.dot(p, r) / np.dot(p, Ap)
        x = x + alpha * p
        r = b - A.dot(x)
        beta = -np.dot(r, Ap) / np.dot(p, Ap)
        p = r + beta * p
        it += 1
        norm_res_ = np.sqrt(np.sum((r**2)))
        norm_res.append(norm_res_)
        cost_inner.append(np.sum(r * x))
        x_list.append(x)
        residuals.append(r)
        if verbose and it % 20 == 0:
            print(f"It: {it:>5}, ||r|| = {norm_res_}")
    if verbose:
        print(f"It: {it:>5}, ||r|| = {norm_res_}")
    result_dict = {
        "success": it != maxiter,
        "niter": it,
        "residuals": residuals,
        "norm_res": norm_res,
        "cost_inner": cost_inner,
        "x_list": x_list,
        "cond": cond,
    }
    return x, result_dict


def prec_conjGrad(
    A: np.ndarray,
    x: np.ndarray,
    b: np.ndarray,
    H: np.ndarray,
    tol: float,
    maxiter: int,
    verbose=False,
):
    """Solves Ax = b with preconditioned CG

    :param A: Matrix to inverse
    :type A: np.ndarray
    :param x: First guess of the solution
    :type x: np.ndarray
    :param b: Expected solution of Ax
    :type b: np.ndarray
    :param H: preconditioner
    :type H: np.ndarray
    :param tol: tolerance
    :type tol: float
    :param maxiter: maximum number of iterations
    :type maxiter: int
    :param verbose: _description_, defaults to False
    :type verbose: bool, optional
    :return: _description_
    :rtype: _type_
    """

    r = b - A.dot(x)
    z = H @ r
    p = z.copy()
    residuals = [r]
    x_list = [x]
    norm_res = []
    it = 0
    cost_inner = [np.sum(r * x)]
    try:
        cond = np.linalg.cond(A)
    except np.linalg.LinAlgError:
        cond = np.inf
    if verbose:
        print(f"Condition number of A: {cond}")
    while (np.sqrt(np.sum((r**2))) >= tol) and (it < maxiter):
        Ap = A.dot(p)
        rz = np.dot(p, z)
        alpha = rz / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        z = H @ r
        beta = np.dot(r, z) / rz
        p = z + beta * p
        it += 1
        norm_res_ = np.sqrt(np.sum((r**2)))
        norm_res.append(norm_res_)
        cost_inner.append(np.sum(r * x))
        x_list.append(x)
        residuals.append(r)
        if verbose and it % 20 == 0:
            print(f"It: {it:>5}, ||r|| = {norm_res_}")
    if verbose:
        print(f"It: {it:>5}, ||r|| = {norm_res_}")
    result_dict = {
        "success": it != maxiter,
        "niter": it,
        "residuals": residuals,
        "norm_res": norm_res,
        "cost_inner": cost_inner,
        "x_list": x_list,
        "cond": cond,
    }
    return x, result_dict


def jacobi_preconditioner(A):
    prec = np.zeros_like(A)
    for i in range(A.shape[0]):
        prec[i, i] = A[i, i] ** (-1)
    return prec


def solve_cg(A, b, maxiter=None, verbose=False):
    if maxiter is None:
        maxiter = len(b)
    x, res = conjGrad(
        A,
        np.zeros_like(b),
        b,
        1e-8,
        maxiter=maxiter,
        verbose=verbose,
    )
    return x, res


def solve_prec_cg(A, H, b, maxiter=None, verbose=False):
    if maxiter is None:
        maxiter = len(b)
    x, res = prec_conjGrad(
        A,
        np.zeros_like(b),
        b,
        H,
        1e-8,
        maxiter=maxiter,
        verbose=verbose,
    )
    return x, res


def solve_cg_jacobi(A, b, maxiter=None, verbose=False):
    if maxiter is None:
        maxiter = len(b)
    prec = jacobi_preconditioner(A)
    x, res = conjGrad(
        prec @ A, np.zeros_like(b), prec @ b, 1e-8, maxiter=maxiter, verbose=verbose
    )
    return x, res


def solve_cg_split_preconditioned(A, b, L, maxiter=None, verbose=False):
    """Ax = b <=> L^T Ax = L^Tb
            <=> L^T A (L L^-1) x = L^Tb
            <=> L^T A L (L^-1 x) = L^T b

    :param A: _description_
    :type A: _type_
    :param b: _description_
    :type b: _type_
    :param R: _description_
    :type R: _type_
    :param maxiter: _description_, defaults to None
    :type maxiter: _type_, optional
    """
    x, res = solve_cg(L.T @ A @ L, L.T @ b, maxiter, verbose)
    return L @ x


def incomplete_cholesky_factorization(A):
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    for k in range(n):
        A[k, k] = np.sqrt(A[k, k])
        # print(f"{k=}, {A[k, k]=}")
        for i in range(k + 1, n):
            # print(f"{k=}, {i=}, {A[i, k]=}")
            if A[i, k] != 0:
                A[i, k] = A[i, k] / A[k, k]
        for j in range(k + 1, n):
            for i in range(j, n):
                if A[i, j] != 0:
                    A[i, j] = A[i, j] - A[i, k] * A[j, k]
    return np.tril(A)


def construct_LMP(n, S, AS, shift=1):
    In = np.eye(n)
    StASm1 = np.linalg.inv(np.matmul(S.T, AS))
    left = In - np.matmul(np.matmul(S, StASm1), AS.T)
    mid = In - np.matmul(np.matmul(AS, StASm1), S.T)
    right = np.matmul(np.matmul(S, StASm1), S.T)
    H = np.matmul(left, mid) + shift * right
    return H


def solve_cg_LMP(A, b, r, maxiter=None, verbose=False):
    _, v = np.linalg.eig(A)
    S = v[:, :r]
    AS = A @ S
    prec = construct_LMP(len(b), S, AS)
    return solve_cg(prec @ A, prec @ b, maxiter=maxiter, verbose=verbose)


# def conjGradTensor(A, x, b, tol, maxiter, verbose=False):
#     r = b - torch.matmul(A, x)
#     p = r.copy()
#     it = 0
#     if verbose:
#         cond = torch.linalg.cond(A)
#         print(f"Condition number of A: {cond}")
#     while (torch.sqrt(torch.sum((r ** 2))) >= tol) and (it < maxiter):
#         Ap = torch.matmul(A, p)
#         alpha = torch.matmul(p, r) / torch.matmul(p, Ap)
#         x = x + alpha * p
#         r = b - torch.matmul(A, x)
#         beta = -torch.matmul(r, Ap) / torch.matmul(p, Ap)
#         p = r + beta * p
#         it += 1
#         if verbose and it % 10 == 0:
#             print(f"It: {it:>5}, ||r|| = {torch.sqrt(torch.sum((r**2)))}")
#     if verbose:
#         print(f"It: {it:>5}, ||r|| = {torch.sqrt(torch.sum((r**2)))}")
#     result_dict = {
#         "success": it != maxiter,
#         "niter": it,
#         "residual": r,
#     }
#     return x, result_dict


if __name__ == "__main__":
    n = 500
    A_ = np.random.normal(size=(n**2)).reshape(n, n)
    A = A_.T @ A_
    # A = np.array(
    #     [
    #         [3, 0, -1, -1, 0, -1],
    #         [0, 2, 0, -1, 0, 0],
    #         [-1, 0, 3, 0, -1, 0],
    #         [-1, -1, 0, 2, 0, -1],
    #         [0, 0, -1, 0, 3, 1],
    #         [-1, 0, 0, -1, -1, 4],
    #     ],
    #     dtype=float,
    # )

    x = np.random.normal(size=n)
    x0 = np.random.normal(size=n)
    b = A @ x
    x_cg, res = conjGrad(A, x0, b, 1e-9, n, True)
    # x_cg_jacobi, res = solve_cg_jacobi(A, b, maxiter=n, verbose=True)
    # x_cg_lmp, res = solve_cg_LMP(A, b, r=n // 2, maxiter=n, verbose=True)

    # L = icholesky(A)
    # L = incomplete_cholesky_factorization(A)
    # x_split = solve_cg_split_preconditioned(A, b, np.linalg.inv(L).T, verbose=True)
