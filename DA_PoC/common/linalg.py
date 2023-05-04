import numpy.linalg as la
import numpy as np


def eff_cond(A, tol=1e-12):
    s = la.svd(A, compute_uv=False)
    s_no_zeros = s[s > tol]
    return s_no_zeros[0] / s_no_zeros[-1]


def randomised_eigval_decomposition(A, r, l):
    n = A.shape[0]
    G = np.random.normal(size=(n, (r + l)))
    Y = A @ G
    Z_1 = la.qr(Y)[0]
    K_1 = Z_1.T @ A @ Z_1
    w, v = la.eigh(K_1)
    Theta_1 = w[::-1]
    W_1 = v[:, ::-1]
    return Theta_1[:-l], Z_1 @ W_1[:, :-l]


def randomised_eigval_ritzit(A, r, l):
    n = A.shape[0]
    G = np.random.normal(size=(n, (r + l)))
    G_3 = la.qr(G)[0]
    Y_3 = A @ G_3
    Z_3, R_3 = la.qr(Y_3)
    K_3 = R_3 @ R_3.T
    w, v = la.eigh(K_3)
    W_3 = v[:, ::-1]
    Theta_32 = w[::-1]
    return np.sqrt(Theta_32)[::-l], Z_3 @ W_3[:, :-l]


if __name__ == "__main__":
    import numpy as np

    n = 20
    r = 15
    L = np.random.normal(size=(n, r))
    A = L @ L.T
    # print(la.svd(A, compute_uv=False))
    print(f"A of dim {n}x{n}")
    print(f"Rank of A: {la.matrix_rank(A)}: singular matrix")
    print(f"Condition number: {la.cond(A)}")
    print(f"Effective condition number: {eff_cond(A)}")

    B = A + np.eye(n)
    print("B = A + I invertible")
    # print(la.svd(B, compute_uv=False))
    print(f"Condition number: {la.cond(B)}")
    print(f"Effective condition number: {eff_cond(B)}")
