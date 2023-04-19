import numpy.linalg as la


def eff_cond(A, tol=1e-12):
    s = la.svd(A, compute_uv=False)
    s_no_zeros = s[s > tol]
    return s_no_zeros[0] / s_no_zeros[-1]


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
