#!/usr/bin/env python
# coding: utf-8


import numpy as np
import tqdm
import matplotlib.pyplot as plt

# import warnings

# warnings.filterwarnings('error')


class NumSolver:
    def __init__(self, step, f) -> None:
        self.step = step
        self.fun = f

    def integrate_step(self, t0, x0):
        pass


def RK4_step(f, t, x, dt, *args):
    """Apply a step from the Runge-Kutta of order 4 method with fixed stepsize
    x' = f(t,x)

    :param f: function which defines the ODE
    :param t: time parameter
    :param x: state vector
    :param dt: stepsize
    :returns: update state vector

    """
    k1 = f(t, x, *args)
    k2 = f(t + dt / 2, x + k1 * dt / 2.0, *args)
    k3 = f(t + dt / 2, x + k2 * dt / 2.0, *args)
    k4 = f(t + dt, x + k3 * dt, *args)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def RK4_step_forward_tlm(f, f_t, t, x, x_t, dt, *args):
    k1 = f(t, x, *args)
    k2 = f(t + dt / 2, x + k1 * dt / 2.0, *args)
    k3 = f(t + dt / 2, x + k2 * dt / 2.0, *args)
    k4 = f(t + dt, x + k3 * dt, *args)
    new_x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    x_t = np.atleast_1d(x_t)
    k1_t = np.dot(f_t(t, x, *args), x_t)
    k2_t = np.dot(f_t(t + dt / 2.0, x + k1 * dt / 2.0, *args), (x_t + k1_t * dt / 2.0))
    k3_t = np.dot(f_t(t + dt / 2.0, x + k2 * dt / 2.0, *args), (x_t + k2_t * dt / 2.0))
    k4_t = np.dot(f_t(t + dt / 2.0, x + k3 * dt, *args), (x_t + k3_t * dt))
    x_t = x_t + (dt / 6.0) * (k1_t + 2 * k2_t + 2 * k3_t + k4_t)
    return new_x, x_t, k1_t, k2_t, k3_t, k4_t


def intermediate_step_t(f, f_t, x, x_t, ki, ki_t, mul):
    kj_t = np.dot(f_t(x + mul * ki), (x_t + mul * ki_t))
    return kj_t


def intermediate_step_a(x_a, ki_a, kj_a, mul, tangent_evaluated):
    x_a = x_a + tangent_evaluated
    ki_a = ki_a + mul * tangent_evaluated
    kj_a = 0
    return x_a, ki_a, kj_a


def test_intermediate_scalarproduct():
    def f_t(x):
        return 2 * x

    x, x_t, ki, ki_t, mul = 1, 2, 3, 4, 5
    kj_t = intermediate_step_t(None, f_t, x, x_t, ki, ki_t, mul)
    tangent_evaluated = f_t(x + mul * ki)

    x_a, ki_a, kj_a = 4.44, 6.66, 5.55
    print(np.dot(np.array([x_t, ki_t, kj_t]), np.array([x_a, ki_a, kj_a])))

    x_a, ki_a, kj_a = intermediate_step_a(x_a, ki_a, kj_a, mul, tangent_evaluated)

    print(np.dot(np.array([x_t, ki_t, 999]), np.array([x_a, ki_a, kj_a])))


def RK4_step_adj(var, var_a, f_t, dt, *args):
    x_a, k1_a, k2_a, k3_a, k4_a = var_a
    x, k1, k2, k3, k4, t = var
    x_a = x_a
    k1_a = dt / 6.0 * x_a + k1_a
    k2_a = dt / 3.0 * x_a + k2_a
    k3_a = dt / 3.0 * x_a + k3_a
    k4_a = dt / 6.0 * x_a + k3_a
    x_a, k4_a, k3_a = intermediate_step_a(
        x_a, k4_a, k3_a, dt / 6.0, f_t(t + dt / 2.0, x + k3 * dt, *args)
    )
    x_a, k3_a, k2_a = intermediate_step_a(
        x_a, k4_a, k3_a, dt / 3.0, f_t(t + dt / 2.0, x + 2 * k2 * dt, *args)
    )
    x_a, k2_a, k1_a = intermediate_step_a(
        x_a, k4_a, k3_a, dt / 3.0, f_t(t + dt / 2.0, x + 2 * k1 * dt, *args)
    )
    x_a = x_a + f_t(t, x, *args)
    k1_a = 0
    return x_a, k1_a, k2_a, k3_a, k4_a


def DoPri45_step(f, t, x, h):
    """Apply a step using Dormand-Prince method

    :param f: function which defines the ODE
    :param t: time
    :param x: state vector
    :param h: stepsize
    :returns: update state vector

    """
    k1 = f(t, x)
    k2 = f(t + 1.0 / 5 * h, x + h * (1.0 / 5 * k1))
    k3 = f(t + 3.0 / 10 * h, x + h * (3.0 / 40 * k1 + 9.0 / 40 * k2))
    k4 = f(
        t + 4.0 / 5 * h,
        x + h * (44.0 / 45 * k1 - 56.0 / 15 * k2 + (32.0 / 9) * k3),
    )
    k5 = f(
        t + 8.0 / 9 * h,
        x
        + h
        * (
            19372.0 / 6561 * k1
            - 25360.0 / 2187 * k2
            + 64448.0 / 6561 * k3
            - 212.0 / 729 * k4
        ),
    )
    k6 = f(
        t + h,
        x
        + h
        * (
            9017.0 / 3168 * k1
            - 355.0 / 33 * k2
            + 46732.0 / 5247 * k3
            + 49.0 / 176 * k4
            - 5103.0 / 18656 * k5
        ),
    )

    v5 = (
        35.0 / 384 * k1
        + 500.0 / 1113 * k3
        + 125.0 / 192 * k4
        - 2187.0 / 6784 * k5
        + 11.0 / 84 * k6
    )
    k7 = f(t + h, x + h * v5)
    v4 = (
        5179.0 / 57600 * k1
        + 7571.0 / 16695 * k3
        + 393.0 / 640 * k4
        - 92097.0 / 339200 * k5
        + 187.0 / 2100 * k6
        + 1.0 / 40 * k7
    )

    return x + h * v5


def integrate_step(step, f, t0, x0, dt, Nsteps, verbose=False, *args):
    t = np.empty(Nsteps + 1)
    t[0] = t0
    t_ = t0
    x = np.empty((len(x0), Nsteps + 1))
    x[:, 0] = x0
    curr_x = x0
    for i in tqdm.trange(Nsteps, disable=(not verbose)):
        curr_x = step(f, t, curr_x, dt, *args)
        t_ += dt
        t[i + 1] = t_
        x[:, i + 1] = curr_x
    return t, x


def integrate_step_tlm(step_tlm, f, t0, x0, dt, f_t, dx0, Nsteps, verbose=False, *args):
    t = np.empty(Nsteps + 1)
    t[0] = t0
    t_ = t0
    x = np.empty((len(x0), Nsteps + 1))
    x_t = np.empty_like(x)
    x[:, 0] = x0
    x_t[:, 0] = dx0
    curr_x = x0
    curr_x_t = dx0
    for i in tqdm.trange(Nsteps, disable=(not verbose)):
        curr_x, curr_x_t, _, _, _, _ = step_tlm(f, f_t, t, curr_x, curr_x_t, dt, *args)
        t_ += dt
        t[i + 1] = t_
        x[:, i + 1] = curr_x
        x_t[:, i + 1] = curr_x_t
    return t, x, x_t


def integrate_RK4(f, t0, x0, dt, Nsteps, verbose=True, *args):
    """Integrate the given ODE for Nsteps

    :param f:
    :param t0:
    :param x0:
    :param dt:
    :param Nsteps:
    :returns:

    """
    return integrate_step(RK4_step, f, t0, x0, dt, Nsteps, verbose=verbose, *args)


def integration_RK4(f, x0, t, dt):
    x = [x0]
    for t_ in t[1:]:
        x.append(RK4_step(f, t_, x[-1], dt))
    return x


def integration_RK4_tlm(f, f_t, t, x0, dx0, dt):
    x = [x0]
    dx = [dx0]
    for t_ in t[1:]:
        new_x, new_dx, _, _, _, _ = RK4_step_forward_tlm(f, f_t, t_, x[-1], dx[-1], dt)
        # print(new_x, new_dx)
        x.append(new_x)
        dx.append(new_dx)
    return x, dx


def test_tlm_1d():
    def f(t, x, lam=1):
        return np.array(-lam * x ** 2)

    def f_t(t, x, lam=1):
        return np.array(-2 * lam * x)

    x0 = 1
    dt = 0.01
    t = np.arange(0, 1, dt)

    dx0 = np.array(1)
    x, dx = integration_RK4_tlm(f, f_t, t, x0, dx0, dt)

    ratio = []
    print("test 1d -------")

    for alpha in [10 ** (-i) for i in range(1, 12)]:
        # plt.plot(t, x)
        # plt.plot(t, np.exp(-t))
        # plt.show()
        finite_difference = np.array(
            integration_RK4(f, x0 + alpha * dx0, t, dt)
        ) - np.array(integration_RK4(f, x0, t, dt))
        rat = np.sum(finite_difference ** 2) / np.sum((alpha * np.asarray(dx)) ** 2)
        ratio.append(rat)
        print(f"{alpha:>6}, {np.abs(rat.item() - 1)}")

    plt.plot([10 ** (-i) for i in range(1, 12)], np.abs(np.array(ratio) - 1))
    plt.title(
        r"Ratio test 1D: $\frac{\|F(x + \alpha \delta x) - F(x)\|^2}{\|\alpha \mathrm{TLM}(\delta x)\|^2}$"
    )
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.show()


def test_tlm_2d():
    x0 = np.array([1, 0])

    def f2D(t, x):
        return np.array([-x[1] * x[0], x[0]])

    def f2D_t(t, x):
        return np.array([[-x[1], -x[0]], [1, 0]])

    dt = 0.01
    dx0 = np.array([1, 1])
    t = np.arange(0, 2 * np.pi, dt)
    x = integration_RK4(f2D, x0, t, dt)
    x, dx = integration_RK4_tlm(f2D, f2D_t, t, x0, dx0, dt)
    ratio = []
    alpha_list = [10 ** (-i) for i in range(1, 12)]
    print("test 2d -------")

    for alpha in alpha_list:
        finite_difference = np.array(
            integration_RK4(f2D, x0 + alpha * dx0.flatten(), t, dt)
        ) - np.array(integration_RK4(f2D, x0, t, dt))
        rat = np.sum(finite_difference ** 2) / np.sum((alpha * np.asarray(dx)) ** 2)
        ratio.append(rat)
        print(f"{alpha:>6}, {np.abs(rat - 1)}")
    plt.plot(alpha_list, np.abs(np.array(ratio) - 1))
    plt.title(
        r"Ratio test 2D: $\frac{\|F(x + \alpha \delta x) - F(x)\|^2}{\|\alpha \mathrm{TLM}(\delta x)\|^2}$"
    )
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.show()


def test_tlm_nd(n=10):
    x0 = np.random.uniform(size=n)

    def f2D(t, x):
        return -np.roll(x, -1) * x

    def f2D_t(t, x):
        jac = -np.diag(np.roll(x, -1)) - np.diag(x[:-1], 1)
        jac[-1, 0] = -x[-1]
        return jac

    dt = 0.01
    dx0 = np.ones(n)
    t = np.arange(0, 2 * np.pi, dt)
    x = integration_RK4(f2D, x0, t, dt)
    x, dx = integration_RK4_tlm(f2D, f2D_t, t, x0, dx0, dt)
    ratio = []
    alpha_list = [10 ** (-i) for i in range(1, 12)]
    print("test n dimensional -------")
    for alpha in alpha_list:
        finite_difference = np.array(
            integration_RK4(f2D, x0 + alpha * dx0.flatten(), t, dt)
        ) - np.array(integration_RK4(f2D, x0, t, dt))
        rat = np.sum(finite_difference ** 2) / np.sum((alpha * np.asarray(dx)) ** 2)
        ratio.append(rat)
        print(f"{alpha:>6}, {np.abs(rat - 1)}")
    plt.plot(alpha_list, np.abs(np.array(ratio) - 1))
    plt.title(
        r"Ratio test nD: $\frac{\|F(x + \alpha \delta x) - F(x)\|^2}{\|\alpha \mathrm{TLM}(\delta x)\|^2}$"
    )
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.show()


def construct_tlm_matrix():
    x0 = np.array([1, 0])

    def f2D(t, x):
        return np.array([-x[1] * x[0], x[0]])

    def f2D_t(t, x):
        return np.array([[-x[1], -x[0]], [1, 0]])

    dt = 0.01
    t = np.arange(0, 2 * np.pi, dt)
    jac = []
    fd = []
    for i in range(len(x0)):
        ei = np.zeros_like(x0)
        ei[i] = 1
        _, fei = integration_RK4_tlm(f2D, f2D_t, t, x0, ei, dt)
        jac.append(fei[-1])

        # finite_difference = np.array(
        #     integration_RK4(f2D, x0 + 1e-7 * ei, t, dt)
        # ) - np.array(integration_RK4(f2D, x0, t, dt))
        # fd.append(finite_difference[-1])
    # print(fd)
    return np.asarray(jac)


if __name__ == "__main__":
    test_tlm_1d()
    test_tlm_2d()
    test_tlm_nd(n=5)
    print(construct_tlm_matrix())
    # test_intermediate_scalarproduct()
