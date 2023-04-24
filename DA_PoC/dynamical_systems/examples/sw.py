import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import tqdm
from functools import partial


@jax.jit
def enforce_boundary_periodic_x(arr):
    arr = arr.at[0, :].set(arr[-2, :])
    arr = arr.at[-1, :].set(arr[1, :])
    return arr


@jax.jit
def enforce_boundary_u(arr):
    return arr.at[-2, :].set(0.0)


class SWModel:
    def __init__(self, n_x, dx, n_y, dy, periodic_x=True) -> None:
        self.n_x = n_x
        self.dx = dx
        self.n_y = n_y
        self.dy = dy
        # grid setup
        # n_x = 200
        # dx = 5e3
        self.state_variable_length = self.n_x * self.n_y
        self.slice_u = slice(0, self.state_variable_length)
        self.slice_v = slice(self.state_variable_length, 2 * self.state_variable_length)
        self.slice_h = slice(2 * self.state_variable_length, None)

        self.l_x = self.n_x * self.dx

        self.l_y = self.n_y * self.dy

        self.grid_shape = (self.n_x, self.n_y)

        # other parameters
        self.periodic_boundary_x = periodic_x
        self.linear_momentum_equation = False

        self.gravity = 9.81  # m/s-2
        self.depth = 100.0  # in meters
        self.coriolis_f = 2e-4  # rad/s
        self.coriolis_beta = 2e-11  #
        self.lateral_viscosity = 1e-3 * self.coriolis_f * self.dx**2

        self.adams_bashforth_a = 1.5 + 0.1
        self.adams_bashforth_b = -(0.5 + 0.1)

    def compute_h_geostrophy(self, u0):
        return np.cumsum(-self.dy * u0 * self.coriolis_param / self.gravity, axis=0)

    def separate_state_vector(self, state):
        u = state[self.slice_u]
        v = state[self.slice_v]
        h = state[self.slice_h]
        return (
            u.reshape(self.grid_shape),
            v.reshape(self.grid_shape),
            h.reshape(self.grid_shape),
        )

    def concatenate_state_var(self, u, v, h):
        return np.concatenate([u.flatten(), v.flatten(), h.flatten()])

    def plot_state_imshow(self, state):
        u, v, h = self.separate_state_vector(state)
        plt.subplot(1, 3, 1)
        plt.imshow(u)
        plt.title("u")
        plt.subplot(1, 3, 2)
        plt.imshow(v)
        plt.title("v")
        plt.subplot(1, 3, 3)
        plt.imshow(h)
        plt.title("h")

    def plot_state_imshow_axs(self, state, axs):
        u, v, h = self.separate_state_vector(state)
        axs[0].imshow(u)
        axs[0].set_title("u")
        axs[1].imshow(v)
        axs[1].set_title("v")
        axs[2].imshow(h)
        axs[2].set_title("h")
        return axs

    def enforce_boundaries(self, arr, grid):
        assert grid in ("h", "u", "v")
        if self.periodic_boundary_x:
            arr[0, :] = arr[-2, :]
            arr[-1, :] = arr[1, :]
        elif grid == "v":
            arr[:, -2] = 0.0
        if grid == "u":
            arr[-2, :] = 0.0
        return arr

    def iterate_steps(self):
        while True:
            u, v, h = self.step()
            yield u, v, h

    def init_ncfile(self, path):
        u, v, h = self.u, self.v, self.h
        print(f"{u.shape=}")
        print(f"{v.shape=}")
        print(f"{h.shape=}")
        with netCDF4.Dataset(path, "w", format="NETCDF4") as ncfile:
            ncfile.createDimension("n_x", self.n_x)
            ncfile.createDimension("n_y", self.n_y)

            ncfile.createDimension("time", None)
            #
            ncfile.createVariable("x", "f4", ("n_x",))
            ncfile.createVariable("y", "f4", ("n_y",))
            ncfile.variables["x"] = self.x
            ncfile.variables["y"] = self.y

            ncfile.createVariable("t", "f4", ("time",))
            ncfile.coordinates = "t x y"
            ncfile.createVariable(
                "u",
                "f4",
                ("time", "n_x", "n_y"),
                compression="zlib",
                least_significant_digit=4,
            )
            ncfile.variables["u"][0, :, :] = u
            ncfile.createVariable(
                "v",
                "f4",
                ("time", "n_x", "n_y"),
                compression="zlib",
                least_significant_digit=4,
            )
            ncfile.variables["v"][0, :, :] = v

            ncfile.createVariable(
                "h",
                "f4",
                ("time", "n_x", "n_y"),
                compression="zlib",
                least_significant_digit=4,
            )
            ncfile.variables["h"][0, :, :] = h
            ncfile.variables["t"][0] = 0

    def write_ncfile(self, path, time_index, time):
        u, v, h = self.u, self.v, self.h
        """Write the temperature data to a netCDF file."""
        with netCDF4.Dataset(path, "a") as ncfile:
            ncfile.variables["t"][time_index] = time
            ncfile.variables["u"][time_index, :, :] = u
            ncfile.variables["v"][time_index, :, :] = v
            ncfile.variables["h"][time_index, :, :] = h

    # fig, ax = plt.subplots()
    # def make_frame(t):
    #     ax.clear()
    #     ax.plot(x, np.sinc(x**2) + np.sin(x + 2*np.pi/duration * t), lw=3)
    #     ax.set_ylim(-1.5, 2.5)
    #     return mplfig_to_npimage(fig)

    # animation = VideoClip(make_frame, duration=duration)
    # animation.ipython_display(fps=20, loop=True, autoplay=True)

    def run(
        self, u0, v0, h0, n_steps, path, log_every=25, array_store=False, pbar=True
    ):
        self.set_initial_state(u0, v0, h0)
        if log_every is not None:
            self.init_ncfile(path)
            i_written = 1
            if array_store:
                log_array = np.empty(
                    (int(n_steps // log_every) + 1, 3 * self.state_variable_length)
                )
        ite = 0
        if pbar:
            iterable = tqdm.tqdm(self.iterate_steps(), total=n_steps)
        else:
            iterable = self.iterate_steps()

        for u, v, h in iterable:
            if log_every is not None:
                if ite % log_every == 0:
                    self.write_ncfile(path, i_written, ite)
                    if array_store:
                        log_array[i_written - 1, :] = self.concatenate_state_var(
                            u, v, h
                        )
                    i_written += 1
            if ite > n_steps:
                break
            ite += 1
        if array_store:
            return (u, v, h), log_array
        else:
            return u, v, h

    def resume_run(self, resume_time, n_steps, path, log_every=25):
        for u, v, h in tqdm.tqdm(self.iterate_steps(), total=n_steps):
            ite, i_written = resume_time
            if log_every is not None:
                if ite % log_every == 0:
                    self.write_ncfile(path, i_written, ite)
                    i_written += 1
            if ite > n_steps:
                break
            ite += 1
        return u, v, h


class SWModelNumpy(SWModel):
    def __init__(self, n_x, dx, n_y, dy, periodic_x=True) -> None:
        super().__init__(n_x, dx, n_y, dy, periodic_x)
        self.x, self.y = (np.arange(self.n_x) * self.dx, np.arange(self.n_y) * self.dy)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        self.dt = 0.125 * min(self.dx, self.dy) / np.sqrt(self.gravity * self.depth)

        self.phase_speed = np.sqrt(self.gravity * self.depth)
        self.rossby_radius = (
            np.sqrt(self.gravity * self.depth) / self.coriolis_param.mean()
        )
        self.coriolis_param = self.coriolis_f + self.Y * self.coriolis_beta

    def set_initial_state(self, u0, v0, h0):
        self.u, self.v, self.h = (
            np.empty(self.grid_shape),
            np.empty(self.grid_shape),
            np.empty(self.grid_shape),
        )
        self.du, self.dv, self.dh = (
            np.empty(self.grid_shape),
            np.empty(self.grid_shape),
            np.empty(self.grid_shape),
        )
        self.du_new, self.dv_new, self.dh_new = (
            np.empty(self.grid_shape),
            np.empty(self.grid_shape),
            np.empty(self.grid_shape),
        )
        self.fe, self.fn = np.empty(self.grid_shape), np.empty(self.grid_shape)
        self.q, self.ke = np.empty(self.grid_shape), np.empty(self.grid_shape)

        # initial conditions
        self.u[...] = u0
        self.v[...] = v0
        self.h[...] = h0

        # boundary values of h must not be used
        self.h[0, :] = self.h[-1, :] = self.h[:, 0] = self.h[:, -1] = np.nan

        self.h = self.enforce_boundaries(self.h, "h")
        self.u = self.enforce_boundaries(self.u, "u")
        self.v = self.enforce_boundaries(self.v, "v")
        self.first_step = True

    def apply_linear_momentum_equation(self):
        self.v_avg = 0.25 * (
            self.v[1:-1, 1:-1] + self.v[:-2, 1:-1] + self.v[1:-1, 2:] + self.v[:-2, 2:]
        )
        self.du_new[1:-1, 1:-1] = (
            self.coriolis_param[1:-1, 1:-1] * self.v_avg
            - self.gravity * (self.h[2:, 1:-1] - self.h[1:-1, 1:-1]) / self.dx
        )  # du <- f * v_bar - g dh/dx
        self.u_avg = 0.25 * (
            self.u[1:-1, 1:-1] + self.u[1:-1, :-2] + self.u[2:, 1:-1] + self.u[2:, :-2]
        )
        self.dv_new[1:-1, 1:-1] = (
            -self.coriolis_param[1:-1, 1:-1] * self.u_avg
            - self.gravity * (self.h[1:-1, 2:] - self.h[1:-1, 1:-1]) / self.dy
        )  # dv = -f * u_bar - g * dh/dy

    def apply_nonlinear_momentum_equation(self):
        self.q[1:-1, 1:-1] = self.coriolis_param[1:-1, 1:-1] + (
            (self.v[2:, 1:-1] - self.v[1:-1, 1:-1]) / self.dx
            - (self.u[1:-1, 2:] - self.u[1:-1, 1:-1]) / self.dy
        )
        # potential vorticity
        # q <- f + (dv/dx - du/dv)
        self.q[1:-1, 1:-1] *= 1.0 / (
            0.25
            * (
                self.hc[1:-1, 1:-1]
                + self.hc[2:, 1:-1]
                + self.hc[1:-1, 2:]
                + self.hc[2:, 2:]
            )
        )  # q = (xi + f /h) where xi is the vertical component of vorticity
        self.q = self.enforce_boundaries(self.q, "h")

        self.du_new[1:-1, 1:-1] = -self.gravity * (
            self.h[2:, 1:-1] - self.h[1:-1, 1:-1]
        ) / self.dx + 0.5 * (
            self.q[1:-1, 1:-1] * 0.5 * (self.fn[1:-1, 1:-1] + self.fn[2:, 1:-1])
            + self.q[1:-1, :-2] * 0.5 * (self.fn[1:-1, :-2] + self.fn[2:, :-2])
        )  # du_new <- -g * dh/dx + (q * (fn_bar))
        self.dv_new[1:-1, 1:-1] = -self.gravity * (
            self.h[1:-1, 2:] - self.h[1:-1, 1:-1]
        ) / self.dy - 0.5 * (
            self.q[1:-1, 1:-1] * 0.5 * (self.fe[1:-1, 1:-1] + self.fe[1:-1, 2:])
            + self.q[:-2, 1:-1] * 0.5 * (self.fe[:-2, 1:-1] + self.fe[:-2, 2:])
        )  # dv_new <- -g * dh/dy - q * fe
        self.ke[1:-1, 1:-1] = 0.5 * (
            0.5 * (self.u[1:-1, 1:-1] ** 2 + self.u[:-2, 1:-1] ** 2)
            + 0.5 * (self.v[1:-1, 1:-1] ** 2 + self.v[1:-1, :-2] ** 2)
        )  # kinetic energy
        self.ke = self.enforce_boundaries(self.ke, "h")
        # see (4.30) https://empslocal.ex.ac.uk/people/staff/gv219/ecmm719/ess-ecmm719.pdf
        self.du_new[1:-1, 1:-1] += (
            -(self.ke[2:, 1:-1] - self.ke[1:-1, 1:-1]) / self.dx
        )  # du_new <- du_new - dke/dx
        self.dv_new[1:-1, 1:-1] += (
            -(self.ke[1:-1, 2:] - self.ke[1:-1, 1:-1]) / self.dy
        )  # dv_new <- dv_new - dke/dy

    def adam_bashforth_step(self):
        if self.first_step:
            self.u[1:-1, 1:-1] += self.dt * self.du_new[1:-1, 1:-1]
            self.v[1:-1, 1:-1] += self.dt * self.dv_new[1:-1, 1:-1]
            self.h[1:-1, 1:-1] += self.dt * self.dh_new[1:-1, 1:-1]
            self.first_step = False
        else:
            self.u[1:-1, 1:-1] += self.dt * (
                self.adams_bashforth_a * self.du_new[1:-1, 1:-1]
                + self.adams_bashforth_b * self.du[1:-1, 1:-1]
            )
            self.v[1:-1, 1:-1] += self.dt * (
                self.adams_bashforth_a * self.dv_new[1:-1, 1:-1]
                + self.adams_bashforth_b * self.dv[1:-1, 1:-1]
            )
            self.h[1:-1, 1:-1] += self.dt * (
                self.adams_bashforth_a * self.dh_new[1:-1, 1:-1]
                + self.adams_bashforth_b * self.dh[1:-1, 1:-1]
            )

        self.h = self.enforce_boundaries(self.h, "h")
        self.u = self.enforce_boundaries(self.u, "u")
        self.v = self.enforce_boundaries(self.v, "v")

    def apply_lateral_friction(self):
        # lateral friction
        self.fn[1:-1, 1:-1] = (
            self.lateral_viscosity * (self.u[1:-1, 2:] - self.u[1:-1, 1:-1]) / self.dy
        )  # fn <- visco* du/dy
        self.fe[1:-1, 1:-1] = (
            self.lateral_viscosity * (self.u[2:, 1:-1] - self.u[1:-1, 1:-1]) / self.dx
        )  # fe <- visco * du/dx
        self.fe = self.enforce_boundaries(self.fe, "u")
        self.fn = self.enforce_boundaries(self.fn, "v")

        self.u[1:-1, 1:-1] += self.dt * (
            (self.fe[1:-1, 1:-1] - self.fe[:-2, 1:-1]) / self.dx
            + (self.fn[1:-1, 1:-1] - self.fn[1:-1, :-2]) / self.dy
        )

        self.fe[1:-1, 1:-1] = (
            self.lateral_viscosity * (self.v[2:, 1:-1] - self.u[1:-1, 1:-1]) / self.dx
        )
        self.fn[1:-1, 1:-1] = (
            self.lateral_viscosity * (self.v[1:-1, 2:] - self.u[1:-1, 1:-1]) / self.dy
        )
        self.fe = self.enforce_boundaries(self.fe, "u")
        self.fn = self.enforce_boundaries(self.fn, "v")

        self.v[1:-1, 1:-1] += self.dt * (
            (self.fe[1:-1, 1:-1] - self.fe[:-2, 1:-1]) / self.dx
            + (self.fn[1:-1, 1:-1] - self.fn[1:-1, :-2]) / self.dy
        )

        # rotate quantities; keep last increment for AB method
        self.du[...] = self.du_new
        self.dv[...] = self.dv_new
        self.dh[...] = self.dh_new

    def apply_source_terms(self):
        pass

    def step(self):
        self.hc = np.pad(self.h[1:-1, 1:-1], 1, "edge")
        self.hc = self.enforce_boundaries(self.hc, "h")

        self.fe[1:-1, 1:-1] = (
            0.5 * (self.hc[1:-1, 1:-1] + self.hc[2:, 1:-1]) * self.u[1:-1, 1:-1]
        )  # 0.5(h_{n} + h_{n+1}) * u (at east boundary) f_{i+1/2, j}
        self.fn[1:-1, 1:-1] = (
            0.5 * (self.hc[1:-1, 1:-1] + self.hc[1:-1, 2:]) * self.v[1:-1, 1:-1]
        )  # at north boundary f_{i, j+1/2}
        self.fe = self.enforce_boundaries(self.fe, "u")
        self.fn = self.enforce_boundaries(self.fn, "v")

        self.dh_new[1:-1, 1:-1] = -(
            (self.fe[1:-1, 1:-1] - self.fe[:-2, 1:-1]) / self.dx
            + (self.fn[1:-1, 1:-1] - self.fn[1:-1, :-2]) / self.dy
        )
        if self.linear_momentum_equation:
            self.apply_linear_momentum_equation()
        else:
            self.apply_nonlinear_momentum_equation()
        self.apply_source_terms()
        self.adam_bashforth_step()
        if self.lateral_viscosity > 0:
            self.apply_lateral_friction()
        return self.u, self.v, self.h


class SWModelJax(SWModel):
    def __init__(self, n_x, dx, n_y, dy, periodic_x=True) -> None:
        super().__init__(n_x, dx, n_y, dy, periodic_x)

        self.x, self.y = (
            jnp.arange(self.n_x) * self.dx,
            jnp.arange(self.n_y) * self.dy,
        )
        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing="ij")
        self.dt = 0.125 * min(self.dx, self.dy) / jnp.sqrt(self.gravity * self.depth)

        self.phase_speed = jnp.sqrt(self.gravity * self.depth)
        self.coriolis_param = self.coriolis_f + self.Y * self.coriolis_beta
        self.rossby_radius = (
            jnp.sqrt(self.gravity * self.depth) / self.coriolis_param.mean()
        )

    # @partial(jax.jit, static_argnums=(0, 2))
    def enforce_boundaries(self, arr, grid):
        assert grid in ("h", "u", "v")
        new_arr = arr.at[...].get()
        if self.periodic_boundary_x:
            new_arr = new_arr.at[0, :].set(arr[-2, :])
            new_arr = new_arr.at[-1, :].set(arr[1, :])
        elif grid == "u":
            new_arr = new_arr.at[-2, :].set(0.0)
        if grid == "v":
            new_arr = new_arr.at[:, -2].set(0.0)
        # print(f"{((new_arr - arr) ==0).all()=}")
        return new_arr

    @partial(jax.jit, static_argnums=0)
    def concatenate_state_var(self, u, v, h):
        return jnp.concatenate([u.flatten(), v.flatten(), h.flatten()])

    def set_initial_state(self, u0, v0, h0):
        self.u, self.v, self.h = (
            jnp.empty(self.grid_shape),
            jnp.empty(self.grid_shape),
            jnp.empty(self.grid_shape),
        )
        self.du, self.dv, self.dh = (
            jnp.empty(self.grid_shape),
            jnp.empty(self.grid_shape),
            jnp.empty(self.grid_shape),
        )
        self.du_new, self.dv_new, self.dh_new = (
            jnp.empty(self.grid_shape),
            jnp.empty(self.grid_shape),
            jnp.empty(self.grid_shape),
        )
        self.fe, self.fn = jnp.empty(self.grid_shape), jnp.empty(self.grid_shape)
        self.q, self.ke = jnp.empty(self.grid_shape), jnp.empty(self.grid_shape)

        # initial conditions
        self.u = self.u.at[...].set(u0)
        self.v = self.v.at[...].set(v0)
        self.h = self.h.at[...].set(h0)

        # boundary values of h must not be used
        # self.h[0, :] = self.h[-1, :] = self.h[:, 0] = self.h[:, -1] = jnp.nan
        self.h = self.h.at[0, :].set(jnp.nan)
        self.h = self.h.at[-1, :].set(jnp.nan)
        self.h = self.h.at[:, 0].set(jnp.nan)
        self.h = self.h.at[:, -1].set(jnp.nan)

        self.h = self.enforce_boundaries(self.h, "h")
        self.u = self.enforce_boundaries(self.u, "u")
        self.v = self.enforce_boundaries(self.v, "v")
        self.first_step = True
        return self.u, self.v, self.h

    def update_q_stateless(self, q, hc, u, v, dx, dy, coriolis_param):
        q = q.at[1:-1, 1:-1].set(
            coriolis_param[1:-1, 1:-1]
            + ((v[2:, 1:-1] - v[1:-1, 1:-1]) / dx - (u[1:-1, 2:] - u[1:-1, 1:-1]) / dy)
        )
        # potential vorticity
        q = q.at[1:-1, 1:-1].multiply(
            1.0 / (0.25 * (hc[1:-1, 1:-1] + hc[2:, 1:-1] + hc[1:-1, 2:] + hc[2:, 2:]))
        )
        q = self.enforce_boundaries(q, "h")
        return q

    def update_dvel(self, q, du_new, dv_new, h, fn, fe, dx, dy, gravity):
        du_new = du_new.at[1:-1, 1:-1].set(
            -gravity * (h[2:, 1:-1] - h[1:-1, 1:-1]) / dx
            + 0.5
            * (
                q[1:-1, 1:-1] * 0.5 * (fn[1:-1, 1:-1] + fn[2:, 1:-1])
                + q[1:-1, :-2] * 0.5 * (fn[1:-1, :-2] + fn[2:, :-2])
            )
        )
        dv_new = dv_new.at[1:-1, 1:-1].set(
            -gravity * (h[1:-1, 2:] - h[1:-1, 1:-1]) / dy
            - 0.5
            * (
                q[1:-1, 1:-1] * 0.5 * (fe[1:-1, 1:-1] + fe[1:-1, 2:])
                + q[:-2, 1:-1] * 0.5 * (fe[:-2, 1:-1] + fe[:-2, 2:])
            )
        )
        return du_new, dv_new

    def update_ke_stateless(self, ke, u, v):
        ke = ke.at[1:-1, 1:-1].set(
            0.5
            * (
                0.5 * (u[1:-1, 1:-1] ** 2 + u[:-2, 1:-1] ** 2)
                + 0.5 * (v[1:-1, 1:-1] ** 2 + v[1:-1, :-2] ** 2)
            )
        )
        ke = self.enforce_boundaries(ke, "h")
        return ke

    def update_duv_new_ke_stateless(self, ke, du_new, dv_new, dx, dy):
        du_new = du_new.at[1:-1, 1:-1].add(-(ke[2:, 1:-1] - ke[1:-1, 1:-1]) / dx)
        dv_new = dv_new.at[1:-1, 1:-1].add(-(ke[1:-1, 2:] - ke[1:-1, 1:-1]) / dy)
        return du_new, dv_new

    def apply_nonlinear_momentum_equation_stateless(self):
        self.q = self.update_q_stateless(
            self.q,
            self.hc,
            self.u,
            self.v,
            self.dx,
            self.dy,
            self.coriolis_param,
        )
        self.du_new, self.dv_new = self.update_dvel(
            self.q,
            self.du_new,
            self.dv_new,
            self.h,
            self.fn,
            self.fe,
            self.dx,
            self.dy,
            self.gravity,
        )
        self.ke = self.update_ke_stateless(
            self.ke,
            self.u,
            self.v,
        )
        self.du_new, self.dv_new = self.update_duv_new_ke_stateless(
            self.ke,
            self.du_new,
            self.dv_new,
            self.dx,
            self.dy,
        )

    def apply_nonlinear_momentum_equation(self):
        # update q: potential vorticity
        self.q = self.q.at[1:-1, 1:-1].set(
            self.coriolis_param[1:-1, 1:-1]
            + (
                (self.v[2:, 1:-1] - self.v[1:-1, 1:-1]) / self.dx
                - (self.u[1:-1, 2:] - self.u[1:-1, 1:-1]) / self.dy
            )
        )
        self.q = self.q.at[1:-1, 1:-1].multiply(
            1.0
            / (
                0.25
                * (
                    self.hc[1:-1, 1:-1]
                    + self.hc[2:, 1:-1]
                    + self.hc[1:-1, 2:]
                    + self.hc[2:, 2:]
                )
            )
        )
        self.q = self.enforce_boundaries(self.q, "h")

        self.du_new = self.du_new.at[1:-1, 1:-1].set(
            -self.gravity * (self.h[2:, 1:-1] - self.h[1:-1, 1:-1]) / self.dx
            + 0.5
            * (
                self.q[1:-1, 1:-1] * 0.5 * (self.fn[1:-1, 1:-1] + self.fn[2:, 1:-1])
                + self.q[1:-1, :-2] * 0.5 * (self.fn[1:-1, :-2] + self.fn[2:, :-2])
            )
        )
        self.dv_new = self.dv_new.at[1:-1, 1:-1].set(
            -self.gravity * (self.h[1:-1, 2:] - self.h[1:-1, 1:-1]) / self.dy
            - 0.5
            * (
                self.q[1:-1, 1:-1] * 0.5 * (self.fe[1:-1, 1:-1] + self.fe[1:-1, 2:])
                + self.q[:-2, 1:-1] * 0.5 * (self.fe[:-2, 1:-1] + self.fe[:-2, 2:])
            )
        )
        # update ke
        self.ke = self.ke.at[1:-1, 1:-1].set(
            0.5
            * (
                0.5 * (self.u[1:-1, 1:-1] ** 2 + self.u[:-2, 1:-1] ** 2)
                + 0.5 * (self.v[1:-1, 1:-1] ** 2 + self.v[1:-1, :-2] ** 2)
            )
        )
        self.ke = self.enforce_boundaries(self.ke, "h")
        # udpate d_velocity_new
        self.du_new = self.du_new.at[1:-1, 1:-1].add(
            -(self.ke[2:, 1:-1] - self.ke[1:-1, 1:-1]) / self.dx
        )
        self.dv_new = self.dv_new.at[1:-1, 1:-1].add(
            -(self.ke[1:-1, 2:] - self.ke[1:-1, 1:-1]) / self.dy
        )
        # return self.q, self.ke, self.du_new, self.dv_new

    def adam_bashforth_step(self):
        if self.first_step:
            self.u = self.u.at[1:-1, 1:-1].add(self.dt * self.du_new[1:-1, 1:-1])
            self.v = self.v.at[1:-1, 1:-1].add(self.dt * self.dv_new[1:-1, 1:-1])
            self.h = self.h.at[1:-1, 1:-1].add(self.dt * self.dh_new[1:-1, 1:-1])
            self.first_step = False
        else:
            self.u = self.u.at[1:-1, 1:-1].add(
                self.dt
                * (
                    self.adams_bashforth_a * self.du_new[1:-1, 1:-1]
                    + self.adams_bashforth_b * self.du[1:-1, 1:-1]
                )
            )
            self.v = self.v.at[1:-1, 1:-1].add(
                self.dt
                * (
                    self.adams_bashforth_a * self.dv_new[1:-1, 1:-1]
                    + self.adams_bashforth_b * self.dv[1:-1, 1:-1]
                )
            )
            self.h = self.h.at[1:-1, 1:-1].add(
                self.dt
                * (
                    self.adams_bashforth_a * self.dh_new[1:-1, 1:-1]
                    + self.adams_bashforth_b * self.dh[1:-1, 1:-1]
                )
            )

        self.h = self.enforce_boundaries(self.h, "h")
        self.u = self.enforce_boundaries(self.u, "u")
        self.v = self.enforce_boundaries(self.v, "v")

    # @partial(jax.jit, static_argnums=0)
    def adam_bashforth_step_stateless(
        self, u, v, h, du, dv, dh, dt, du_new, dv_new, dh_new, a, b, first_step
    ):
        # if first_step:
        u = u.at[1:-1, 1:-1].add(dt * du_new[1:-1, 1:-1])
        v = v.at[1:-1, 1:-1].add(dt * dv_new[1:-1, 1:-1])
        h = h.at[1:-1, 1:-1].add(dt * dh_new[1:-1, 1:-1])
        # first_step = False
        # else:
        #     u = u.at[1:-1, 1:-1].add(dt * (a * du_new[1:-1, 1:-1] + b * du[1:-1, 1:-1]))
        #     v = v.at[1:-1, 1:-1].add(dt * (a * dv_new[1:-1, 1:-1] + b * dv[1:-1, 1:-1]))
        #     h = h.at[1:-1, 1:-1].add(dt * (a * dh_new[1:-1, 1:-1] + b * dh[1:-1, 1:-1]))

        u = self.enforce_boundaries(u, "u")
        v = self.enforce_boundaries(v, "v")
        h = self.enforce_boundaries(h, "h")
        return u, v, h

    def apply_lateral_friction(self):
        # lateral friction
        self.fn = self.fn.at[1:-1, 1:-1].set(
            self.lateral_viscosity * (self.u[1:-1, 2:] - self.u[1:-1, 1:-1]) / self.dy
        )
        self.fe = self.fe.at[1:-1, 1:-1].set(
            self.lateral_viscosity * (self.u[2:, 1:-1] - self.u[1:-1, 1:-1]) / self.dx
        )
        self.fe = self.enforce_boundaries(self.fe, "u")
        self.fn = self.enforce_boundaries(self.fn, "v")

        self.u = self.u.at[1:-1, 1:-1].add(
            self.dt
            * (
                (self.fe[1:-1, 1:-1] - self.fe[:-2, 1:-1]) / self.dx
                + (self.fn[1:-1, 1:-1] - self.fn[1:-1, :-2]) / self.dy
            )
        )

        self.fn = self.fn.at[1:-1, 1:-1].set(
            self.lateral_viscosity * (self.v[1:-1, 2:] - self.u[1:-1, 1:-1]) / self.dy
        )
        self.fe = self.fe.at[1:-1, 1:-1].set(
            self.lateral_viscosity * (self.v[2:, 1:-1] - self.u[1:-1, 1:-1]) / self.dx
        )
        self.fe = self.enforce_boundaries(self.fe, "u")
        self.fn = self.enforce_boundaries(self.fn, "v")

        self.v = self.v.at[1:-1, 1:-1].add(
            self.dt
            * (
                (self.fe[1:-1, 1:-1] - self.fe[:-2, 1:-1]) / self.dx
                + (self.fn[1:-1, 1:-1] - self.fn[1:-1, :-2]) / self.dy
            )
        )
        # rotate quantities
        self.du = self.du.at[...].set(self.du_new)
        self.dv = self.dv.at[...].set(self.dv_new)
        self.dh = self.dh.at[...].set(self.dh_new)

    def step(self):
        self.hc = jnp.pad(self.h[1:-1, 1:-1], 1, "edge")
        self.hc = self.enforce_boundaries(self.hc, "h")

        self.fe = self.fe.at[1:-1, 1:-1].set(
            0.5 * (self.hc[1:-1, 1:-1] + self.hc[2:, 1:-1]) * self.u[1:-1, 1:-1]
        )
        self.fn = self.fn.at[1:-1, 1:-1].set(
            0.5 * (self.hc[1:-1, 1:-1] + self.hc[1:-1, 2:]) * self.v[1:-1, 1:-1]
        )
        self.fe = self.enforce_boundaries(self.fe, "u")
        self.fn = self.enforce_boundaries(self.fn, "v")

        self.dh_new = self.dh_new.at[1:-1, 1:-1].set(
            -(
                (self.fe[1:-1, 1:-1] - self.fe[:-2, 1:-1]) / self.dx
                + (self.fn[1:-1, 1:-1] - self.fn[1:-1, :-2]) / self.dy
            )
        )

        self.apply_nonlinear_momentum_equation()
        self.adam_bashforth_step()
        # self.u, self.v, self.h = self.adam_bashforth_step_stateless(
        #     self.u,
        #     self.v,
        #     self.h,
        #     self.du,
        #     self.dv,
        #     self.dh,
        #     self.dt,
        #     self.du_new,
        #     self.dv_new,
        #     self.dh_new,
        #     self.adams_bashforth_a,
        #     self.adams_bashforth_a,
        #     self.first_step,
        # )
        if self.lateral_viscosity > 0:
            self.apply_lateral_friction()
        return self.u, self.v, self.h

    def forward(self, state, n_steps, aux=False, pbar=True):
        u, v, h = self.separate_state_vector(state)
        new_state = self.concatenate_state_var(
            *self.run(u, v, h, n_steps, log_every=None, path=None, pbar=pbar)
        )
        if aux:
            return new_state, new_state
        else:
            return new_state

    def forward_TLM(self, state, n_steps):
        def fun_to_diff(state):
            state_to_modify, full_state = self.forward(state, n_steps, aux=True)
            return state_to_modify, full_state

        tl = jax.jacfwd(
            fun_to_diff,
            has_aux=True,
        )
        return tl(state)

    def forward_TLM_h(self, state, n_steps):
        def fun_to_diff(state):
            state_to_modify, full_state = self.forward(state, n_steps, aux=True)
            return state_to_modify[self.slice_h], full_state

        tl = jax.jacfwd(
            fun_to_diff,
            has_aux=True,
        )
        return tl(state)


if __name__ == "__main__":
    n_x = 25
    dx = 5e3
    n_y = 25
    dy = 5e3
    model = SWModelJax(n_x, dx, n_y, dy, periodic_x=True)

    u0 = 0 * np.exp(-((model.Y - model.y[n_y // 2]) ** 2) / (0.02 * model.l_x) ** 2)
    v0 = np.zeros_like(u0)
    h_geostrophy = model.compute_h_geostrophy(u0)
    h0 = (
        model.depth
        + h_geostrophy
        # make sure h0 is centered around depth
        - h_geostrophy.mean()
        # small perturbation
        + 0.2
        * np.sin(model.X / model.l_x * 10 * np.pi)
        * np.cos(model.Y / model.l_y * 8 * np.pi)
    )
    final_time = 24  # in hours
    n_steps = final_time * 3600
    log_every_n_seconds = 60  # every minutes
    state_run = model.concatenate_state_var(
        *model.run(
            u0,
            v0,
            h0,
            n_steps=n_steps,
            path="/home/sw/script.nc",
            log_every=log_every_n_seconds,
        )
    )
    # initial_state = model.concatenate_state_var(u0, v0, h0)

    # new_state = model.forward(initial_state, n_steps)
    # plt.plot(initial_state, label="initial")
    # plt.plot(new_state, label="forward")
    # plt.plot(state_run, ":", label="run")
    # plt.legend()
    # plt.savefig("/home/sw/hello.png")
    # integration_time = [0.5, 1, 5, 10, 60]  # in minutes
    # plt.set_cmap("Spectral")
    # for i, minutes in enumerate(integration_time):
    #     tl, forw = model.forward_TLM_h(initial_state, minutes * 60)
    #     print(f"{tl.shape=}")
    #     print(f"{forw.shape=}")
    #     plt.subplot(2, 2, 1)
    #     gn = tl.T @ tl
    #     plt.imshow(gn)
    #     plt.title(f"{minutes} minutes")
    #     plt.colorbar()
    #     plt.subplot(2, 2, 2)
    #     plt.plot(jnp.linalg.svd(gn)[1])
    #     plt.yscale("log")
    #     plt.subplot(2, 2, 4)
    #     plt.plot(jnp.linalg.svd(gn)[1])
    #     # plt.yscale("log")
    #     plt.savefig(f"/home/sw/gaussnewton_h_{i}_s.png")
    #     plt.close()
