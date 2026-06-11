import numpy as np
import casadi as ca
from casadi import MX, Opti, DM

from physics.aircraft_kinematics_max_constrains import AircraftKinematicsMaxConstrains
from physics.aircraft_kinematics_state import AircraftKinematicsState


def state_to_casadi(s: AircraftKinematicsState) -> np.ndarray:
    return np.array([
        s.x,
        s.y,
        s.h,
        s.v,
        np.radians(s.chi),
        np.radians(s.gamma),
    ])


class LandingTrajectoryOptimizer:
    """
    Optimization objectives
    The optimizer minimizes a weighted cost function consisting of:

    1. Terminal position error: x position, y position, altitude z
    2. Terminal velocity error: airspeed v at touchdown
    3. Terminal heading error: course angle chi
    4. Terminal flight path angle error: gamma (desired descent angle)
    5. Total flight time: shorter trajectories are preferred
    6. Control smoothness: minimize thrust changes, minimize load factor changes, minimize roll angle changes
    7. Minimal height: above 0
    """

    def __init__(
        self,
        state:           AircraftKinematicsState,
        max_constraints: AircraftKinematicsMaxConstrains,
        N:               int   = 80,
        T_min:           float = 60.0,
        T_max:           float = 1800.0,
    ) -> None:
        self.N: int = N
        self.T_min: float = T_min
        self.T_max: float = T_max
        self.state: AircraftKinematicsState = state
        self.max_constraints: AircraftKinematicsMaxConstrains = max_constraints

        self.opti: Opti = ca.Opti()
        self.X: MX = self.opti.variable(6, N + 1)    # [x, y, z, v, chi, gamma]
        self.U: MX = self.opti.variable(3, N)        # [thrust, load_factor, roll_deg]
        self.T: MX = self.opti.variable()                  # free solving time t [s]

        self.x0_param: MX = self.opti.parameter(6)   # start state
        self.goal_param: MX = self.opti.parameter(4)   # [x, y, z, v_td]
        self.heading_param: MX = self.opti.parameter(1)   # χ_goal [rad]

        self._build()

    def _f(
            self,
            xk: MX,
            uk: MX,
            dt: float
    ) -> DM:
        x = xk[0]
        y = xk[1]
        z = xk[2]
        v = xk[3]
        chi = xk[4]
        gamma = xk[5]

        thrust = uk[0]
        load_factor = uk[1]
        roll_deg = uk[2]
        phi = roll_deg * ca.pi / 180.0

        vx = v * ca.cos(gamma) * ca.sin(chi)
        vy = v * ca.cos(gamma) * ca.cos(chi)
        vz = v * ca.sin(gamma)

        v_safe = ca.fmax(v, 1.0)
        tx = vx / v_safe; ty = vy / v_safe; tz = vz / v_safe

        b_norm = ca.sqrt(tx**2 + ty**2 + 1e-8)
        bx =  ty / b_norm; by = -tx / b_norm

        nx  =  by * tz;        ny  = -bx * tz;        nz  = bx * ty - by * tx
        tnx = ty * nz - tz * ny; tny = tz * nx - tx * nz; tnz = tx * ny - ty * nx

        nr_x = nx * ca.cos(phi) + tnx * ca.sin(phi)
        nr_y = ny * ca.cos(phi) + tny * ca.sin(phi)
        nr_z = nz * ca.cos(phi) + tnz * ca.sin(phi)

        rho    = self.state.rho0 * ca.exp(-ca.fmax(z, 0.0) / self.state.H)
        v2     = v_safe**2
        CL_req = (2.0 * load_factor * self.state.mass * self.state.g / (rho * v2 * self.state.area))
        CL_act = ca.fmin(CL_req, self.max_constraints.cl_max)
        CD     = self.state.C_D0 + self.state.k * CL_act**2
        q      = 0.5 * rho * v2
        drag   = q * CD  * self.state.area
        lift   = q * CL_act * self.state.area

        acc  = (thrust - drag) / self.state.mass
        vd_x = acc * tx + (lift / self.state.mass) * nr_x
        vd_y = acc * ty + (lift / self.state.mass) * nr_y
        vd_z = acc * tz + (lift / self.state.mass) * nr_z - self.state.g

        vx_n = vx + vd_x * dt
        vy_n = vy + vd_y * dt
        vz_n = vz + vd_z * dt

        x_new = x + vx_n * dt
        y_new = y + vy_n * dt
        z_new = z + vz_n * dt

        v_new = ca.sqrt(vx_n**2 + vy_n**2 + vz_n**2 + 1e-8)

        v_dot     = (vx_n * vd_x + vy_n * vd_y + vz_n * vd_z) / ca.fmax(v_new, 1e-6)
        cos_g     = ca.fmax(ca.cos(gamma), 1e-4)
        gamma_dot = (v_new * vd_z - vz_n * v_dot) / ca.fmax(v_new**2 * cos_g, 1e-6)
        gamma_new = gamma + gamma_dot * dt

        chi_dot = self.state.g * ca.tan(phi) / ca.fmax(v_new, 1e-6)
        chi_new = chi + chi_dot * dt

        return ca.vertcat(x_new, y_new, z_new, v_new, chi_new, gamma_new)

    def _build(self) -> None:
        N = self.N

        self.opti.subject_to(self.T >= self.T_min)
        self.opti.subject_to(self.T <= self.T_max)

        # normalized time step
        #   Δτ = 1/N
        #   dt = T · Δτ = T/N
        #
        #   τ_k = k/N ∈ [0,1] is always the same
        #   T only scales the dynamic, not the gitter.
        delta_tau = 1.0 / N             # fixed time step ∈ [0,1]
        dt        = self.T * delta_tau  # physical step = T · Δτ

        self.opti.subject_to(self.X[:, 0] == self.x0_param)

        # x_{k+1} = x_k + T·Δτ · f(x_k, u_k)
        for k in range(N):
            self.opti.subject_to(
                self.X[:, k + 1] == self._f(self.X[:, k], self.U[:, k], dt)
            )

        self.opti.subject_to(self.U[0, :] >= 0.0)         # thrust ≥ 0
        self.opti.subject_to(self.U[0, :] <= 3e5)         # thrust ≤ 300 kN
        self.opti.subject_to(self.U[1, :] >= 0.3)         # n ≥ 0.3
        self.opti.subject_to(self.U[1, :] <= 2.5)         # n ≤ 2.5
        self.opti.subject_to(self.U[2, :] >= -60.0)       # roll ≥ −60°
        self.opti.subject_to(self.U[2, :] <=  60.0)       # roll ≤  60°

        self.opti.subject_to(self.X[2, :] >= 0.0)         # z ≥ 0 (terrain)
        self.opti.subject_to(self.X[3, :] >= 70.0)        # v ≥ 10 m/s (no Stall)
        self.opti.subject_to(self.X[5, :] >= -0.70)       # γ ≥ −40°
        self.opti.subject_to(self.X[5, :] <=  0.35)       # γ ≤  20°

        gx = self.goal_param[0]; gy = self.goal_param[1]
        gz = self.goal_param[2]; gv = self.goal_param[3]
        chi_goal = self.heading_param[0]  # [rad]
        gam_goal = -3.0 * ca.pi / 180.0  # −3 ° fixed glide path

        TOL_XY = 5.0  # m lateral / longitudinal
        TOL_Z = 5.0  # m altitude at threshold
        TOL_CHI = 3.0 * ca.pi / 180.0  # rad heading (±3°)
        TOL_GAM = 1.0 * ca.pi / 180.0  # rad glideslope (±1°)

        self.opti.subject_to(self.X[0, N] >= gx - TOL_XY)
        self.opti.subject_to(self.X[0, N] <= gx + TOL_XY)
        self.opti.subject_to(self.X[1, N] >= gy - TOL_XY)
        self.opti.subject_to(self.X[1, N] <= gy + TOL_XY)
        self.opti.subject_to(self.X[2, N] >= gz - TOL_Z)
        self.opti.subject_to(self.X[2, N] <= gz + TOL_Z)

        e_chi = ca.atan2(
            ca.sin(self.X[4, N] - chi_goal),
            ca.cos(self.X[4, N] - chi_goal),
        )
        self.opti.subject_to(e_chi >= -TOL_CHI)
        self.opti.subject_to(e_chi <= TOL_CHI)

        e_gam = self.X[5, N] - gam_goal
        self.opti.subject_to(e_gam >= -TOL_GAM)
        self.opti.subject_to(e_gam <= TOL_GAM)

        e_v = (self.X[3, N] - gv) / 20.0
        J = 0.5 * e_v ** 2
        J += 0.05 * (self.T / self.T_max)

        smooth = ca.MX(0.0)
        for k in range(N - 1):
            smooth += (
                ((self.U[0, k+1] - self.U[0, k]) / 30000.0) ** 2
              + ((self.U[1, k+1] - self.U[1, k]) /    0.3 ) ** 2
              + ((self.U[2, k+1] - self.U[2, k]) /   10.0 ) ** 2
            )
        J += 0.1 * smooth / (N - 1)

        self.opti.minimize(J)

        self.opti.solver("ipopt", {}, {
            "max_iter":           2000,
            "tol":                1e-4,
            "acceptable_tol":     1e-3,
            "acceptable_iter":      15,
            "print_level":           3,
            "linear_solver":    "mumps",
            "mu_strategy":   "adaptive",
            "bound_relax_factor": 1e-8,
        })

    def solve(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        heading_goal_deg: float,
        x_init: np.ndarray | None = None,
        u_init: np.ndarray | None = None,
        T_init: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        X_val : ndarray (6, N+1)  — optimized state trajectory
        U_val : ndarray (3, N)    — optimized control sequence
        T_val : float             — optimized landing time [s]
        """
        self.opti.set_value(self.x0_param,     x0)
        self.opti.set_value(self.goal_param,    goal)
        self.opti.set_value(self.heading_param, float(np.radians(heading_goal_deg)))

        if T_init is not None:
            T_guess = float(T_init)
        else:
            dist    = np.linalg.norm(goal[:3] - x0[:3])
            v_avg   = 0.5 * (float(x0[3]) + max(float(goal[3]), 30.0))
            T_guess = float(np.clip(dist / v_avg * 1.5, self.T_min, self.T_max))

        self.opti.set_initial(self.T, T_guess)

        if x_init is not None:
            self.opti.set_initial(self.X, x_init)
        else:
            pos0     = x0[:3]
            v0       = float(x0[3])
            chi0     = float(x0[4])
            gam0     = float(x0[5])
            pos_goal = goal[:3]
            v_goal   = max(float(goal[3]), 30.0)
            chi_goal = float(np.radians(heading_goal_deg))
            d3       = pos_goal - pos0
            d_chi    = ((chi_goal - chi0) + np.pi) % (2 * np.pi) - np.pi
            gam_goal = float(np.arctan2(d3[2], np.linalg.norm(d3[:2]) + 1e-6))

            for k in range(self.N + 1):
                tau = k / self.N
                self.opti.set_initial(self.X[:, k], np.array([
                    *(pos0 + tau * d3),
                    max(v0 + tau * (v_goal - v0), 30.0),
                    chi0 + tau * d_chi,
                    gam0 + tau * (gam_goal - gam0),
                ]))

        if u_init is not None:
            self.opti.set_initial(self.U, u_init)
        else:
            for k in range(self.N):
                self.opti.set_initial(self.U[:, k], [120_000.0, 1.0, 0.0])

        try:
            sol   = self.opti.solve()
            X_val = sol.value(self.X)
            U_val = sol.value(self.U)
            T_val = float(sol.value(self.T))
            print(f"[LandingOptimizer] IPOPT converged  T = {T_val:.1f} s")
        except RuntimeError as e:
            print(f"[LandingOptimizer] IPOPT failed: {e}\n -> best-so-far")
            X_val = self.opti.debug.value(self.X)
            U_val = self.opti.debug.value(self.U)
            T_val = float(self.opti.debug.value(self.T))

        return X_val, U_val, T_val