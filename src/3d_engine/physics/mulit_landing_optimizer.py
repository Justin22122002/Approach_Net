import numpy as np
import casadi as ca
from casadi import Opti, MX, DM

from physics.aircraft_kinematics_max_constrains import AircraftKinematicsMaxConstrains
from physics.aircraft_kinematics_state import AircraftKinematicsState


def state_to_casadi(s: AircraftKinematicsState) -> np.ndarray:
    return np.array([
        s.x,
        s.y,
        s.h,
        s.v,
        np.radians(s.chi),
        np.radians(s.gamma)
    ])


class MultiLandingTrajectoryOptimizer:
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
    8. horizontal and vertical separation: 1000m x 300m
    9. time separation between flight landings: 90s
    """

    def __init__(
        self,
        states:           list[AircraftKinematicsState],
        max_constraints:  list[AircraftKinematicsMaxConstrains],
        N:                int   = 80,
        T_min:            float = 60.0,
        T_max:            float = 600.0,
        D_horizontal:     float = 1000.0,
        D_vertical:       float = 300.0,
        sep_stride:       int   = 5,
    ) -> None:
        self.N: int                                                 = N
        self.T_min: float                                           = T_min
        self.T_max: float                                           = T_max
        self.M: int                                                 = len(states)
        self.states: list[AircraftKinematicsState]                  = states
        self.max_constraints: list[AircraftKinematicsMaxConstrains] = max_constraints
        self.D_h2: float                                            = D_horizontal ** 2
        self.D_v2: float                                            = D_vertical   ** 2
        self.sep_stride: int                                        = max(1, sep_stride)

        self.opti: Opti = ca.Opti()

        # states: (6*M, N+1), Controls: (3*M, N)
        self.X: MX = self.opti.variable(6 * self.M, N + 1)
        self.U: MX = self.opti.variable(3 * self.M, N)

        # individual T per aircraft
        self.T_list: list[MX] = [self.opti.variable() for _ in range(self.M)]

        self.x0_param: MX      = self.opti.parameter(6 * self.M)
        self.goal_param: MX    = self.opti.parameter(4 * self.M)
        self.heading_param: MX = self.opti.parameter(self.M)

        self._build()

    def _f(
            self,
            xk: MX,
            uk: MX,
            dt: float,
            s: AircraftKinematicsState,
            c: AircraftKinematicsMaxConstrains
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
        bx = ty / b_norm; by = -tx / b_norm

        nx  =  by * tz;        ny  = -bx * tz;        nz  = bx * ty - by * tx
        tnx = ty * nz - tz * ny; tny = tz * nx - tx * nz; tnz = tx * ny - ty * nx

        nr_x = nx * ca.cos(phi) + tnx * ca.sin(phi)
        nr_y = ny * ca.cos(phi) + tny * ca.sin(phi)
        nr_z = nz * ca.cos(phi) + tnz * ca.sin(phi)

        rho    = s.rho0 * ca.exp(-ca.fmax(z, 0.0) / s.H)
        v2     = v_safe ** 2
        CL_req = 2.0 * load_factor * s.mass * s.g / (rho * v2 * s.area)
        CL_act = ca.fmin(CL_req, c.cl_max)
        CD     = s.C_D0 + s.k * CL_act ** 2
        q      = 0.5 * rho * v2
        drag   = q * CD  * s.area
        lift   = q * CL_act * s.area

        acc  = (thrust - drag) / s.mass
        vd_x = acc * tx + (lift / s.mass) * nr_x
        vd_y = acc * ty + (lift / s.mass) * nr_y
        vd_z = acc * tz + (lift / s.mass) * nr_z - s.g

        vx_n = vx + vd_x * dt; vy_n = vy + vd_y * dt; vz_n = vz + vd_z * dt
        x_new = x + vx_n * dt;  y_new = y + vy_n * dt;  z_new = z + vz_n * dt

        v_new = ca.sqrt(vx_n**2 + vy_n**2 + vz_n**2 + 1e-8)

        v_dot     = (vx_n*vd_x + vy_n*vd_y + vz_n*vd_z) / ca.fmax(v_new, 1e-6)
        cos_g     = ca.fmax(ca.cos(gamma), 1e-4)
        gamma_dot = (v_new*vd_z - vz_n*v_dot) / ca.fmax(v_new**2 * cos_g, 1e-6)
        gamma_new = gamma + gamma_dot * dt

        chi_dot   = s.g * ca.tan(phi) / ca.fmax(v_new, 1e-6)
        chi_new   = chi + chi_dot * dt

        return ca.vertcat(x_new, y_new, z_new, v_new, chi_new, gamma_new)

    def _build(self) -> None:
        N = self.N; M = self.M

        # start condition
        self.opti.subject_to(self.X[:, 0] == self.x0_param)

        J = ca.MX(0.0)

        # individual t-bound and time penalty
        for i, Ti in enumerate(self.T_list):
            self.opti.subject_to(Ti >= self.T_min)
            self.opti.subject_to(Ti <= self.T_max)

        # dynamic constants per airplane
        for i in range(M):
            si  = i * 6; ci = i * 3
            dt_i = self.T_list[i] / N

            for k in range(N):
                xk   = self.X[si:si+6, k]
                xnext= self.X[si:si+6, k+1]
                uk   = self.U[ci:ci+3, k]
                self.opti.subject_to(xnext == self._f(xk, uk, dt_i, self.states[i], self.max_constraints[i]))

                # Control-Bounds
                self.opti.subject_to(uk[0] >= 0.0)
                self.opti.subject_to(uk[0] <= 3e5)
                self.opti.subject_to(uk[1] >= 0.3)
                self.opti.subject_to(uk[1] <= 2.5)
                self.opti.subject_to(uk[2] >= -60.0)
                self.opti.subject_to(uk[2] <=  60.0)

            # state bounds in all time steps t
            self.opti.subject_to(self.X[si+2, :] >= 0.0)
            self.opti.subject_to(self.X[si+3, :] >= 70.0)
            self.opti.subject_to(self.X[si+5, :] >= -0.70)
            self.opti.subject_to(self.X[si+5, :] <=  0.35)

        # separations
        check_ks = list(range(0, N, self.sep_stride))

        for k in check_ks:
            for i in range(M):
                for j in range(i + 1, M):
                    si = i * 6; sj = j * 6
                    dx = self.X[si,   k] - self.X[sj,   k]
                    dy = self.X[si+1, k] - self.X[sj+1, k]
                    dz = self.X[si+2, k] - self.X[sj+2, k]
                    self.opti.subject_to(
                        (dx**2 + dy**2) / self.D_h2 + dz**2 / self.D_v2 >= 1.0
                    )

        TOL_XY = 5.0  # m   lateral / longitudinal
        TOL_Z = 5.0  # m   altitude
        TOL_CHI = 3.0 * ca.pi / 180.0  # rad heading  (±3°)
        TOL_GAM = 1.0 * ca.pi / 180.0  # rad glideslope (±1°)
        GAM_REF = -3.0 * ca.pi / 180.0  # −3° glide path

        for i in range(M):
            si = i * 6;
            ci = i * 3
            gx = self.goal_param[i * 4]
            gy = self.goal_param[i * 4 + 1]
            gz = self.goal_param[i * 4 + 2]
            gv = self.goal_param[i * 4 + 3]
            chi_goal = self.heading_param[i]

            # Position
            self.opti.subject_to(self.X[si, N] >= gx - TOL_XY)
            self.opti.subject_to(self.X[si, N] <= gx + TOL_XY)
            self.opti.subject_to(self.X[si + 1, N] >= gy - TOL_XY)
            self.opti.subject_to(self.X[si + 1, N] <= gy + TOL_XY)
            self.opti.subject_to(self.X[si + 2, N] >= gz - TOL_Z)
            self.opti.subject_to(self.X[si + 2, N] <= gz + TOL_Z)

            # chi
            e_chi = ca.atan2(
                ca.sin(self.X[si + 4, N] - chi_goal),
                ca.cos(self.X[si + 4, N] - chi_goal),
            )
            self.opti.subject_to(e_chi >= -TOL_CHI)
            self.opti.subject_to(e_chi <= TOL_CHI)

            # gamma
            e_gam = self.X[si + 5, N] - GAM_REF
            self.opti.subject_to(e_gam >= -TOL_GAM)
            self.opti.subject_to(e_gam <= TOL_GAM)

            # soft -> velocity
            e_v = (self.X[si + 3, N] - gv) / 20.0
            J += 0.5 * e_v ** 2

            # Smoothness
            smooth = ca.MX(0.0)
            for k in range(N - 1):
                smooth += (
                        ((self.U[ci, k + 1] - self.U[ci, k]) / 30000.0) ** 2
                        + ((self.U[ci + 1, k + 1] - self.U[ci + 1, k]) / 0.3) ** 2
                        + ((self.U[ci + 2, k + 1] - self.U[ci + 2, k]) / 10.0) ** 2
                )
            J += 0.1 * smooth / (N - 1)

        self.opti.minimize(J)
        self.opti.solver("ipopt", {}, {
            "max_iter":         2000,
            "tol":              1e-4,
            "acceptable_tol":   1e-3,
            "acceptable_iter":    15,
            "print_level":         3,
            "linear_solver":  "mumps",
            "mu_strategy": "adaptive",
            "bound_relax_factor": 1e-8,
        })

    def solve(
        self,
        x0_list: list[np.ndarray],
        goal_list: list[np.ndarray],
        heading_goals_deg: list[float],
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """
          X_val  : (6*M, N+1)
          U_val  : (3*M, N)
          T_vals : list[float]  – for individual flight time per aircraft
        """
        self.opti.set_value(self.x0_param,      np.concatenate(x0_list))
        self.opti.set_value(self.goal_param,     np.concatenate(goal_list))
        self.opti.set_value(self.heading_param,  np.radians(heading_goals_deg))

        T_guesses = []
        for i in range(self.M):
            dist  = np.linalg.norm(goal_list[i][:3] - x0_list[i][:3])
            v_avg = 0.5 * (float(x0_list[i][3]) + max(float(goal_list[i][3]), 30.0))
            T_g   = float(np.clip(dist / v_avg * 1.5, self.T_min, self.T_max))
            T_guesses.append(T_g)

        for Ti, T_g in zip(self.T_list, T_guesses):
            self.opti.set_initial(Ti, T_g)

        for i in range(self.M):
            si = i * 6; ci = i * 3
            p0 = x0_list[i][:3]
            pG = goal_list[i][:3]
            d3 = pG - p0
            d_xy = d3[:2]
            d_norm = np.linalg.norm(d_xy)

            lat_off = np.zeros(3)
            if d_norm > 1.0 and i > 0:
                perp = np.array([-d_xy[1], d_xy[0]]) / d_norm
                sign = 1 if i % 2 == 1 else -1
                lat_off[:2] = sign * perp * np.sqrt(self.D_h2) * 1.5

            v0    = float(x0_list[i][3])
            vG    = max(float(goal_list[i][3]), 30.0)
            chi0  = float(x0_list[i][4])
            gam0  = float(x0_list[i][5])
            chiG  = float(np.radians(heading_goals_deg[i]))
            d_chi = ((chiG - chi0) + np.pi) % (2*np.pi) - np.pi
            gamG  = float(np.arctan2(d3[2], np.linalg.norm(d3[:2]) + 1e-6))

            for k in range(self.N + 1):
                tau  = k / self.N
                base = p0 + tau * d3
                fade = 1.0 - tau
                self.opti.set_initial(self.X[si:si+6, k], np.array([
                    *(base + lat_off * fade),
                    max(v0 + tau * (vG - v0), 30.0),
                    chi0 + tau * d_chi,
                    gam0 + tau * (gamG - gam0),
                ]))
            for k in range(self.N):
                self.opti.set_initial(self.U[ci:ci+3, k], [120_000.0, 1.0, 0.0])

        try:
            sol   = self.opti.solve()
            X_val = sol.value(self.X)
            U_val = sol.value(self.U)
            T_vals = [float(sol.value(Ti)) for Ti in self.T_list]
            for i, t in enumerate(T_vals):
                print(f"[MultiOptimizer] AC{i} IPOPT converged  T = {t:.1f} s")
        except RuntimeError as e:
            print(f"[MultiOptimizer] IPOPT failed: {e}\n ->  best-so-far")
            X_val  = self.opti.debug.value(self.X)
            U_val  = self.opti.debug.value(self.U)
            T_vals = [float(self.opti.debug.value(Ti)) for Ti in self.T_list]

        return X_val, U_val, T_vals