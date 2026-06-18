import numpy as np
import casadi as ca
from casadi import MX, Opti, DM

from physics.aircraft_environment_casadi import AircraftEnvironmentCasadi
from physics.aircraft_dynamics import AircraftDynamics
from physics.aircraft_kinematics_state_casadi import AircraftKinematicsStateCasadi


def state_to_casadi(s: AircraftKinematicsStateCasadi) -> np.ndarray:
    return np.array([
        s.x,
        s.y,
        s.h,
        s.v,
        np.radians(s.chi),
        np.radians(s.gamma),
    ])

def compute_glideslope(via_point: np.ndarray, touchdown: np.ndarray) -> float:
    dh  = touchdown[2] - via_point[2]
    dxy = np.linalg.norm(touchdown[:2] - via_point[:2])
    gam = float(np.arctan2(dh, max(dxy, 1.0)))
    print(f"[compute_glideslope] Δh={dh:.1f} m  Δxy={dxy:.1f} m  γ_ref={np.degrees(gam):.2f}°")
    return gam


class LandingTrajectoryRnavOptimizer:
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
    8. Dynamic via-point constraint (including heading and glide path alignment within a window)
    9. Directional stability (holding heading and glideslope) AFTER passing the via-point
    """

    def __init__(
        self,
        state:           AircraftKinematicsStateCasadi,
        min_height:      float,
        N:               int   = 80,
        T_min:           float = 60.0,
        T_max:           float = 1800.0,
        k_via_frac_start: float = 0.50,
        k_via_frac_end: float = 0.85,
    ) -> None:
        self.N: int = N
        self.T_min: float = T_min
        self.T_max: float = T_max
        self.state: AircraftKinematicsStateCasadi = state

        self.k_start: int = int(N * k_via_frac_start)
        self.k_end: int = int(N * k_via_frac_end)

        self.opti: Opti = ca.Opti()
        self.X: MX = self.opti.variable(6, N + 1)    # [x, y, z, v, chi, gamma]
        self.U: MX = self.opti.variable(3, N)        # [thrust, load_factor, roll_deg]
        self.T: MX = self.opti.variable()                  # free solving time t [s]

        self.x0_param: MX = self.opti.parameter(6)   # start state
        self.goal_param: MX = self.opti.parameter(4)   # [x, y, z, v_td]
        self.heading_param: MX = self.opti.parameter(1)   # χ_goal [rad]
        self.via_param: MX = self.opti.parameter(3)  # [x, y, z] via-point
        self.min_height_param: MX = self.opti.parameter() # min height
        self.gam_ref_param:    MX = self.opti.parameter()   # γ_ref [rad]

        self.min_height: float = min_height

        self._build()

    def _f(
            self,
            xk: MX,
            uk: MX,
            dt: MX,
    ) -> MX:
        temp_state = AircraftKinematicsStateCasadi(
            x     = xk[0],
            y     = xk[1],
            h     = xk[2],
            v     = xk[3],
            chi   = xk[4] * (180.0 / ca.pi),
            gamma = xk[5] * (180.0 / ca.pi),
            rho0  = self.state.rho0,
            aircraft_type=self.state.aircraft_type,
        )

        temp_env = AircraftEnvironmentCasadi(
            state       = temp_state,
            goal        = None,
            dt          = dt,
        )

        s = temp_env.apply_action(
            thrust      = uk[0],
            load_factor = uk[1],
            roll        = uk[2],
        )

        return ca.vertcat(
            s[0],
            s[1],
            s[2],
            s[3],
            s[5] * (ca.pi / 180.0),
            s[4] * (ca.pi / 180.0),
        )

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

        self.opti.subject_to(self.U[0, :] >= 0.0)                                         # thrust ≥ 0
        self.opti.subject_to(self.U[0, :] <= self.state.aircraft_type.max_thrust)         # thrust
        self.opti.subject_to(self.U[1, :] >= self.state.aircraft_type.min_load_factor)    # n
        self.opti.subject_to(self.U[1, :] <= self.state.aircraft_type.max_load_factor)    # n
        self.opti.subject_to(self.U[2, :] >= -self.state.aircraft_type.max_roll_angle)    # roll
        self.opti.subject_to(self.U[2, :] <=  self.state.aircraft_type.max_roll_angle)    # roll

        self.opti.subject_to(self.X[2, :] >= self.min_height_param)    # terrain
        self.opti.subject_to(self.X[3, :] >= self.state.stall_speed(self.min_height) * 1.2) # v stall speed (no Stall)
        # this is set to prevent anything strange -> e.g. numerical overflow etc. and max load factor is for normal flight
        self.opti.subject_to(self.X[5, :] >= -0.70) # γ ≥ −40° delta, tied: 0.35, 0.70 and 0.2 is really smooth, but can get in the way when the load factor is too small
        self.opti.subject_to(self.X[5, :] <=  0.35) # γ ≤ 20° delta

        gx = self.goal_param[0]
        gy = self.goal_param[1]
        gz = self.goal_param[2]
        gv = self.goal_param[3]
        chi_goal = self.heading_param[0]

        TOL_XY = 5.0  # m lateral / longitudinal
        TOL_Z = 5.0  # m altitude at threshold
        TOL_CHI = 3.0 * ca.pi / 180.0  # rad heading (±3°)
        TOL_GAM = 1.0 * ca.pi / 180.0  # rad glideslope (±1°)
        GAM_REF = self.gam_ref_param  # Soll-Gleitpfad

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

        e_gam = self.X[5, N] - GAM_REF
        self.opti.subject_to(e_gam >= -TOL_GAM)
        self.opti.subject_to(e_gam <= TOL_GAM)

        e_v = (self.X[3, N] - gv) / 20.0
        J = 0.5 * e_v ** 2
        # time penalty -> leave out for faster convergence
        # J += 0.05 * (self.T / self.T_max)

        smooth = ca.MX(0.0)
        for k in range(N - 1):
            smooth += (
                ((self.U[0, k+1] - self.U[0, k]) / 30000.0) ** 2
              + ((self.U[1, k+1] - self.U[1, k]) /    0.3 ) ** 2
              + ((self.U[2, k+1] - self.U[2, k]) /   10.0 ) ** 2
            )
        J += 0.1 * smooth / (N - 1)

        vx, vy, vz = self.via_param[0], self.via_param[1], self.via_param[2]

        via_costs = []
        for k in range(self.k_start, self.k_end):
            dx = (self.X[0, k] - vx) / 50.0
            dy = (self.X[1, k] - vy) / 50.0
            dz = (self.X[2, k] - vz) / 10.0
            pos_error_sq = dx ** 2 + dy ** 2 + dz ** 2

            e_chi_k = ca.atan2(
                ca.sin(self.X[4, k] - chi_goal),
                ca.cos(self.X[4, k] - chi_goal)
            )
            chi_error_sq = (e_chi_k / (3.0 * ca.pi / 180.0)) ** 2

            e_gam_k = self.X[5, k] - GAM_REF
            gam_error_sq = (e_gam_k / (1.0 * ca.pi / 180.0)) ** 2

            total_k_cost = pos_error_sq + 0.5 * chi_error_sq + 0.5 * gam_error_sq
            via_costs.append(total_k_cost)

        J_via_best = via_costs[0]
        for cost in via_costs[1:]:
            J_via_best = ca.fmin(J_via_best, cost)

        tracking_penalty = ca.MX(0.0)
        for k in range(self.k_start, N):
            e_chi_track = ca.atan2(ca.sin(self.X[4, k] - chi_goal), ca.cos(self.X[4, k] - chi_goal))
            e_gam_track = self.X[5, k] - GAM_REF

            error_sq = (e_chi_track / (3.0 * ca.pi / 180.0)) ** 2 + (e_gam_track / (1.0 * ca.pi / 180.0)) ** 2
            tracking_penalty += error_sq

        J += 8.0 * J_via_best + 0.5 * tracking_penalty

        self.opti.minimize(J)

        self.opti.solver("ipopt", {}, {
            "max_iter":           3000,
            "tol":                1e-6,      # 1e-4
            "acceptable_tol":     1e-4,      # 1e-3
            "acceptable_iter":      5,       # 15
            "print_level":           3,
            "linear_solver":    "mumps",
            "mu_strategy":   "adaptive",
            "bound_relax_factor": 1e-9,     # 1e-8
        })

    def solve(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        heading_goal_deg: float,
        min_height: float,
        via_point: np.ndarray,
        x_init: np.ndarray | None = None,
        u_init: np.ndarray | None = None,
        T_init: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        X_val : ndarray (6, N+1)  — optimized state trajectory
        U_val : ndarray (3, N)    — optimized control sequence
        T_val : float             — optimized landing time [s]
        """
        self.opti.set_value(self.x0_param, x0)
        self.opti.set_value(self.goal_param, goal)
        self.opti.set_value(self.heading_param, float(np.radians(heading_goal_deg)))
        self.opti.set_value(self.min_height_param, min_height)

        if via_point is not None:
            self.opti.set_value(self.via_param, via_point)

            gam_ref = compute_glideslope(via_point, goal[:3])
            self.opti.set_value(self.gam_ref_param, gam_ref)
        else:
            gam_ref = -3.0 * np.pi / 180.0
            self.opti.set_value(self.gam_ref_param, gam_ref)

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
            d_chi    = ((chi_goal - chi0) + np.pi) % (2 * np.pi) - np.pi

            via_init = via_point

            k_mid = int((self.k_start + self.k_end) / 2)
            tau_via = k_mid / self.N

            for k in range(self.N + 1):
                tau = k / self.N

                if tau <= tau_via:
                    t_seg = tau / tau_via if tau_via > 0 else 0.0
                    p = pos0 + t_seg * (via_init - pos0)
                    gam_k = gam0 + (tau / tau_via) * (0.0 - gam0)
                else:
                    t_seg = (tau - tau_via) / (1.0 - tau_via)
                    p     = via_init + t_seg * (pos_goal - via_init)
                    gam_k = 0.0 + t_seg * (gam_ref - 0.0)

                self.opti.set_initial(self.X[:, k], np.array([
                    *p,
                    max(v0 + tau * (v_goal - v0), 30.0),
                    chi0 + tau * d_chi,
                    gam_k,
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