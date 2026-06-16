import numpy as np
from numpy.typing import NDArray

from physics.aircraft_goal import AircraftGoal
from physics.aircraft_kinematics_state import AircraftKinematicsState


class AircraftEnvironment:
    """
    Point-mass aircraft kinematics (no sideslip / coordinated flight).

    Force balance (Newton, vector form):
        F = m · a = m · v⃗ = (T − D) · t̂ + L · n̂ + m · g⃗

        v̇ = ((T - D) / m) · (v⃗ / ‖v⃗‖) + (L / m) · n⃗ + g⃗

        T   — thrust                     [N]
        D   — drag                       [N]
        L   — lift                       [N]
        m   — aircraft mass              [kg]
        g⃗  = [0, 0, −g]ᵀ   gravity     [m/s²]  (downward)

    Aerodynamic model:
        D = 1/2 · ρ · v² · C_D · A

        L = 1/2 · ρ · v² · C_L · A

        ρ    — air density               [kg/m³]
        v    — aircraft speed            [m/s]
        C_D  — drag coefficient          [-]
        C_L  — lift coefficient          [-]
        A    — reference / wing area     [m²]

    Unit vectors n̂
        t⃗ or F = v⃗ / ‖v⃗‖                        unit tangent
        u⃗      = [0, 0, 1]ᵀ                     vertical unit vector
        b⃗      = (t⃗ × u⃗) / ‖t⃗ × u⃗ ‖             unit binormal  (wing direction)
        n⃗      = b⃗ × t⃗                          unit normal    (lift direction)

    Load factor:
        n  = L / (m · g)
    """

    def __init__(
            self,
            state:       AircraftKinematicsState,
            goal:        AircraftGoal,
            dt:          float,
    ) -> None:
        self.state:       AircraftKinematicsState          = state
        self.goal:        AircraftGoal                     = goal
        self.dt:          float                            = dt
        self.gravity:     NDArray[np.float64]              = np.array([0.0, 0.0, -9.81])

    def unit_vectors_with_roll(
            self,
            roll: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        t⃗   =  v⃗ / ‖v⃗‖                        unit tangent
        u⃗   =  [0, 0, 1]ᵀ                     vertical unit vector
        b⃗   =  (t⃗  × u⃗ ) / ‖t⃗  × u⃗ ‖          unit binormal  (wing direction)
        n⃗   =  b⃗  × t⃗                         unit normal    (lift direction)

        Note:  L⃗ ⊥ v⃗  →   lift is always perpendicular to velocity, L⃗ ⊥ φ (roll)

        Roll φ:
            rotation of lift vector n⃗ around t⃗

        n⃗_rot = n⃗ cos(φ) + (t⃗ × n⃗) · sin(φ) + t⃗ (t⃗ · n⃗)(1 - cos(φ))

        Returns  (t⃗ , b⃗ , n⃗ )
        """
        # t⃗   =  v⃗ / ‖v⃗‖
        v_vec: NDArray[np.float64] = self.state.velocity_vector()
        t_hat = v_vec / np.linalg.norm(v_vec)

        # u⃗   =  [0, 0, 1]ᵀ
        u_hat = np.array([0.0, 0.0, 1.0])

        # b⃗   =  (t⃗  × u⃗ ) / ‖t⃗  × u⃗ ‖
        b_hat = np.cross(t_hat, u_hat)
        b_hat = b_hat / np.linalg.norm(b_hat)

        # n⃗   =  b⃗  × t⃗
        n_hat = np.cross(b_hat, t_hat)

        # φ = roll angle
        phi = np.radians(roll)

        # n⃗_rot = n⃗ cos(φ) + (t⃗ × n⃗) · sin(φ) + t⃗ (t⃗ · n⃗)(1 - cos(φ))
        n_hat_rot = (
                n_hat * np.cos(phi)
                + np.cross(t_hat, n_hat) * np.sin(phi)
                + t_hat * np.dot(t_hat, n_hat) * (1 - np.cos(phi))
        )

        return t_hat, b_hat, n_hat_rot


    def apply_action(
            self,
            thrust: float,
            load_factor: float,
            roll: float,
    ) -> NDArray[np.float64]:
        """
        Velocity dynamics:
            v̇ = ((T - D) / m) · (v⃗ / ‖v⃗‖) + (L / m) · n⃗ + g⃗

        Position update:
            ẋ = v_x
            ẏ = v_y
            ḣ = v_z

        State update (Euler step):
            v⃗_{k+1} = v⃗_k + v̇ · dt
            p_{k+1} = p_k + v⃗_{k+1} · dt

        Flight path angle update:
            Δγ = γ_cmd - γ
            γ_{k+1} = γ_k + Δγ

        State reconstruction from velocity:
            v = ||v⃗||
            γ = arcsin(v_z / v)
            𝒳 = arctan2(v_x, v_y)

        Heading dynamics:
            𝒳̇ = (g · tan(φ)) / v
            𝒳̇ _{k+1} = 𝒳_k + 𝒳̇ · dt

        Forces:
            D = 1/2 · ρ · v² · C_D · A
            L = 1/2 · ρ · v² · C_L · A

        Unit vectors:
            t⃗ = v⃗ / ||v⃗||
            n⃗ = lift-normal direction (perpendicular to velocity in vertical plane)
            n⃗ = L / (m · g)
        """

        # v⃗
        v_vec = self.state.velocity_vector()

        # t⃗ , _, n⃗
        t_hat, _, n_hat = self.unit_vectors_with_roll(roll)

        # C_L = (2·n·m·g)  / (ρ · v² · S)
        requested_cl = self.state.required_lift_coefficient(
            load_factor=load_factor
        )

        actual_cl = min(
            requested_cl,
            self.state.aircraft_type.cl_max
        )

        # D = 1/2 · ρ · v² · C_D · A
        drag = self.state.drag_force(actual_cl)
        # L = n · m · g
        # lift is included in the load factor now
        # L = 1/2 · ρ · v² · C_L · A
        actual_lift = self.state.lift_force(actual_cl)

        # v̇ = ((T - D) / m) · (v⃗ / ‖v⃗‖) + (L / m) · n⃗ + g⃗
        v_dot = (
                ((thrust - drag) / self.state.aircraft_type.mass) * t_hat
                + (actual_lift / self.state.aircraft_type.mass) * n_hat # old code: + (lift / self.state.mass) * n_hat
                + self.gravity
        )

        # v⃗_{k+1} = v⃗_k + v̇ · dt
        v_vec = v_vec + v_dot * self.dt

        # p_{k+1} = p_k + v⃗_{k+1} · dt
        self.state.x += v_vec[0] * self.dt
        self.state.y += v_vec[1] * self.dt
        self.state.h += v_vec[2] * self.dt

        # v = ||v⃗||
        self.state.v = np.linalg.norm(v_vec)

        # γ̇ = (v * v̇_z - v_z * v̇) / (v² * cos(γ))
        # v = ||v⃗||
        # v̇ = (v⃗ · v̇⃗) / ||v⃗||   (scalar speed derivative)
        v_dot_scalar = np.dot(v_vec, v_dot) / max(self.state.v, 1e-6)

        # γ̇ = (v * v̇_z - v_z * v̇) / (v² * cos(γ))
        gamma_dot = (
                (self.state.v * v_dot[2] - v_vec[2] * v_dot_scalar)
                / max(self.state.v ** 2 * np.cos(np.radians(self.state.gamma)), 1e-6)
        )
        # γ_{k+1} = γ_k + γ̇ · dt
        self.state.gamma += np.degrees(gamma_dot * self.dt)

        # 𝒳̇ = (g · tan(φ)) / v
        chi_dot = (9.81 * np.tan(np.radians(roll))) / max(self.state.v, 1e-6)
        # 𝒳̇ _{k+1} = 𝒳_k + 𝒳̇ · dt
        self.state.chi += np.degrees(chi_dot * self.dt)

        return self.state.to_array()

    def autopilot(self) -> tuple[float, float, float]:
        p0 = np.array([self.state.x, self.state.y, self.state.h])
        v_vec = self.state.velocity_vector()

        # ---- Regler system -> v_dot = np.array([goal.x, goal.y, goal.h]) - p0
        pos_error = np.array([self.goal.x, self.goal.y, self.goal.h]) - p0
        dist = np.linalg.norm(pos_error)

        if dist > 1e-6:
            direction = pos_error / dist
        else:
            direction = np.zeros(3)

        K = np.clip(dist * 0.0001, 0.2, 2.0)
        v_desired = direction * min(dist * 0.5, 250.0)
        v_dot = K * (v_desired - v_vec)
        # ----

        desired_chi = np.arctan2(
            self.goal.x - self.state.x,
            self.goal.y - self.state.y
        )

        vx, vy, _ = v_vec
        current_chi = np.arctan2(vx, vy)

        chi_dot = desired_chi - current_chi

        roll = np.degrees(
            np.arctan((self.state.v * chi_dot) / 9.81)
        )

        # this can be done, but we can also input for roll and load_factor = 0.0
        t_hat, _, n_hat = self.unit_vectors_with_roll(roll=roll)
        g_vec = self.gravity

        t_hat, _, n_hat = self.unit_vectors_with_roll(roll)
        load_factor = np.dot(
            (v_dot - self.gravity),
            n_hat
        ) / 9.81

        requested_cl = self.state.required_lift_coefficient(load_factor=load_factor)
        actual_cl = min(requested_cl, self.state.aircraft_type.cl_max)

        drag = self.state.drag_force(actual_cl)

        # v̇ = ((T - D) / m) · (v⃗ / ‖v⃗‖) + (L / m) · n⃗ + g⃗
        # T = m ( v̇ · t̂ ) + D
        thrust = self.state.aircraft_type.mass * np.dot(v_dot, t_hat) + drag

        # n = ((v̇ - g) · n̂) / g
        load_factor = np.dot((v_dot - g_vec), n_hat) / 9.81

        # 𝒳̇ = (g · tan(φ)) / v
        desired_chi = np.arctan2(self.goal.x - self.state.x, self.goal.y - self.state.y)

        vx, vy, _ = v_vec
        current_chi = np.arctan2(vx, vy)

        chi_dot = (desired_chi - current_chi)

        # φ = arctan( v · 𝒳̇ / g )
        roll = np.arctan((self.state.v * chi_dot) / 9.81)
        roll = np.degrees(roll)

        return thrust, load_factor, roll
