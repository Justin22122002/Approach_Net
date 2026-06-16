import casadi as ca

from physics.aircraft_goal import AircraftGoal
from physics.aircraft_kinematics_state_casadi import AircraftKinematicsStateCasadi


def _cross(a: ca.MX, b: ca.MX) -> ca.MX:
    """CasADi cross product of two 3-vectors (manually implemented)."""
    return ca.vertcat(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


class AircraftEnvironmentCasadi:
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
            state:       AircraftKinematicsStateCasadi,
            goal:        AircraftGoal,
            dt:          float,
    ) -> None:
        self.state:       AircraftKinematicsStateCasadi    = state
        self.goal:        AircraftGoal                     = goal
        self.dt:          float                            = dt
        self.gravity:     ca.MX                           = ca.vertcat(0.0, 0.0, -9.81)

    def unit_vectors_with_roll(
            self,
            roll: ca.MX
    ) -> tuple[ca.MX, ca.MX, ca.MX]:
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
        v_vec: ca.MX = self.state.velocity_vector()
        t_hat = v_vec / ca.norm_2(v_vec)

        # u⃗   =  [0, 0, 1]ᵀ
        u_hat = ca.vertcat(0.0, 0.0, 1.0)

        # b⃗   =  (t⃗  × u⃗ ) / ‖t⃗  × u⃗ ‖
        b_hat = _cross(t_hat, u_hat)
        b_hat = b_hat / ca.norm_2(b_hat)

        # n⃗   =  b⃗  × t⃗
        n_hat = _cross(b_hat, t_hat)

        # φ = roll angle
        phi = roll * (ca.pi / 180.0)

        # n⃗_rot = n⃗ cos(φ) + (t⃗ × n⃗) · sin(φ) + t⃗ (t⃗ · n⃗)(1 - cos(φ))
        n_hat_rot = (
                n_hat * ca.cos(phi)
                + _cross(t_hat, n_hat) * ca.sin(phi)
                + t_hat * ca.dot(t_hat, n_hat) * (1 - ca.cos(phi))
        )

        return t_hat, b_hat, n_hat_rot

    # todo: only update the state at the end of the function to mitigate side effects
    def apply_action(
            self,
            thrust: ca.MX,
            load_factor: ca.MX,
            roll: ca.MX,
    ) -> ca.MX:
        """
        Velocity dynamics:
            v̇ = ((T - D) / m) · (v⃗ / ‖v⃗‖) + (L / m) · n⃗ + g⃗

        Position update:
            ẋ = v_x
            ẏ = v_y
            ḣ = v_z

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

        actual_cl = ca.fmin(requested_cl, self.state.aircraft_type.cl_max)

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
        self.state.v = ca.norm_2(v_vec)

        # γ̇ = (v * v̇_z - v_z * v̇) / (v² * cos(γ))
        # v = ||v⃗||
        # v̇ = (v⃗ · v̇⃗) / ||v⃗||   (scalar speed derivative)
        v_dot_scalar = ca.dot(v_vec, v_dot) / ca.fmax(self.state.v, 1e-6)

        # γ̇ = (v * v̇_z - v_z * v̇) / (v² * cos(γ))
        gamma_dot = (
                (self.state.v * v_dot[2] - v_vec[2] * v_dot_scalar)
                / ca.fmax(self.state.v ** 2 * ca.cos(self.state.gamma * (ca.pi / 180.0)), 1e-6)
        )
        # γ_{k+1} = γ_k + γ̇ · dt
        self.state.gamma += (gamma_dot * self.dt) * (180.0 / ca.pi)

        # 𝒳̇ = (g · tan(φ)) / v
        chi_dot = (9.81 * ca.tan(roll * (ca.pi / 180.0))) / ca.fmax(self.state.v, 1e-6)
        # 𝒳̇ _{k+1} = 𝒳_k + 𝒳̇ · dt
        self.state.chi += (chi_dot * self.dt) * (180.0 / ca.pi)

        return self.state.to_array()

    def autopilot(self) -> tuple[ca.MX, ca.MX, ca.MX]:
        p0 = ca.vertcat(self.state.x, self.state.y, self.state.h)
        v_vec = self.state.velocity_vector()

        # ---- Regler system -> v_dot = np.array([goal.x, goal.y, goal.h]) - p0
        pos_error = ca.vertcat(self.goal.x, self.goal.y, self.goal.h) - p0
        dist = ca.norm_2(pos_error)

        direction = pos_error / ca.fmax(dist, 1e-6)

        K = ca.fmin(ca.fmax(dist * 0.0001, 0.2), 2.0)
        v_desired = direction * ca.fmin(dist * 0.5, 250.0)
        v_dot = K * (v_desired - v_vec)
        # ----

        desired_chi = ca.atan2(
            self.goal.x - self.state.x,
            self.goal.y - self.state.y
        )

        current_chi = ca.atan2(v_vec[0], v_vec[1])

        chi_dot = desired_chi - current_chi

        roll = ca.atan((self.state.v * chi_dot) / 9.81) * (180.0 / ca.pi)

        # this can be done, but we can also input for roll and load_factor = 0.0
        t_hat, _, n_hat = self.unit_vectors_with_roll(roll=roll)
        g_vec = self.gravity

        t_hat, _, n_hat = self.unit_vectors_with_roll(roll)
        load_factor = ca.dot(
            (v_dot - self.gravity),
            n_hat
        ) / 9.81

        requested_cl = self.state.required_lift_coefficient(load_factor=load_factor)
        actual_cl = ca.fmin(requested_cl, self.state.aircraft_type.cl_max)

        drag = self.state.drag_force(actual_cl)

        # v̇ = ((T - D) / m) · (v⃗ / ‖v⃗‖) + (L / m) · n⃗ + g⃗
        # T = m ( v̇ · t̂ ) + D
        thrust = self.state.aircraft_type.mass * ca.dot(v_dot, t_hat) + drag

        # n = ((v̇ - g) · n̂) / g
        load_factor = ca.dot((v_dot - g_vec), n_hat) / 9.81

        # 𝒳̇ = (g · tan(φ)) / v
        desired_chi = ca.atan2(self.goal.x - self.state.x, self.goal.y - self.state.y)

        current_chi = ca.atan2(v_vec[0], v_vec[1])

        chi_dot = (desired_chi - current_chi)

        # φ = arctan( v · 𝒳̇ / g )
        roll = ca.atan((self.state.v * chi_dot) / 9.81) * (180.0 / ca.pi)

        return thrust, load_factor, roll

def main():
    import numpy as np
    from physics.aircraft_kinematics_state_casadi import AircraftKinematicsStateCasadi
    from physics.aircraft_goal import AircraftGoal
    from physics.aircraft_type import AircraftType

    constraints = AircraftType(
        cl_max=1.5,
        mass=60000.0,
        C_D0=0.02,
        area=122.0,
    )

    state = AircraftKinematicsStateCasadi(
        x=0.0,
        y=0.0,
        h=1000.0,
        v=120.0,
        gamma=0.0,
        chi=0.0,
        aircraft_type=constraints,
    )

    goal = AircraftGoal(
        x=5000.0,
        y=3000.0,
        h=500.0,
    )

    dt = ca.MX.sym("dt")

    env = AircraftEnvironmentCasadi(
        state=state,
        goal=goal,
        constraints=constraints,
        dt=dt
    )

    thrust = ca.MX.sym("thrust")
    load_factor = ca.MX.sym("n")
    roll = ca.MX.sym("roll")

    next_state = env.apply_action(thrust, load_factor, roll)

    p = next_state[:3]
    cost = ca.sumsqr(p - ca.vertcat(goal.x, goal.y, goal.h))

    x = ca.vertcat(thrust, load_factor, roll, dt)

    g = ca.gradient(cost, x)
    # H = ca.hessian(cost, x)[0]

    f_grad = ca.Function("grad", [x], [g])
    f_cost = ca.Function("cost", [x], [cost])

    x0 = [100000, 1.2, 10, 3.75]

    print("Cost:", f_cost(x0))
    print("Gradient:", f_grad(x0))

    # numeric sanity check
    print("\nJacobian norm:", np.linalg.norm(np.array(f_grad(x0)).flatten()))

    grad_val = np.array(f_grad(x0)).flatten()

    print("Cost:              ", f_cost(x0))
    print("∂J/∂thrust:        ", grad_val[0])
    print("∂J/∂load_factor:   ", grad_val[1])
    print("∂J/∂roll:          ", grad_val[2])
    print("∂J/∂dt:            ", grad_val[3])

    print("\n∂J/∂T (bei N=80):  ", grad_val[3] / 80)


if __name__ == "__main__":
    main()