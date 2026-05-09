from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt

class AircraftKinematicsMaxConstrains:
    """
    those are the maximum values for the aircraft kinematics, these are not dependent
    """
    def __init__(
            self,
            v_min: float,
            v_max: float,
            roc_max: float,
            rod_max: float,
            gamma_dot_max: float,
            chi_dot_max: float,
    ) -> None:
        self.v_min: float = v_min                   # v_min        minimum speed (stall)            [m/s]
        self.v_max: float = v_max                   # v_max        maximum speed                    [m/s]
        self.roc_max: float = roc_max               # ROC_max      max rate of climb   ḣ ≤  ROC_max [m/s]
        self.rod_max: float = rod_max               # ROD_max      max rate of descent ḣ ≥ -ROD_max [m/s]
        self.gamma_dot_max: float = gamma_dot_max   # γ_max        max |Δγ/Δt|                      [°/s]
        self.chi_dot_max: float = chi_dot_max       # χ_max        max |Δχ/Δt|                      [°/s]

class AircraftGoal:
    def __init__(self, x: float, y: float, h: float) -> None:
        self.x: float = x  # East  [m]
        self.y: float = y  # North [m]
        self.h: float = h  # Altitude [m]

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.h])

class AircraftKinematicsState:
    def __init__(
            self,
            x: float,
            y: float,
            h: float,
            v: float,
            gamma: float,
            chi: float,
            mass: float,
            C_D: float,
            C_L: float,
            area: float,
            rho: float = 1.225
    ) -> None:
        self.x: float = x         # m or lon
        self.y: float = y         # m or lat
        self.h: float = h         # m or feet
        self.v: float = v         # m/s
        self.gamma: float = gamma # γ / θ, flight path angle (Flugbahnwinkel)
        self.chi: float = chi     # φ / χ, Track angle / Course angle (Kursbahnwinkel) from North, clockwise

        self.mass: float = mass      # [kg]
        self.C_D: float = C_D        # drag coefficient
        self.C_L: float = C_L        # lift coefficient
        self.area: float = area      # wing reference area [m²]
        self.rho: float = rho        # air density [kg/m³]

    def velocity_vector(self) -> NDArray[np.float64]:
        """
        3-D velocity vector in (East, North, Up):

            ⎡ v·cos(γ)·sin(χ) ⎤   ⎡ ẋ ⎤
        v⃗ = ⎢ v·cos(γ)·cos(χ) ⎥ = ⎢ ẏ ⎥
            ⎣ v·sin(γ)        ⎦   ⎣ ḣ ⎦

        γ = flight path angle
        χ = track angle from True North, clockwise
        """
        return np.array([
            self.v * np.cos(np.radians(self.gamma)) * np.sin(np.radians(self.chi)),
            self.v * np.cos(np.radians(self.gamma)) * np.cos(np.radians(self.chi)),
            self.v * np.sin(np.radians(self.gamma)),
        ])

    def drag_force(self) -> float:
        """
        D = 1/2 · ρ · v² · C_D · A
        """
        return 0.5 * self.rho * self.v ** 2 * self.C_D * self.area

    def lift_force(self) -> float:
        """
        L = 1/2 · ρ · v² · C_L · A
        """
        base_lift = 0.5 * self.rho * self.v ** 2 * self.C_L * self.area
        return base_lift

    def to_array(self) -> NDArray[np.float64]:
        return np.array([
            self.x,
            self.y,
            self.h,
            self.v,
            self.gamma,
            self.chi,
            self.mass,
            self.C_D,
            self.C_L,
            self.area
        ])

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "AircraftKinematicsState":
        return cls(
            x=arr[0],
            y=arr[1],
            h=arr[2],
            v=arr[3],
            gamma=arr[4],
            chi=arr[5],
            mass=arr[6],
            C_D=arr[7],
            C_L=arr[8],
            area=arr[9],
        )


class AircraftEnvironment:
    """
    Point-mass aircraft kinematics (no sideslip / coordinated flight).

    Force balance (Newton, vector form):
        F = m · a = m · v⃗ =  (T − D) · t̂  +  L · n̂  +  m · g⃗

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
        t⃗ or F =  v⃗ / ‖v⃗‖                        unit tangent
        u⃗      =  [0, 0, 1]ᵀ                     vertical unit vector
        b⃗      =  (t⃗  × u⃗ ) / ‖t⃗  × u⃗ ‖          unit binormal  (wing direction)
        n⃗      =  b⃗  × t⃗                         unit normal    (lift direction)

    Load factor:
        n  = L / (m · g)
    """

    def __init__(
            self,
            state:       AircraftKinematicsState,
            goal:        AircraftGoal,
            constraints: AircraftKinematicsMaxConstrains,
            dt:          float,
    ) -> None:
        self.state:       AircraftKinematicsState          = state
        self.goal:        AircraftGoal                     = goal
        self.constraints: AircraftKinematicsMaxConstrains  = constraints
        self.dt:          float                            = dt
        self.gravity:     NDArray[np.float64]              = np.array([0.0, 0.0, -9.81])

    def unit_vectors(
            self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        t⃗   =  v⃗ / ‖v⃗‖                        unit tangent
        u⃗   =  [0, 0, 1]ᵀ                     vertical unit vector
        b⃗   =  (t⃗  × u⃗ ) / ‖t⃗  × u⃗ ‖          unit binormal  (wing direction)
        n⃗   =  b⃗  × t⃗                         unit normal    (lift direction)

        Note:  L⃗ ⊥ v⃗  →   lift is always perpendicular to velocity, but not L⃗ ⊥ φ (roll)

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

        return t_hat, b_hat, n_hat

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

        n⃗_rot = n⃗ cos(φ) + (t⃗ × n⃗) sinφ + t⃗ (t⃗ · n⃗)(1 - cos(φ))

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

        # n⃗_rot = n⃗ cos(φ) + (t⃗ × n⃗) sinφ + t⃗ (t⃗ · n⃗)(1 - cos(φ))
        n_hat_rot = (
                n_hat * np.cos(phi)
                + np.cross(t_hat, n_hat) * np.sin(phi)
                + t_hat * np.dot(t_hat, n_hat) * (1 - np.cos(phi))
        )

        return t_hat, b_hat, n_hat_rot

    def apply_action(
            self,
            thrust: float,
            flight_path_angle: float,
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
        """
        # Δγ = γ_cmd - γ
        gamma_delta = flight_path_angle - self.state.gamma
        # γ_{k+1} = γ_k + Δγ
        self.state.gamma += gamma_delta

        # v⃗
        v_vec = self.state.velocity_vector()

        # t⃗ , _, n⃗
        t_hat, _, n_hat = self.unit_vectors()
        t_hat, _, n_hat = self.unit_vectors_with_roll(roll)

        # D = 1/2 · ρ · v² · C_D · A
        drag = self.state.drag_force()
        # L = 1/2 · ρ · v² · C_L · A
        lift = self.state.lift_force()

        # v̇ = ((T - D) / m) · (v⃗ / ‖v⃗‖) + (L / m) · n⃗ + g⃗
        v_dot = (
                ((thrust - drag) / self.state.mass) * t_hat
                + (lift / self.state.mass) * n_hat
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

        # γ = arcsin(v_z / v)
        self.state.gamma = np.degrees(np.arcsin(v_vec[2] / self.state.v))
        # 𝒳 = arctan2(v_x, v_y)
        self.state.chi = np.degrees(np.arctan2(v_vec[0], v_vec[1]))

        # 𝒳̇ = (g · tan(φ)) / v
        chi_dot = (9.81 * np.tan(np.radians(roll))) / max(self.state.v, 1e-6)
        # 𝒳̇ _{k+1} = 𝒳_k + 𝒳̇ · dt
        self.state.chi += np.degrees(chi_dot * self.dt)

        return self.state.to_array()

def main():
    dt = 0.1
    steps = 500

    state: AircraftKinematicsState = AircraftKinematicsState(
        x=0.0,
        y=0.0,
        h=3000.0,
        v=120.0,
        gamma=0.0,
        chi=45.0,
        mass=6000.0,
        C_D=0.03,
        C_L=0.8,
        area=16.2
    )

    constraints: AircraftKinematicsMaxConstrains = AircraftKinematicsMaxConstrains(
        v_min=60,
        v_max=250,
        roc_max=20,
        rod_max=20,
        gamma_dot_max=15,
        chi_dot_max=15
    )

    goal: AircraftGoal = AircraftGoal(
        x=40000.0,
        y=25000.0,
        h=1000.0
    )

    env: AircraftEnvironment = AircraftEnvironment(
        state=state,
        goal=goal,
        constraints=constraints,
        dt=dt
    )

    trajectory = []

    for _ in range(steps):
        thrust = 20000.0

        flight_path_angle = -2.0

        roll = -30.0

        state_vec = env.apply_action(
            thrust=thrust,
            flight_path_angle=flight_path_angle,
            roll=roll
        )

        trajectory.append(state_vec.copy())

    trajectory = np.array(trajectory)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        label="Trajectory"
    )

    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        trajectory[0, 2],
        color="green",
        s=80,
        label="Start"
    )

    ax.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        trajectory[-1, 2],
        color="red",
        s=80,
        label="End"
    )

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Altitude [m]")

    plt.title("Aircraft Trajectory Simulation")
    plt.legend()

    plt.show()

def main2():
    dt = 0.1
    steps = 500

    constraints = AircraftKinematicsMaxConstrains(
        v_min=60,
        v_max=250,
        roc_max=20,
        rod_max=20,
        gamma_dot_max=15,
        chi_dot_max=15
    )

    goal = AircraftGoal(
        x=40000.0,
        y=25000.0,
        h=1000.0
    )

    state_left = AircraftKinematicsState(
        x=0.0, y=0.0, h=3000.0,
        v=120.0, gamma=0.0, chi=0.0,
        mass=6000.0, C_D=0.03, C_L=0.8, area=16.2
    )

    env_left = AircraftEnvironment(state_left, goal, constraints, dt)

    state_right = AircraftKinematicsState(
        x=0.0, y=0.0, h=3000.0,
        v=120.0, gamma=0.0, chi=0.0,
        mass=6000.0, C_D=0.03, C_L=0.8, area=16.2
    )

    env_right = AircraftEnvironment(state_right, goal, constraints, dt)

    traj_left = []
    traj_right = []

    thrust_1 = 20000.0
    thrust_2 = 20000.0
    flight_path_angle_1 = 10.0
    flight_path_angle_2 = -10.0

    roll_left = -30.0
    roll_right = +30.0

    for _ in range(steps):
        traj_left.append(
            env_left.apply_action(
                thrust=thrust_1,
                flight_path_angle=flight_path_angle_1,
                roll=roll_left
            ).copy()
        )

        traj_right.append(
            env_right.apply_action(
                thrust=thrust_2,
                flight_path_angle=flight_path_angle_2,
                roll=roll_right
            ).copy()
        )

    traj_left = np.array(traj_left)
    traj_right = np.array(traj_right)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(
        traj_left[:, 0],
        traj_left[:, 1],
        traj_left[:, 2],
        label="Left Turn",
        color="blue"
    )

    ax.plot(
        traj_right[:, 0],
        traj_right[:, 1],
        traj_right[:, 2],
        label="Right Turn",
        color="orange"
    )

    # Start points
    ax.scatter(traj_left[0, 0], traj_left[0, 1], traj_left[0, 2], color="green", s=60)
    ax.scatter(traj_right[0, 0], traj_right[1, 1], traj_right[0, 2], color="green", s=60)

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Altitude [m]")
    ax.set_title("Opposite Bank Angle Test (Left vs Right Turn)")
    ax.legend()

    plt.show()

    fig2, ax2 = plt.subplots(figsize=(8, 8))

    ax2.plot(
        traj_left[:, 0],
        traj_left[:, 1],
        label="Left Turn"
    )

    ax2.plot(
        traj_right[:, 0],
        traj_right[:, 1],
        label="Right Turn"
    )

    ax2.scatter(
        traj_left[0, 0],
        traj_left[0, 1],
        s=80,
        label="Start"
    )

    ax2.scatter(
        traj_left[-1, 0],
        traj_left[-1, 1],
        s=80,
        label="Left End"
    )

    ax2.scatter(
        traj_right[-1, 0],
        traj_right[-1, 1],
        s=80,
        label="Right End"
    )

    ax2.set_xlabel("East [m]")
    ax2.set_ylabel("North [m]")

    ax2.set_title("Top View (East-North Plane)")

    ax2.axis("equal")
    ax2.grid(True)
    ax2.legend()

    plt.show()

def main3():
    dt = 0.1
    steps = 500

    constraints = AircraftKinematicsMaxConstrains(
        v_min=60,
        v_max=300,
        roc_max=20,
        rod_max=20,
        gamma_dot_max=15,
        chi_dot_max=15
    )

    goal = AircraftGoal(
        x=40000.0,
        y=25000.0,
        h=1000.0
    )

    # Boeing 777
    b777 = AircraftKinematicsState(
        x=0.0, y=0.0, h=3000.0,
        v=230.0, gamma=0.0, chi=0.0,
        mass=297000.0,
        C_D=0.025,
        C_L=0.7,
        area=427.8
    )

    # Boeing 737
    b737 = AircraftKinematicsState(
        x=0.0, y=0.0, h=3000.0,
        v=220.0, gamma=0.0, chi=0.0,
        mass=65000.0,
        C_D=0.028,
        C_L=0.7,
        area=124.6
    )

    # A380
    a380 = AircraftKinematicsState(
        x=0.0, y=0.0, h=3000.0,
        v=240.0, gamma=0.0, chi=0.0,
        mass=560000.0,
        C_D=0.025,
        C_L=0.65,
        area=845.0
    )

    env_b777 = AircraftEnvironment(b777, goal, constraints, dt)
    env_b737 = AircraftEnvironment(b737, goal, constraints, dt)
    env_a380 = AircraftEnvironment(a380, goal, constraints, dt)

    traj_777, traj_b, traj_a = [], [], []

    thrust = 20000.0
    flight_path_angle = -5.0
    roll = 30.0

    for _ in range(steps):
        traj_777.append(env_b777.apply_action(thrust, flight_path_angle, roll).copy())
        traj_b.append(env_b737.apply_action(thrust, flight_path_angle, roll).copy())
        traj_a.append(env_a380.apply_action(thrust, flight_path_angle, roll).copy())

    traj_777 = np.array(traj_777)
    traj_b = np.array(traj_b)
    traj_a = np.array(traj_a)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(traj_777[:, 0], traj_777[:, 1], traj_777[:, 2], label="B777", color="green")
    ax.plot(traj_b[:, 0], traj_b[:, 1], traj_b[:, 2], label="B737", color="blue")
    ax.plot(traj_a[:, 0], traj_a[:, 1], traj_a[:, 2], label="A380", color="red")

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Altitude [m]")
    ax.set_title("Aircraft Comparison Simulation")
    ax.legend()

    plt.show()

    fig2, ax2 = plt.subplots(figsize=(8, 8))

    ax2.plot(traj_777[:, 0], traj_777[:, 1], label="B777")
    ax2.plot(traj_b[:, 0], traj_b[:, 1], label="B737")
    ax2.plot(traj_a[:, 0], traj_a[:, 1], label="A380")

    ax2.set_xlabel("East [m]")
    ax2.set_ylabel("North [m]")
    ax2.set_title("Top View Comparison")
    ax2.axis("equal")
    ax2.grid(True)
    ax2.legend()

    plt.show()

if __name__ == "__main__":
    main3()