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
        self.v_min: float = v_min # -> m/s
        self.v_max: float = v_max # -> m/s
        self.roc_max: float = roc_max # rate of climb max   -> m/s
        self.rod_max: float = rod_max # rate of decent max  -> m/s
        self.gamma_dot_max: float = gamma_dot_max # -> °/s
        self.chi_dot_max: float = chi_dot_max # -> °/s

class AircraftGoal:
    def __init__(self, x: float, y: float, h: float) -> None:
        self.x: float = x
        self.y: float = y
        self.h: float = h

class AircraftKinematicsState:
    def __init__(
            self,
            x: float,
            y: float,
            h: float,
            v: float,
            gamma: float,
            chi: float
    ) -> None:
        self.x: float = x         # m or lon
        self.y: float = y         # m or lat
        self.h: float = h         # m or feet
        self.v: float = v         # m/s
        self.gamma: float = gamma # flight path angle (Flugbahnwinkel)
        self.chi: float = chi     # Track angle / Course angle (Kursbahnwinkel) from North, clockwise

class AircraftEnvironment:
    """
    Point-mass aircraft kinematics environment.

    Assumptions:
        - The aircraft is modeled as a point mass (kinematic model)
        - No independent heading state is included
        - The state consists of position (x, y, h) and speed magnitude v
        - The direction of motion is fully defined by the velocity vector
        - gamma defines the vertical inclination of the velocity vector
        - chi defines the horizontal direction (track angle) of the velocity vector
        - Coordinated flight is assumed: velocity direction fully determines motion direction
        - Zero-order hold: control inputs are constant within each timestep
    """

    def __init__(
            self,
            state: AircraftKinematicsState,
            goal: AircraftGoal,
            dt: float
    ) -> None:
        self.state = state
        self.goal = goal
        self.dt = dt

    def step(self) -> None:
        """
        Propagate one timestep Δt

        Continuous model:
            dx/dt = v * cos(gamma) * sin(chi)
            dy/dt = v * cos(gamma) * cos(chi)
            dh/dt = v * sin(gamma)

        Discretization:
            x(t+Δt) = x(t) + v*cos(gamma)*sin(chi)*Δt
            y(t+Δt) = y(t) + v*cos(gamma)*cos(chi)*Δt
            h(t+Δt) = h(t) + v*sin(gamma)*Δt
        """

        v: float = self.state.v
        gamma: NDArray[np.float64] = np.radians(self.state.gamma)
        chi: NDArray[np.float64] = np.radians(self.state.chi)

        self.state.x += v * np.cos(gamma) * np.sin(chi) * self.dt
        self.state.y += v * np.cos(gamma) * np.cos(chi) * self.dt
        self.state.h += v * np.sin(gamma) * self.dt

    def compute_required_action(self, N: int | None = None, mode: str = "multi_step") -> tuple[float, float, float]:
        """
        Computes required control action

        Modes:
            - "multi_step": reach goal in N * dt (default)
            - "single_step": reach goal in ONE dt

        Returns:
            v_req, gamma_deg, chi_deg
        """

        dx: float = self.goal.x - self.state.x
        dy: float = self.goal.y - self.state.y
        dh: float = self.goal.h - self.state.h

        d_horiz: NDArray[np.float64] = np.sqrt(dx ** 2 + dy ** 2)
        d_3d: NDArray[np.float64] = np.sqrt(dx ** 2 + dy ** 2 + dh ** 2)

        if mode == "multi_step":
            if N is None:
                raise ValueError("N must be provided for multi_step mode")
            T = N * self.dt

        elif mode == "single_step":
            T = self.dt

        else:
            raise ValueError("mode must be 'multi_step' or 'single_step'")

        v_req = d_3d / T
        gamma: NDArray = np.arctan2(dh, d_horiz)
        chi: NDArray = np.arctan2(dx, dy)

        return v_req, np.degrees(gamma), np.degrees(chi)


class AircraftTrajectoryVisualizer:
    """
    Visualizes aircraft trajectory in:
        - 3D space (x, y, h)
        - 2D ground track (x, y)
        - altitude over time
    """

    def __init__(self) -> None:
        pass

    def plot(self, trajectory: list[tuple[float, float, float]], goal, dt: float) -> None:
        """
        trajectory: [(x, y, h), ...]
        goal: AircraftGoal
        dt: timestep
        """

        trajectory = np.array(trajectory)

        x = trajectory[:, 0]
        y = trajectory[:, 1]
        h = trajectory[:, 2]

        t = np.arange(len(trajectory)) * dt

        fig = plt.figure(figsize=(15, 5))

        # -------------------------
        # 1) 2D Top Down View
        # -------------------------
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(x, y, label="trajectory")
        ax1.scatter(x[0], y[0], c="green", label="start")
        ax1.scatter(goal.x, goal.y, c="red", label="goal")
        ax1.set_title("Top-Down View (x-y)")
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        ax1.legend()
        ax1.axis("equal")

        # -------------------------
        # 2) Altitude Profile
        # -------------------------
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(t, h)
        ax2.set_title("Altitude Profile")
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("h [m]")

        # -------------------------
        # 3) 3D Trajectory
        # -------------------------
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        ax3.plot(x, y, h, label="trajectory")
        ax3.scatter(x[0], y[0], h[0], c="green", label="start")
        ax3.scatter(goal.x, goal.y, goal.h, c="red", label="goal")

        ax3.set_title("3D Trajectory")
        ax3.set_xlabel("x [m]")
        ax3.set_ylabel("y [m]")
        ax3.set_zlabel("h [m]")

        plt.tight_layout()
        plt.show()

def main():
    state: AircraftKinematicsState = AircraftKinematicsState(
        x=0.0,
        y=0.0,
        h=3000.0,
        v=120.0,
        gamma=0.0,
        chi=45.0
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

    dt = 10.0
    N = 40

    env: AircraftEnvironment = AircraftEnvironment(state=state, goal=goal, dt=dt)

    mode = "single_step" # single_step, multi_step

    v_req, gamma_req, chi_req = env.compute_required_action(N, mode=mode)

    print("Required Action:")
    print(f"v_req    = {v_req:.2f} m/s")
    print(f"gamma_req= {gamma_req:.2f} deg")
    print(f"chi_req  = {chi_req:.2f} deg")

    state.v = v_req
    state.gamma = gamma_req
    state.chi = chi_req

    trajectory = []

    steps = 1 if mode == "single_step" else N

    for _ in range(steps):
        trajectory.append((state.x, state.y, state.h))
        env.step()

    trajectory.append((state.x, state.y, state.h))

    dx = state.x - goal.x
    dy = state.y - goal.y
    dh = state.h - goal.h

    error = np.sqrt(dx**2 + dy**2 + dh**2)

    print("\nFinal State:")
    print(f"x = {state.x:.2f}, y = {state.y:.2f}, h = {state.h:.2f}")
    print(f"Total error to goal = {error:.3f} m")

    visualizer = AircraftTrajectoryVisualizer()
    visualizer.plot(trajectory, goal, dt)


if __name__ == "__main__":
    main()