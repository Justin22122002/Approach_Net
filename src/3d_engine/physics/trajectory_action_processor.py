import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from typing import Literal

from pandas import Series
from pyproj import Transformer
from scipy.interpolate import UnivariateSpline

from physics.aircraft_dynamics import AircraftDynamics
from physics.aircraft_kinematics_state_casadi import AircraftKinematicsStateCasadi
from physics.aircraft_environment_casadi import AircraftEnvironmentCasadi
from utils_project.geo_transformer import GeoTransformer
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset


TrajectoryMode: type[str] = Literal["raw", "reconstructed", "bspline_position", "bspline_state"]


class FlightTrajectoryProcessor:

    def __init__(self, dataset: LazyFlightParquetDataset):
        self.dataset: LazyFlightParquetDataset = dataset

    def load(
            self,
            idx: int,
            mode: TrajectoryMode,
            smoothing_factor: float | dict[str, float] | None = None,
            spline_degree: int = 5,
    ) -> tuple[list[AircraftKinematicsStateCasadi], AircraftDynamics, list[Series]]:

        df = self.dataset[idx].sort_values("time").reset_index(drop=True)
        icao24, _ = self.dataset.flights[idx]
        _, dynamics = self.dataset.get_aircraft(icao24)

        lat0, lon0 = float(df["lat"].iloc[0]), float(df["lon"].iloc[0])

        transformer = Transformer.from_crs(
            "EPSG:4326", GeoTransformer.utm_epsg(lat0, lon0), always_xy=True
        )

        if mode in ("raw", "reconstructed"):
            return self._load_legacy(df, transformer, dynamics, mode)
        elif mode == "bspline_position":
            return self._load_bspline_position(df, transformer, dynamics, smoothing_factor, spline_degree)
        elif mode == "bspline_state":
            return self._load_bspline_state(df, transformer, dynamics, smoothing_factor, spline_degree)
        else:
            raise ValueError(mode)

    def _load_legacy(
            self,
            df: pd.DataFrame,
            transformer: Transformer,
            dynamics: AircraftDynamics,
            mode: TrajectoryMode,
    ) -> tuple[list[AircraftKinematicsStateCasadi], AircraftDynamics, list[Series]]:

        if mode == "raw":

            traj = []
            ts = []

            for _, r in df.iterrows():
                x, y = transformer.transform(float(r["lon"]), float(r["lat"]))

                v = float(r["velocity"])
                h = float(r["geoaltitude"])

                chi = float(r["heading"]) if pd.notna(r["heading"]) else 0.0
                vr = float(r["vertrate"]) if pd.notna(r["vertrate"]) else 0.0
                gamma = float(np.degrees(np.arctan2(vr, max(v, 1e-3))))

                traj.append(
                    AircraftKinematicsStateCasadi(
                        x=x,
                        y=y,
                        h=h,
                        v=v,
                        gamma=gamma,
                        chi=chi,
                        aircraft_type=dynamics
                    )
                )
                ts.append(r["time"])

            return traj, dynamics, ts

        elif mode == "reconstructed":

            coords = [transformer.transform(float(r["lon"]), float(r["lat"])) for _, r in df.iterrows()]
            xs = np.array([c[0] for c in coords])
            ys = np.array([c[1] for c in coords])
            hs = df["geoaltitude"].values

            ts = np.array([(t - df["time"].iloc[0]).total_seconds() for t in df["time"]])

            traj = []
            timestamps = []

            for i in range(len(ts)):

                i0 = i
                i1 = min(i + 1, len(ts) - 1)

                if i == len(ts) - 1:
                    i0 = i - 1
                    i1 = i

                dt = max(ts[i1] - ts[i0], 1e-3)

                dx = xs[i1] - xs[i0]
                dy = ys[i1] - ys[i0]
                dh = hs[i1] - hs[i0]

                v = math.sqrt(dx**2 + dy**2) / dt
                chi = np.degrees(np.arctan2(dx, dy)) % 360.0
                gamma = np.degrees(np.arctan2(dh, max(np.sqrt(dx**2 + dy**2), 1e-3)))

                traj.append(
                    AircraftKinematicsStateCasadi(
                        x=xs[i],
                        y=ys[i],
                        h=hs[i],
                        v=v,
                        gamma=gamma,
                        chi=chi,
                        aircraft_type=dynamics
                    )
                )
                timestamps.append(df["time"].iloc[i])

            return traj, dynamics, timestamps

        else:
            raise ValueError(mode)

    @staticmethod
    def _time_seconds(df: pd.DataFrame) -> np.ndarray:
        t0 = df["time"].iloc[0]
        return np.array([(t - t0).total_seconds() for t in df["time"]])

    def _load_bspline_position(
            self,
            df: pd.DataFrame,
            transformer: Transformer,
            dynamics: AircraftDynamics,
            smoothing_factor: float | dict[str, float] | None,
            k: int,
    ) -> tuple[list[AircraftKinematicsStateCasadi], AircraftDynamics, list[Series]]:

        coords = [transformer.transform(float(r["lon"]), float(r["lat"])) for _, r in df.iterrows()]
        xs = np.array([c[0] for c in coords])
        ys = np.array([c[1] for c in coords])
        hs = df["geoaltitude"].to_numpy(dtype=float)
        ts = self._time_seconds(df)

        m = len(ts)
        if isinstance(smoothing_factor, dict):
            s_x = smoothing_factor.get("x", m * 4.0 ** 2)
            s_y = smoothing_factor.get("y", m * 4.0 ** 2)
            s_h = smoothing_factor.get("h", m * 8.0 ** 2)
        elif smoothing_factor is None:
            s_x = s_y = m * 4.0 ** 2
            s_h = m * 8.0 ** 2
        else:
            s_x = s_y = s_h = float(smoothing_factor)

        spl_x = UnivariateSpline(ts, xs, k=k, s=s_x)
        spl_y = UnivariateSpline(ts, ys, k=k, s=s_y)
        spl_h = UnivariateSpline(ts, hs, k=k, s=s_h)

        x_s, y_s, h_s = spl_x(ts), spl_y(ts), spl_h(ts)
        vx, vy, vh = spl_x.derivative(1)(ts), spl_y.derivative(1)(ts), spl_h.derivative(1)(ts)

        v = np.sqrt(vx ** 2 + vy ** 2)
        chi = np.degrees(np.arctan2(vx, vy)) % 360.0
        gamma = np.degrees(np.arctan2(vh, np.maximum(v, 1e-3)))

        traj = [
            AircraftKinematicsStateCasadi(
                x=float(x_s[i]), y=float(y_s[i]), h=float(h_s[i]),
                v=float(v[i]), gamma=float(gamma[i]), chi=float(chi[i]),
                aircraft_type=dynamics,
            )
            for i in range(m)
        ]
        return traj, dynamics, list(df["time"])

    def _load_bspline_state(
            self,
            df: pd.DataFrame,
            transformer: Transformer,
            dynamics: AircraftDynamics,
            smoothing_factor: float | dict[str, float] | None,
            k: int,
    ) -> tuple[list[AircraftKinematicsStateCasadi], AircraftDynamics, list[Series]]:
        coords = [transformer.transform(float(r["lon"]), float(r["lat"])) for _, r in df.iterrows()]
        xs = np.array([c[0] for c in coords])
        ys = np.array([c[1] for c in coords])
        hs = df["geoaltitude"].to_numpy(dtype=float)
        ts = self._time_seconds(df)
        m = len(ts)

        v_raw = df["velocity"].to_numpy(dtype=float)
        chi_raw = df["heading"].fillna(0.0).to_numpy(dtype=float)
        vr_raw = df["vertrate"].fillna(0.0).to_numpy(dtype=float)

        chi_unwrapped = np.degrees(np.unwrap(np.radians(chi_raw)))

        if isinstance(smoothing_factor, dict):
            s_v = smoothing_factor.get("v", m * 1.0 ** 2)
            s_chi = smoothing_factor.get("chi", m * 3.0 ** 2)
            s_vr = smoothing_factor.get("vr", m * 0.3 ** 2)
        elif smoothing_factor is None:
            s_v, s_chi, s_vr = m * 1.0 ** 2, m * 3.0 ** 2, m * 0.3 ** 2
        else:
            s_v = s_chi = s_vr = float(smoothing_factor)

        spl_v = UnivariateSpline(ts, v_raw, k=k, s=s_v)
        spl_chi = UnivariateSpline(ts, chi_unwrapped, k=k, s=s_chi)
        spl_vr = UnivariateSpline(ts, vr_raw, k=k, s=s_vr)

        v_s = spl_v(ts)
        chi_s = spl_chi(ts) % 360.0
        vr_s = spl_vr(ts)
        gamma_s = np.degrees(np.arctan2(vr_s, np.maximum(v_s, 1e-3)))

        spl_x = UnivariateSpline(ts, xs, k=k, s=m * 4.0 ** 2)
        spl_y = UnivariateSpline(ts, ys, k=k, s=m * 4.0 ** 2)
        spl_h = UnivariateSpline(ts, hs, k=k, s=m * 8.0 ** 2)
        x_s, y_s, h_s = spl_x(ts), spl_y(ts), spl_h(ts)

        traj = [
            AircraftKinematicsStateCasadi(
                x=float(x_s[i]), y=float(y_s[i]), h=float(h_s[i]),
                v=float(v_s[i]), gamma=float(gamma_s[i]), chi=float(chi_s[i]),
                aircraft_type=dynamics,
            )
            for i in range(m)
        ]
        return traj, dynamics, list(df["time"])


def clone(s: AircraftKinematicsStateCasadi) -> AircraftKinematicsStateCasadi:
    return AircraftKinematicsStateCasadi(
        x=float(s.x), y=float(s.y), h=float(s.h),
        v=float(s.v), gamma=float(s.gamma), chi=float(s.chi),
        aircraft_type=s.aircraft_type
    )


def reconstruct_and_replay(
        trajectory: list[AircraftKinematicsStateCasadi],
        dynamics: AircraftDynamics,
        timestamps: list[Series],
        max_dt: float = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray,]:
    actions = []
    dts = []

    for i in range(len(trajectory) - 1):
        dt = float((timestamps[i + 1] - timestamps[i]).total_seconds())

        if dt <= 0:
            continue
        if max_dt is not None and dt > max_dt:
            continue

        env = AircraftEnvironmentCasadi(state=trajectory[i], goal=None, dt=dt)
        thrust, load_factor, roll = env.inverse_action(trajectory[i + 1])

        thrust = float(thrust)
        load_factor = float(load_factor)
        roll = float(roll)

        # capping to the aircraft dynamics
        thrust = np.clip(thrust, 0.0, dynamics.max_thrust)
        load_factor = np.clip(load_factor,
                              dynamics.min_load_factor,
                              dynamics.max_load_factor)
        roll = np.clip(roll,
                       -dynamics.max_roll_angle,
                       dynamics.max_roll_angle)

        actions.append((thrust, load_factor, roll))

        # when no capping
        # actions.append((float(thrust), float(load_factor), float(roll)))
        dts.append(dt)

    sim_state = clone(trajectory[0])
    sim = [[sim_state.x, sim_state.y, sim_state.h, sim_state.v]]

    for (T, n, phi), dt in zip(actions, dts):
        env = AircraftEnvironmentCasadi(state=sim_state, goal=None, dt=dt)
        env.apply_action(T, n, phi)
        sim.append([float(sim_state.x), float(sim_state.y), float(sim_state.h), float(sim_state.v)])

    real = np.array([[float(s.x), float(s.y), float(s.h), float(s.v)] for s in trajectory[:len(sim)]])

    return real, np.array(sim), np.array(actions)


def plot(
    real: np.ndarray,
    sim: np.ndarray,
    actions: np.ndarray | None = None,
    timestamps: list[Series] | None = None,
) -> None:
    t = np.arange(len(real)) if timestamps is None else np.array(
        [(ts - timestamps[0]).total_seconds() for ts in timestamps[:len(real)]]
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(real[:, 0], real[:, 1], "k-", label="Real trajectory")
    axes[0, 0].plot(sim[:, 0], sim[:, 1], "r--", label="Reconstructed")
    axes[0, 0].scatter(real[0, 0], real[0, 1], c="green", s=80)
    axes[0, 0].scatter(real[-1, 0], real[-1, 1], c="red", s=80)
    axes[0, 0].set_title("Ground track")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(t, real[:, 2], "k-", label="Real")
    axes[0, 1].plot(t, sim[:, 2], "r--", label="Reconstructed")
    axes[0, 1].set_title("Altitude")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(t, real[:, 3], "k-", label="Real")
    axes[1, 0].plot(t, sim[:, 3], "r--", label="Reconstructed")
    axes[1, 0].set_title("Velocity")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    if actions is not None and len(actions) > 0:
        actions = np.asarray(actions, dtype=np.float64)

        fig2, ax2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        ax2[0].plot(t[:len(actions)], actions[:, 0])
        ax2[0].set_title("Thrust")
        ax2[0].grid(True)

        ax2[1].plot(t[:len(actions)], actions[:, 1])
        ax2[1].set_title("Load factor")
        ax2[1].grid(True)

        ax2[2].plot(t[:len(actions)], actions[:, 2])
        ax2[2].set_title("Roll")
        ax2[2].set_xlabel("t [s]")
        ax2[2].grid(True)

        fig2.suptitle("Control actions")
        fig2.tight_layout()

    else:
        axes[1, 1].axis("off")

    fig.suptitle("Trajectory Reconstruction + Controls")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    dataset = LazyFlightParquetDataset("../../data/CYVR/20260625_1508")

    processor = FlightTrajectoryProcessor(dataset)

    # "raw" | "reconstructed" | "bspline_position" | "bspline_state"
    trajectory, dynamics, timestamps = processor.load(idx=128, mode="bspline_state")
    # trajectory, dynamics, timestamps = processor.load(idx=128, mode="bspline_position")
    # trajectory, dynamics, timestamps = processor.load(idx=128, mode="raw")
    # trajectory, dynamics, timestamps = processor.load(idx=4, mode="reconstructed")

    print(dynamics.max_thrust)

    real, sim, actions = reconstruct_and_replay(
        trajectory, dynamics, timestamps, max_dt=5.0
    )

    plot(real, sim, actions, timestamps)