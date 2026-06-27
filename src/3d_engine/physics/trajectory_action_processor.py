from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from typing import Literal
from pyproj import Transformer

from physics.aircraft_kinematics_state_casadi import AircraftKinematicsStateCasadi
from physics.aircraft_environment_casadi import AircraftEnvironmentCasadi
from utils_project.geo_transformer import GeoTransformer
from src.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset


TrajectoryMode: type[str] = Literal["raw", "reconstructed"]


class FlightTrajectoryProcessor:

    def __init__(self, dataset: LazyFlightParquetDataset):
        self.dataset: LazyFlightParquetDataset = dataset

    def load(self, idx: int, mode: TrajectoryMode):

        df = self.dataset[idx].sort_values("time").reset_index(drop=True)
        icao24, _ = self.dataset.flights[idx]
        _, dynamics = self.dataset.get_aircraft(icao24)

        lat0, lon0 = float(df["lat"].iloc[0]), float(df["lon"].iloc[0])

        transformer = Transformer.from_crs(
            "EPSG:4326",
            GeoTransformer.utm_epsg(lat0, lon0),
            always_xy=True
        )

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


def clone(s: AircraftKinematicsStateCasadi) -> AircraftKinematicsStateCasadi:
    return AircraftKinematicsStateCasadi(
        x=float(s.x), y=float(s.y), h=float(s.h),
        v=float(s.v), gamma=float(s.gamma), chi=float(s.chi),
        aircraft_type=s.aircraft_type
    )


def reconstruct_and_replay(trajectory, dynamics, timestamps, max_dt=None):
    actions = []
    dts = []

    for i in range(len(trajectory) - 1):
        dt = float((timestamps[i + 1] - timestamps[i]).total_seconds())

        if dt <= 0:
            continue
        if max_dt is not None and dt > max_dt:
            continue

        env = AircraftEnvironmentCasadi(state=trajectory[i], goal=None, dt=dt)
        # todo: consider max actions for capping the actions !!!
        thrust, load_factor, roll = env.inverse_action(trajectory[i + 1])

        actions.append((float(thrust), float(load_factor), float(roll)))
        dts.append(dt)

    sim_state = clone(trajectory[0])
    sim = [[sim_state.x, sim_state.y, sim_state.h, sim_state.v]]

    for (T, n, phi), dt in zip(actions, dts):
        env = AircraftEnvironmentCasadi(state=sim_state, goal=None, dt=dt)
        env.apply_action(T, n, phi)
        sim.append([float(sim_state.x), float(sim_state.y), float(sim_state.h), float(sim_state.v)])

    real = np.array([[float(s.x), float(s.y), float(s.h), float(s.v)] for s in trajectory[:len(sim)]])

    return real, np.array(sim), np.array(actions)


def plot(real, sim, timestamps=None):
    t = np.arange(len(real)) if timestamps is None else np.array(
        [(ts - timestamps[0]).total_seconds() for ts in timestamps[:len(real)]]
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(real[:, 0], real[:, 1], "k-", label="Real trajectory")
    axes[0].plot(sim[:, 0], sim[:, 1], "r--", label="Reconstructed")
    axes[0].scatter(real[0, 0], real[0, 1], c="green", s=80, zorder=5)
    axes[0].scatter(real[-1, 0], real[-1, 1], c="red", s=80, zorder=5)
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("ground track")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, real[:, 2], "k-", label="Real trajectory")
    axes[1].plot(t, sim[:, 2], "r--", label="Reconstructed")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("h [m]")
    axes[1].set_title("altitude")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t, real[:, 3], "k-", label="Real trajectory")
    axes[2].plot(t, sim[:, 3], "r--", label="Reconstructed")
    axes[2].set_xlabel("t [s]")
    axes[2].set_ylabel("v [m/s]")
    axes[2].set_title("velocity")
    axes[2].grid(True)
    axes[2].legend()

    fig.suptitle("Trajectory Reconstruction (Inverse Dynamics + Replay)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    dataset = LazyFlightParquetDataset("../../data/CYVR/20260625_1508")

    processor = FlightTrajectoryProcessor(dataset)

    trajectory, dynamics, timestamps = processor.load(idx=128, mode="raw")
    # trajectory, dynamics, timestamps = processor.load(idx=4, mode="reconstructed")

    real, sim, actions = reconstruct_and_replay(
        trajectory, dynamics, timestamps, max_dt=5.0
    )

    plot(real, sim, timestamps)