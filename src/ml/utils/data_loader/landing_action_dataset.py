import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset

from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from physics.aircraft_environment_casadi import AircraftEnvironmentCasadi
from physics.trajectory_action_processor import FlightTrajectoryProcessor


class LandingActionDataset(Dataset):
    """
    input  (19): aircraft dynamics (9) + state lat/lon/h/v/gamma/chi (6)
                 + landing runway lat/lon/elevation/heading (4)
    output (3) : thrust, load_factor, roll
    """

    DYNAMICS_KEYS = (
        "mass", "C_D0", "area", "cl_max",
        "max_thrust", "max_load_factor", "min_load_factor", "max_roll_angle", "k",
    )

    def __init__(
        self,
        flight_dataset: LazyFlightParquetDataset,
        max_dt: float = 5.0,
    ) -> None:
        self.flight_dataset: LazyFlightParquetDataset = flight_dataset
        self.processor: FlightTrajectoryProcessor = FlightTrajectoryProcessor(flight_dataset)
        self.max_dt = max_dt

        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # flight_idx -> (X, Y)
        self.samples: list[tuple[int, int]] = []                    # (flight_idx, step_idx)
        self.flight_id_of_sample: list[int] = []

        self._build_index()

        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None

    def _runway_features(self, icao24: str, flight_id: int) -> np.ndarray:
        try:
            row = self.flight_dataset.get_runway_assignment(icao24, flight_id)
        except KeyError:
            return np.zeros(4, dtype=np.float32)

        return np.array([
            row["fix_latitude"], row["fix_longitude"],
            row["fix_elevation_m"], row["runway_bearing"],
        ], dtype=np.float32)

    def _build_flight(self, flight_idx: int) -> tuple[np.ndarray, np.ndarray]:
        traj, dynamics, timestamps = self.processor.load(flight_idx, mode="raw")
        icao24, flight_id = self.flight_dataset.flights[flight_idx]

        df = self.flight_dataset[flight_idx].sort_values("time").reset_index(drop=True)
        lat, lon = df["lat"].to_numpy(), df["lon"].to_numpy()

        dyn_feat = np.array([getattr(dynamics, k) for k in self.DYNAMICS_KEYS], dtype=np.float32)
        rwy_feat = self._runway_features(icao24, flight_id)

        xs, ys = [], []
        for i in range(len(traj) - 1):
            dt = (timestamps[i + 1] - timestamps[i]).total_seconds()
            if dt <= 0 or dt > self.max_dt:
                continue

            env = AircraftEnvironmentCasadi(state=traj[i], goal=None, dt=dt)
            thrust, load_factor, roll = env.inverse_action(traj[i + 1])

            state_feat = np.array(
                [lat[i], lon[i], traj[i].h, traj[i].v, traj[i].gamma, traj[i].chi],
                dtype=np.float32,
            )

            xs.append(np.concatenate([dyn_feat, state_feat, rwy_feat]))
            ys.append([float(thrust), float(load_factor), float(roll)])

        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    def _build_flight_cap(self, flight_idx: int) -> tuple[np.ndarray, np.ndarray]:
        traj, dynamics, timestamps = self.processor.load(flight_idx, mode="raw")
        icao24, flight_id = self.flight_dataset.flights[flight_idx]

        df = self.flight_dataset[flight_idx].sort_values("time").reset_index(drop=True)
        lat, lon = df["lat"].to_numpy(), df["lon"].to_numpy()

        dyn_feat = np.array([getattr(dynamics, k) for k in self.DYNAMICS_KEYS], dtype=np.float32)
        rwy_feat = self._runway_features(icao24, flight_id)

        xs, ys = [], []
        for i in range(len(traj) - 1):
            dt = (timestamps[i + 1] - timestamps[i]).total_seconds()
            if dt <= 0 or dt > self.max_dt:
                continue

            env = AircraftEnvironmentCasadi(state=traj[i], goal=None, dt=dt)
            thrust, load_factor, roll = env.inverse_action(traj[i + 1])
            thrust, load_factor, roll = float(thrust), float(load_factor), float(roll)

            thrust = float(np.clip(thrust, 0.0, dynamics.max_thrust))
            load_factor = float(np.clip(load_factor, dynamics.min_load_factor, dynamics.max_load_factor))
            roll = float(np.clip(roll, -dynamics.max_roll_angle, dynamics.max_roll_angle))

            state_feat = np.array(
                [lat[i], lon[i], traj[i].h, traj[i].v, traj[i].gamma, traj[i].chi],
                dtype=np.float32,
            )

            xs.append(np.concatenate([dyn_feat, state_feat, rwy_feat]))
            ys.append([thrust, load_factor, roll])

        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    def _build_index(self) -> None:
        for flight_idx in range(len(self.flight_dataset)):
            try:
                X, Y = self._build_flight(flight_idx) # _build_flight_cap _build_flight
            except Exception as e:
                print(f"[WARN] skipping flight {flight_idx}: {e}")
                continue
            if len(X) == 0:
                continue

            self._cache[flight_idx] = (X, Y)
            for step_idx in range(len(X)):
                self.samples.append((flight_idx, step_idx))
                self.flight_id_of_sample.append(flight_idx)

    @property
    def input_dim(self) -> int:
        return len(self.DYNAMICS_KEYS) + 6 + 4  # 19

    def __len__(self) -> int:
        return len(self.samples)

    def fit_normalization_stats(self, flight_indices: list[int] | np.ndarray) -> None:
        xs = [self._cache[f][0] for f in flight_indices if f in self._cache]
        ys = [self._cache[f][1] for f in flight_indices if f in self._cache]

        all_x = np.concatenate(xs, axis=0)
        all_y = np.concatenate(ys, axis=0)

        self.x_mean, self.x_std = all_x.mean(0), all_x.std(0) + 1e-8
        self.y_mean, self.y_std = all_y.mean(0), all_y.std(0) + 1e-8

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if self.x_mean is None:
            raise RuntimeError(
                "Normalization stats not set. Call flight_level_split() first."
            )

        flight_idx, step_idx = self.samples[idx]
        X, Y = self._cache[flight_idx]

        x = (X[step_idx] - self.x_mean) / self.x_std
        y = (Y[step_idx] - self.y_mean) / self.y_std
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))

    def flight_level_split(self, val_frac: float = 0.2, seed: int = 42) -> tuple[Subset, Subset]:
        rng = np.random.default_rng(seed)
        flight_indices = np.array(sorted(self._cache.keys()))
        rng.shuffle(flight_indices)

        n_val = max(1, int(len(flight_indices) * val_frac))
        val_flights = set(flight_indices[:n_val].tolist())
        train_flights = [f for f in flight_indices if f not in val_flights]

        self.fit_normalization_stats(train_flights)

        train_idx = [i for i, f in enumerate(self.flight_id_of_sample) if f not in val_flights]
        val_idx = [i for i, f in enumerate(self.flight_id_of_sample) if f in val_flights]
        return Subset(self, train_idx), Subset(self, val_idx)


if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt

    base = LazyFlightParquetDataset("../../../data/CYVR/20260628_1254")

    dataset = LandingActionDataset(base)

    print(f"Loaded {len(dataset)} transitions")
    print(f"Valid flights: {len(dataset._cache)}")

    n_plots = min(5, len(dataset._cache))

    random.seed(42)
    flight_indices = random.sample(list(dataset._cache.keys()), n_plots)

    for flight_idx in flight_indices:

        df = base[flight_idx].sort_values("time")

        icao24, flight_id = base.flights[flight_idx]

        plt.figure(figsize=(8, 8))

        scatter = plt.scatter(
            df["lon"],
            df["lat"],
            c=df["geoaltitude"],
            cmap="viridis",
            s=8,
        )

        plt.plot(
            df["lon"].iloc[0],
            df["lat"].iloc[0],
            "go",
            markersize=10,
            label="Start",
        )

        plt.plot(
            df["lon"].iloc[-1],
            df["lat"].iloc[-1],
            "ro",
            markersize=10,
            label="End",
        )

        try:
            rwy = base.get_runway_assignment(icao24, flight_id)

            plt.plot(
                rwy["fix_longitude"],
                rwy["fix_latitude"],
                "kx",
                markersize=12,
                markeredgewidth=3,
                label="Runway Fix",
            )
        except KeyError:
            pass

        plt.title(f"{icao24}  Flight {flight_id}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()

        cbar = plt.colorbar(scatter)
        cbar.set_label("Geo Altitude [m]")

        plt.show()