from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from pyproj import Transformer

from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from physics.trajectory_action_processor import FlightTrajectoryProcessor, TrajectoryMode
from utils.airport import Airport
from utils_project.geo_transformer import GeoTransformer


class LandingNextStateDataset(Dataset):
    """
    input  (15): aircraft dynamics (5) + state x/y/h/v/gamma/chi (6)
                 + landing runway x/y/elevation/heading (4)
    output (6) : NEXT state x/y/h/v/gamma/chi (one step ahead)

    Same feature layout / frame conventions as LandingActionDataset, but
    predicts the next state directly instead of the action that produced it
    -> no AircraftEnvironmentCasadi.inverse_action() needed.
    """

    DYNAMICS_KEYS = (
        "mass", "C_D0", "area", "cl_max", "k",
    )

    def __init__(
        self,
        flight_dataset: LazyFlightParquetDataset,
        mode: TrajectoryMode,
        airport_center_latlon: tuple[float, float],
        airport_elevation_m: float,
        max_dt: float = 5.0,
    ) -> None:
        self.flight_dataset: LazyFlightParquetDataset = flight_dataset
        self.processor: FlightTrajectoryProcessor = FlightTrajectoryProcessor(flight_dataset)
        self.max_dt: float = max_dt
        self.mode: TrajectoryMode = mode

        self.airport_center_latlon: tuple[float, float] = airport_center_latlon
        self.airport_elevation_m: float = float(airport_elevation_m)
        alat, alon = airport_center_latlon
        self._utm: Transformer = Transformer.from_crs(
            "EPSG:4326", GeoTransformer.utm_epsg(alat, alon), always_xy=True
        )
        self.airport_cx, self.airport_cy = self._utm.transform(alon, alat)

        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # flight_idx -> (X, Y)
        self.samples: list[tuple[int, int]] = []
        self.flight_id_of_sample: list[int] = []

        self._build_index()

        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None

    def _to_airport_frame(self, lon: float, lat: float) -> tuple[float, float]:
        x, y = self._utm.transform(float(lon), float(lat))
        return x - self.airport_cx, y - self.airport_cy

    def _runway_features(self, icao24: str, flight_id: int) -> np.ndarray:
        try:
            row = self.flight_dataset.get_runway_assignment(icao24, flight_id)
        except KeyError:
            return np.zeros(4, dtype=np.float32)

        fx, fy = self._to_airport_frame(row["fix_longitude"], row["fix_latitude"])
        return np.array([
            fx, fy,
            float(row["fix_elevation_m"]) - self.airport_elevation_m,
            row["runway_bearing"],
        ], dtype=np.float32)

    def _state_feat(self, s) -> np.ndarray:
        return np.array(
            [s.x - self.airport_cx, s.y - self.airport_cy,
             s.h - self.airport_elevation_m, s.v, s.gamma, s.chi],
            dtype=np.float32,
        )

    def _build_flight(self, flight_idx: int) -> tuple[np.ndarray, np.ndarray]:
        traj, dynamics, timestamps = self.processor.load(flight_idx, mode=self.mode)
        icao24, flight_id = self.flight_dataset.flights[flight_idx]

        dyn_feat = np.array([getattr(dynamics, k) for k in self.DYNAMICS_KEYS], dtype=np.float32)
        rwy_feat = self._runway_features(icao24, flight_id)

        xs, ys = [], []
        for i in range(len(traj) - 1):
            dt = (timestamps[i + 1] - timestamps[i]).total_seconds()
            if dt <= 0 or dt > self.max_dt:
                continue

            state_feat = self._state_feat(traj[i])
            next_state_feat = self._state_feat(traj[i + 1])

            xs.append(np.concatenate([dyn_feat, state_feat, rwy_feat]))
            ys.append(next_state_feat)

        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    def _build_index(self) -> None:
        for flight_idx in range(len(self.flight_dataset)):
            try:
                X, Y = self._build_flight(flight_idx)
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
        return len(self.DYNAMICS_KEYS) + 6 + 4  # 15

    @property
    def output_dim(self) -> int:
        return 6

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

    def split_flights(self, val_frac: float = 0.2, seed: int = 42) -> tuple[list[int], list[int]]:
        rng = np.random.default_rng(seed)
        flight_indices = np.array(sorted(self._cache.keys()))
        rng.shuffle(flight_indices)
        n_val = max(1, int(len(flight_indices) * val_frac))
        val_flights = flight_indices[:n_val].tolist()
        train_flights = [f for f in flight_indices if f not in set(val_flights)]
        return train_flights, val_flights

    def indices_for_flights(self, flights: set[int]) -> list[int]:
        return [i for i, f in enumerate(self.flight_id_of_sample) if f in flights]

    def set_normalization_stats(
            self, x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray
    ) -> None:
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std