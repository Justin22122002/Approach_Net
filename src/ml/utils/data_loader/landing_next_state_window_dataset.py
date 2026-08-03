import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from pyproj import Transformer

from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from physics.trajectory_action_processor import FlightTrajectoryProcessor, TrajectoryMode
from utils_project.geo_transformer import GeoTransformer


class LandingNextStateWindowDataset(Dataset):
    """
    Sliding-window dataset for the iTransformer-based NEXT-STATE model: given
    the last `window_len` states, predict the state ONE step ahead
    (x, y, h, v, gamma, chi) — no action, no physics inversion needed.

    Per-timestep dynamic features (6): x, y, h, v, gamma, chi
    Static features, repeated at every timestep (9):
        dynamics (5): mass, C_D0, area, cl_max, k
        runway   (4): fix x, fix y, fix elevation, runway bearing
    -> num_variates = 15 (same layout as LandingActionWindowDataset)

    x: (window_len, 15)   y: (6,)  [x, y, h, v, gamma, chi] at step end+1
    """

    DYNAMICS_KEYS = ("mass", "C_D0", "area", "cl_max", "k")

    def __init__(
        self,
        flight_dataset: LazyFlightParquetDataset,
        mode: TrajectoryMode,
        airport_center_latlon: tuple[float, float],
        airport_elevation_m: float,
        window_len: int = 16,
        max_dt: float = 5.0,
    ) -> None:
        self.flight_dataset = flight_dataset
        self.processor = FlightTrajectoryProcessor(flight_dataset)
        self.mode = mode
        self.window_len = window_len
        self.max_dt = max_dt

        self.airport_center_latlon = airport_center_latlon
        self.airport_elevation_m = float(airport_elevation_m)
        alat, alon = airport_center_latlon
        self._utm = Transformer.from_crs("EPSG:4326", GeoTransformer.utm_epsg(alat, alon), always_xy=True)
        self.airport_cx, self.airport_cy = self._utm.transform(alon, alat)

        self._cache: dict[int, np.ndarray] = {}          # flight_idx -> states (N, 6)
        self._static_cache: dict[int, np.ndarray] = {}   # flight_idx -> static feat (9,)
        self.samples: list[tuple[int, int]] = []          # (flight_idx, end_step_idx) -> target is states[end+1]
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
        return np.array(
            [fx, fy, float(row["fix_elevation_m"]) - self.airport_elevation_m, row["runway_bearing"]],
            dtype=np.float32,
        )

    def _build_flight(self, flight_idx: int) -> tuple[np.ndarray, np.ndarray]:
        traj, dynamics, _ = self.processor.load(flight_idx, mode=self.mode)
        icao24, flight_id = self.flight_dataset.flights[flight_idx]

        dyn_feat = np.array([getattr(dynamics, k) for k in self.DYNAMICS_KEYS], dtype=np.float32)
        rwy_feat = self._runway_features(icao24, flight_id)
        static_feat = np.concatenate([dyn_feat, rwy_feat])  # (9,)

        states = np.array([
            [
                s.x - self.airport_cx, s.y - self.airport_cy,
                s.h - self.airport_elevation_m, s.v, s.gamma, s.chi,
            ]
            for s in traj
        ], dtype=np.float32)

        return states, static_feat

    def _build_index(self) -> None:
        for flight_idx in range(len(self.flight_dataset)):
            try:
                states, static_feat = self._build_flight(flight_idx)
            except Exception as e:
                print(f"[WARN] skipping flight {flight_idx}: {e}")
                continue
            # need window_len states as input + 1 more as target -> window_len + 1 min length
            if len(states) < self.window_len + 1:
                continue

            self._cache[flight_idx] = states
            self._static_cache[flight_idx] = static_feat
            for end in range(self.window_len - 1, len(states) - 1):
                self.samples.append((flight_idx, end))
                self.flight_id_of_sample.append(flight_idx)

    @property
    def num_variates(self) -> int:
        return 6 + 9  # 15

    def __len__(self) -> int:
        return len(self.samples)

    def fit_normalization_stats(self, flight_indices: list[int] | np.ndarray) -> None:
        xs, ys = [], []
        for f in flight_indices:
            if f not in self._cache:
                continue
            states = self._cache[f]
            static_tiled = np.tile(self._static_cache[f], (len(states), 1))
            xs.append(np.concatenate([states, static_tiled], axis=1))
            ys.append(states)  # next-state targets are the same channels as the state features

        all_x = np.concatenate(xs, axis=0)
        all_y = np.concatenate(ys, axis=0)
        self.x_mean, self.x_std = all_x.mean(0), all_x.std(0) + 1e-8
        self.y_mean, self.y_std = all_y.mean(0), all_y.std(0) + 1e-8

    def set_normalization_stats(self, x_mean, x_std, y_mean, y_std) -> None:
        self.x_mean, self.x_std, self.y_mean, self.y_std = x_mean, x_std, y_mean, y_std

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if self.x_mean is None:
            raise RuntimeError("Normalization stats not set. Call flight_level_split() first.")

        flight_idx, end = self.samples[idx]
        states = self._cache[flight_idx]
        static_tiled = np.tile(self._static_cache[flight_idx], (self.window_len, 1))

        window_states = states[end - self.window_len + 1 : end + 1]   # (T, 6)
        x = np.concatenate([window_states, static_tiled], axis=1)     # (T, 15)
        y = states[end + 1]                                            # (6,) next state

        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))

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

    def flight_level_split(self, val_frac: float = 0.2, seed: int = 42) -> tuple[Subset, Subset]:
        train_flights, val_flights = self.split_flights(val_frac, seed)
        self.fit_normalization_stats(train_flights)
        train_idx = self.indices_for_flights(set(train_flights))
        val_idx = self.indices_for_flights(set(val_flights))
        return Subset(self, train_idx), Subset(self, val_idx)