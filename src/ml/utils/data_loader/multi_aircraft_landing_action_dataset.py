from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from pandas import Timestamp
from pyproj import Transformer

from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from physics.aircraft_environment_casadi import AircraftEnvironmentCasadi
from physics.trajectory_action_processor import FlightTrajectoryProcessor, TrajectoryMode
from utils.airport import Airport
from utils_project.geo_transformer import GeoTransformer


class MultiAircraftLandingActionDataset:
    """
    Multi-aircraft variant of LandingActionDataset.

    Per-aircraft feature/action computation (dynamics, state, runway fix,
    inverse_action) is identical to LandingActionDataset - only the grouping
    changes. No rotation augmentation is applied here.

    input  (15) per aircraft: dynamics (5) + state x/y/h/v/gamma/chi (6)
                              + landing runway x/y/elevation/heading (4)
    output (3)  per aircraft: thrust, load_factor, roll
    """

    DYNAMICS_KEYS = (
        "mass", "C_D0", "area", "cl_max", "k",
    )

    def __init__(
        self,
        flight_dataset: LazyFlightParquetDataset,
        mode: TrajectoryMode,
        airport_center_latlon: tuple[float, float],  # note: (lat, lon)
        airport_elevation_m: float,                  # reference height subtracted from h and fix_elevation
        max_dt: float = 5.0,
    ) -> None:
        self.flight_dataset: LazyFlightParquetDataset = flight_dataset
        self.processor: FlightTrajectoryProcessor = FlightTrajectoryProcessor(
            dataset=flight_dataset,
            target_epsg=GeoTransformer.utm_epsg(*airport_center_latlon)
        )
        self.max_dt: float = max_dt
        self.mode: TrajectoryMode = mode

        self.airport_center_latlon: tuple[float, float] = airport_center_latlon
        self.airport_elevation_m: float = float(airport_elevation_m)
        alat, alon = airport_center_latlon
        self._utm: Transformer = Transformer.from_crs(
            "EPSG:4326", GeoTransformer.utm_epsg(alat, alon), always_xy=True
        )
        self.airport_cx, self.airport_cy = self._utm.transform(alon, alat)

        # flight_idx -> (X (n,15), Y (n,3), ts (list[Timestamp], len n, aligned with X/Y rows)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, list[Timestamp]]] = {}
        self._flight_start_time: dict[int, Timestamp] = {}

        self._build_index()

        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None

    def _to_airport_frame(self, lon: float, lat: float) -> tuple[float, float]:
        """lon/lat (deg) -> airport-centered UTM (x, y) in meters."""
        x, y = self._utm.transform(float(lon), float(lat))
        return x - self.airport_cx, y - self.airport_cy

    def _runway_features(self, icao24: str, flight_id: int) -> np.ndarray:
        try:
            row = self.flight_dataset.get_runway_assignment(icao24, flight_id)
        except KeyError:
            return np.zeros(4, dtype=np.float32)

        fx, fy = self._to_airport_frame(row["fix_longitude"], row["fix_latitude"])
        bearing = float(row["runway_bearing"])
        elev_rel = float(row["fix_elevation_m"]) - self.airport_elevation_m

        return np.array([fx, fy, elev_rel, bearing], dtype=np.float32)

    def _build_flight(self, flight_idx: int) -> tuple[np.ndarray, np.ndarray, list[Timestamp]]:
        traj, dynamics, timestamps = self.processor.load(flight_idx, mode=self.mode)
        icao24, flight_id = self.flight_dataset.flights[flight_idx]

        dyn_feat = np.array([getattr(dynamics, k) for k in self.DYNAMICS_KEYS], dtype=np.float32)
        rwy_feat = self._runway_features(icao24, flight_id)

        xs, ys, ts = [], [], []
        for i in range(len(traj) - 1):
            dt = (timestamps[i + 1] - timestamps[i]).total_seconds()
            if dt <= 0 or dt > self.max_dt:
                continue

            env = AircraftEnvironmentCasadi(state=traj[i], goal=None, dt=dt)
            thrust, load_factor, roll = env.inverse_action(traj[i + 1])

            state_feat = np.array(
                [traj[i].x - self.airport_cx, traj[i].y - self.airport_cy,
                 traj[i].h - self.airport_elevation_m, traj[i].v, traj[i].gamma, traj[i].chi],
                dtype=np.float32,
            )

            xs.append(np.concatenate([dyn_feat, state_feat, rwy_feat]))
            ys.append([float(thrust), float(load_factor), float(roll)])
            ts.append(timestamps[i])  # start-timestamp of this transition, used as the grouping key

        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), ts

    def _build_index(self) -> None:
        for flight_idx in range(len(self.flight_dataset)):
            try:
                X, Y, ts = self._build_flight(flight_idx)
            except Exception as e:
                print(f"[WARN] skipping flight {flight_idx}: {e}")
                continue
            if len(X) == 0:
                continue

            self._cache[flight_idx] = (X, Y, ts)
            self._flight_start_time[flight_idx] = ts[0]

    @property
    def input_dim(self) -> int:
        return len(self.DYNAMICS_KEYS) + 6 + 4  # 15

    def split_flights_by_arrivals(self, val_frac: float = 0.2) -> tuple[list[int], list[int]]:
        flights_sorted = sorted(self._cache.keys(), key=lambda f: self._flight_start_time[f])
        n_train = max(1, int(round(len(flights_sorted) * (1.0 - val_frac))))
        return flights_sorted[:n_train], flights_sorted[n_train:]

    def fit_normalization_stats(self, flight_indices: list[int]) -> None:
        xs = [self._cache[f][0] for f in flight_indices if f in self._cache]
        ys = [self._cache[f][1] for f in flight_indices if f in self._cache]

        all_x = np.concatenate(xs, axis=0)
        all_y = np.concatenate(ys, axis=0)

        self.x_mean, self.x_std = all_x.mean(0), all_x.std(0) + 1e-8
        self.y_mean, self.y_std = all_y.mean(0), all_y.std(0) + 1e-8

    def _group_by_time(self, flight_indices: list[int]) -> dict[int, list[tuple[int, int]]]:
        # seconds with zero active aircraft never get a key here -> naturally skipped
        groups: dict[int, list[tuple[int, int]]] = {}
        for f in flight_indices:
            _, _, ts = self._cache[f]
            for step_idx, t in enumerate(ts):
                key = int(round(t.timestamp()))
                groups.setdefault(key, []).append((f, step_idx))
        return groups

    def split_by_arrivals(self, val_frac: float = 0.2) -> tuple["MultiAircraftTimestepView", "MultiAircraftTimestepView"]:
        train_flights, val_flights = self.split_flights_by_arrivals(val_frac)
        self.fit_normalization_stats(train_flights)
        return (
            MultiAircraftTimestepView(self, train_flights),
            MultiAircraftTimestepView(self, val_flights),
        )

    def set_normalization_stats(
            self, x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray
    ) -> None:
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std


class MultiAircraftTimestepView(Dataset):
    """
    One split (train OR val) of MultiAircraftLandingActionDataset.

    __getitem__(idx) returns the stacked, normalized (n_t, input_dim) /
    (n_t, 3) tensors of every aircraft (from THIS split's own flights only)
    with a valid transition starting at time_keys[idx]. n_t varies per
    sample -> use collate_fn with a DataLoader.
    """
    def __init__(self, parent: MultiAircraftLandingActionDataset, flight_indices: list[int]) -> None:
        self.parent: MultiAircraftLandingActionDataset = parent
        self.flight_indices: list[int] = list(flight_indices)
        self.groups: dict[int, list[tuple[int, int]]] = parent._group_by_time(self.flight_indices)
        self.time_keys: list[int] = sorted(self.groups.keys())

    def __len__(self) -> int:
        return len(self.time_keys)

    def n_aircraft(self, idx: int) -> int:
        return len(self.groups[self.time_keys[idx]])

    def flight_keys(self, idx: int) -> list[tuple[str, int]]:
        entries = self.groups[self.time_keys[idx]]
        return [self.parent.flight_dataset.flights[f] for f, _ in entries]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if self.parent.x_mean is None:
            raise RuntimeError("Normalization stats not set. Call split_by_arrivals() on the parent first.")

        entries = self.groups[self.time_keys[idx]]
        x_rows = np.stack([self.parent._cache[f][0][s] for f, s in entries])
        y_rows = np.stack([self.parent._cache[f][1][s] for f, s in entries])

        x = (x_rows - self.parent.x_mean) / self.parent.x_std
        y = (y_rows - self.parent.y_mean) / self.parent.y_std
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))

    @staticmethod
    def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor, Tensor]:
        xs, ys, batch_idx = [], [], []
        for i, (x, y) in enumerate(batch):
            xs.append(x)
            ys.append(y)
            batch_idx.append(torch.full((x.shape[0],), i, dtype=torch.long))
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0), torch.cat(batch_idx, dim=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    icao: str = "CYVR"

    base = LazyFlightParquetDataset(f"../../../data/{icao}/train/20260708_1247") # 20260802_0920 # 10000, 20260708_1247 # 2000
    airport: Airport = Airport.load_from_folder(base_path=Path("../../../data/"), icao=icao)

    dataset = MultiAircraftLandingActionDataset(
        base, mode="bspline_state", max_dt=5.0,
        airport_center_latlon=(airport.lat, airport.lon),
        airport_elevation_m=airport.arinc_424_runway[0].runway_altitude * 0.3048,
    )

    print(f"Valid flights: {len(dataset._cache)}")

    train_view, val_view = dataset.split_by_arrivals(val_frac=0.2)
    print(f"Train timesteps: {len(train_view)}   Val timesteps: {len(val_view)}")

    n_ac = [train_view.n_aircraft(i) for i in range(len(train_view))]
    print(f"Aircraft/timestep (train): min={min(n_ac)}  mean={sum(n_ac) / len(n_ac):.2f}  max={max(n_ac)}")

    loader = DataLoader(train_view, batch_size=8, shuffle=True, collate_fn=MultiAircraftTimestepView.collate_fn)
    x, y, batch_idx = next(iter(loader))
    print(f"batch: x={tuple(x.shape)}  y={tuple(y.shape)}  batch_idx={tuple(batch_idx.shape)}  "
          f"n_timesteps_in_batch={batch_idx.max().item() + 1}")

    def plot_time_window(
        view: MultiAircraftTimestepView,
        window_seconds: float = 600.0,
        window_start_ts: float | None = None,
        title: str | None = None,
    ) -> None:
        if len(view) == 0:
            print("[plot_time_window] view is empty, nothing to plot.")
            return

        if window_start_ts is None:
            counts = [view.n_aircraft(i) for i in range(len(view))]
            center_t = view.time_keys[int(np.argmax(counts))]
            t_start = center_t - window_seconds / 2.0
        else:
            t_start = window_start_ts
        t_end = t_start + window_seconds

        flights_in_window: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for f in view.flight_indices:
            X, _, ts = view.parent._cache[f]
            secs = np.array([t.timestamp() for t in ts])
            mask = (secs >= t_start) & (secs <= t_end)
            if mask.any():
                flights_in_window[f] = (X[mask], secs[mask])

        if not flights_in_window:
            print(f"[plot_time_window] no flights found in the {window_seconds:.0f}s window starting at {t_start:.0f}.")
            return

        cmap = plt.get_cmap("tab10")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        for i, (f, (Xw, secs)) in enumerate(flights_in_window.items()):
            icao24, flight_id = view.parent.flight_dataset.flights[f]
            color = cmap(i % 10)
            rel_t = secs - t_start
            xs, ys, hs = Xw[:, 5], Xw[:, 6], Xw[:, 7]  # state_feat = x,y,h,v,gamma,chi -> indices 5..10

            ax1.plot(xs, ys, "-", color=color, linewidth=2, label=f"{icao24} ({flight_id})")
            ax1.scatter(xs[0], ys[0], color=color, marker="o", s=40, zorder=3)
            ax1.scatter(xs[-1], ys[-1], color=color, marker="x", s=50, zorder=3)

            ax2.plot(rel_t, hs, "-", color=color, linewidth=2, label=f"{icao24} ({flight_id})")

        times_window = [view.time_keys[i] - t_start for i in range(len(view))
                        if t_start <= view.time_keys[i] <= t_end]
        counts_window = [view.n_aircraft(i) for i in range(len(view))
                          if t_start <= view.time_keys[i] <= t_end]

        ax1.set_title(f"Ground track ({len(flights_in_window)} aircraft, {window_seconds:.0f}s window)")
        ax1.set_xlabel("x [m, airport-centered]")
        ax1.set_ylabel("y [m, airport-centered]")
        ax1.axis("equal")
        ax1.grid(True)
        ax1.legend(fontsize=7, loc="best")

        ax2.set_title("Altitude over time")
        ax2.set_xlabel("t [s] (relative to window start)")
        ax2.set_ylabel("h [m, above airport ref.]")
        ax2.grid(True)
        ax2.legend(fontsize=7, loc="best")

        ax3.step(times_window, counts_window, where="post", color="tab:gray")
        ax3.set_title("Aircraft in airspace")
        ax3.set_xlabel("t [s] (relative to window start)")
        ax3.set_ylabel("count")
        ax3.grid(True)

        fig.suptitle(title or "Multi-aircraft window")
        fig.tight_layout()
        plt.show()

    plot_time_window(train_view, window_seconds=600.0, title=f"{icao} - multi-aircraft window")