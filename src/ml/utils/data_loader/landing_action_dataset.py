from pathlib import Path
import copy

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from pyproj import Transformer

from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from engine.physics.aircraft_environment_casadi import AircraftEnvironmentCasadi
from engine.physics.trajectory_action_processor import FlightTrajectoryProcessor, TrajectoryMode
from utils.airport import Airport
from engine.utils_project.geo_transformer import GeoTransformer
from utils.fix_identifier import FixIdentifier


class LandingActionDataset(Dataset):
    """
    input  (15): aircraft dynamics (5) + state x/y/h/v/gamma/chi (6)
                 + landing runway x/y/elevation/heading (4)
    output (3) : thrust, load_factor, roll

    All x/y live in a UTM frame centered on the airport reference point,
    i.e. an aircraft over the airport center is at (0, 0). Heights are given
    RELATIVE to a fixed reference elevation (airport_elevation_m): an aircraft
    at runway level has h ~ 0, and fix_elevation_m ~ 0. All runways in this
    project share the same elevation, so one reference is enough.
    Angles chi and runway bearing are in degrees (clockwise from true north).

    NOTE: all flights are projected into the airport's UTM zone
    (target_epsg passed to FlightTrajectoryProcessor), so traj[i].x/y and
    airport_c* are always in the same frame.

    Rotation augmentation rotates x/y about the airport center (clockwise in
    the E/N frame) and adds rotation_deg to chi and runway bearing. The runway
    fix, the FAF AND the IF are rotated the same way. Augmented copies always
    land in the same split as their source flight.

    Approach fixes (FAF + IF): taken from the approach procedure of the landing runway
    (LazyFlightParquetDataset.get_procedure -> Approach.get_final_approach_fixes),
    projected to UTM and centered on the airport like the runway fix. They are
    NOT model inputs, only extra raw info for the CBF-QP layer.
      - FAF: the FAF fix of the final segment
      - IF:  the first entry fix (RNAV GNSS -> exactly one IF, RNP may have several -> take the first)
    Flights without a resolvable FAF or IF are skipped.
    Runway length: taken from the ArincRunway of the same procedure lookup (stored in ft in the data,
    converted to m here), NaN if missing -> the QP layer then falls back to its default (9023 ft FPAP floor).

    __getitem__ returns (x, y):
      provide_raw_features=False: x = normalized input, shape (15,)
      provide_raw_features=True:  x = packed tensor, shape (39,)
        x[0:15]   normalized input (as above)
        x[15:39]  RAW (unnormalized) aux for the CBF-QP safety layer
                  (constraint_layer_cvxpy.py); indices relative to aux = x[15:]:
          aux[0:5]   aircraft_dyn    (mass, C_D0, area, cl_max, k)
          aux[5:7]   aircraft_pos xy (x, y)
          aux[7:10]  aircraft_state  (h, v, gamma)
          aux[10]    aircraft_pos chi
          aux[11:13] runway_pos xy   (fx, fy)
          aux[13]    runway fix elevation (unused by the QP layer)
          aux[14]    runway_pos hdg  (bearing)
          aux[15:19] aircraft_limits (max_thrust, min_load_factor, max_load_factor, max_roll_angle)
          aux[19:21] faf_pos xy      (faf_x, faf_y)   <- extra info, not a model input
          aux[21:23] if_pos xy       (if_x, if_y)     <- extra info, not a model input
          aux[23]    runway length   [m]              <- extra info, not a model input (NaN if unknown)
      y = normalized target (thrust, load_factor, roll), shape (3,)

    Normalization stats must be set before indexing, via flight_level_split(),
    fit_normalization_stats() or set_normalization_stats().
    """

    DYNAMICS_KEYS = (
        "mass", "C_D0", "area", "cl_max", "k",
    )
    LIMIT_KEYS = (
        "max_thrust", "min_load_factor", "max_load_factor", "max_roll_angle",
    )
    FT_TO_M = 0.3048

    def __init__(
        self,
        flight_dataset: LazyFlightParquetDataset,
        mode: TrajectoryMode,
        airport_center_latlon: tuple[float, float],  # note: (lat, lon)
        airport_elevation_m: float,                  # reference height subtracted from h and fix_elevation
        max_dt: float = 5.0,
        augment_rotations_deg: list[float] | None = None,
        provide_raw_features: bool = False,
        cap_data: bool = False,
        approach_index: int = 0,
    ) -> None:
        self.flight_dataset: LazyFlightParquetDataset = flight_dataset
        self.processor: FlightTrajectoryProcessor = FlightTrajectoryProcessor(
            dataset=flight_dataset,
            target_epsg=GeoTransformer.utm_epsg(*airport_center_latlon)
        )
        self.max_dt: float = max_dt
        self.mode: TrajectoryMode = mode
        self.provide_raw_features: bool = provide_raw_features
        self.cap_data: bool = cap_data
        self.approach_index: int = approach_index

        self.augment_rotations_deg: list[float] = list(augment_rotations_deg) if augment_rotations_deg else []

        # single fixed airport-centered UTM frame (shared by all flights + runways + approach fixes)
        self.airport_center_latlon: tuple[float, float] = airport_center_latlon
        self.airport_elevation_m: float = float(airport_elevation_m)
        alat, alon = airport_center_latlon
        self._utm: Transformer = Transformer.from_crs(
            "EPSG:4326", GeoTransformer.utm_epsg(alat, alon), always_xy=True
        )
        self.airport_cx, self.airport_cy = self._utm.transform(alon, alat)

        # runway name -> (faf_x, faf_y, if_x, if_y, rw_x, rw_y, rw_height, runway_length_m, rw_bearing) in the airport-centered frame, UNROTATED
        self._approach_cache: dict[str, tuple[float, float, float, float, float, float, float, float, float]] = {}

        self._cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}  # cache_idx -> (X, Y, L, AP)
        self._source_flight: dict[int, int] = {}                    # cache_idx -> real flight_idx it derives from
        self.samples: list[tuple[int, int]] = []                    # (cache_idx, step_idx)
        self.flight_id_of_sample: list[int] = []

        self._build_index()

        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None

    def _to_airport_frame(self, lon: float, lat: float) -> tuple[float, float]:
        """lon/lat (deg) -> airport-centered UTM (x, y) in meters."""
        x, y = self._utm.transform(float(lon), float(lat))
        return x - self.airport_cx, y - self.airport_cy

    @staticmethod
    def _rotate_xy(x: float, y: float, rotation_deg: float) -> tuple[float, float]:
        theta = np.radians(rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        return x * cos_t + y * sin_t, -x * sin_t + y * cos_t

    def _rotate_trajectory_states(self, traj: list, rotation_deg: float) -> list:
        rotated = []
        for state in traj:
            x_rel, y_rel = self._rotate_xy(
                state.x - self.airport_cx, state.y - self.airport_cy, rotation_deg
            )
            new_state = copy.copy(state)
            new_state.x = x_rel + self.airport_cx
            new_state.y = y_rel + self.airport_cy
            new_state.chi = (state.chi + rotation_deg) % 360.0
            rotated.append(new_state)
        return rotated

    def _runway_features(self, icao24: str, flight_id: int, rotation_deg: float = 0.0) -> np.ndarray:
        row = self.flight_dataset.get_runway_assignment(icao24, flight_id)

        _, _, _, _, fx, fy, f_height_m, _, f_bearing = self._approach_fixes_airport_frame(str(row["landing_runway"]))
        elev_rel = f_height_m - self.airport_elevation_m

        if rotation_deg != 0.0:
            fx, fy = self._rotate_xy(fx, fy, rotation_deg)
            f_bearing = (f_bearing + rotation_deg) % 360.0

        return np.array([fx, fy, elev_rel, f_bearing], dtype=np.float32)

    @staticmethod
    def _fix_latlon(fix: FixIdentifier) -> tuple[float, float]:
        return fix.latitude, fix.longitude

    def _approach_fixes_airport_frame(self, runway_name: str) -> tuple[float, float, float, float, float, float, float, float, float]:
        """runway name -> (faf_x, faf_y, if_x, if_y, runway_length_m) airport-centered, unrotated. Cached per runway."""
        if runway_name in self._approach_cache:
            return self._approach_cache[runway_name]

        base = runway_name[2:] if runway_name.startswith("RW") else runway_name
        last_err: Exception | None = None
        for candidate in (runway_name, base, f"RW{base}"):
            try:
                approach, arinc_runway = self.flight_dataset.get_procedure(candidate, approach_index=self.approach_index)
            except (AttributeError, IndexError) as e:  # runway not found / no approach
                last_err = e
                continue

            entry_fixes, faf_fix, rw_fix = approach.get_final_approach_fixes()
            if faf_fix is None:
                raise ValueError(f"no FAF in approach {approach.procedure_identifier!r} for runway {runway_name!r}")
            if not entry_fixes:
                raise ValueError(f"no IF in approach {approach.procedure_identifier!r} for runway {runway_name!r}")

            # RNAV GNSS -> exactly one IF, RNP may list several -> take the first one
            if_fix = entry_fixes[0]

            faf_lat, faf_lon = self._fix_latlon(faf_fix)
            if_lat, if_lon = self._fix_latlon(if_fix)
            rw_lat, rw_lon = self._fix_latlon(rw_fix)
            faf_x, faf_y = self._to_airport_frame(faf_lon, faf_lat)
            if_x, if_y = self._to_airport_frame(if_lon, if_lat)
            rw_x, rw_y = self._to_airport_frame(rw_lon, rw_lat)

            # runway length is stored in ft in the data -> m, NaN if missing (QP layer falls back to its default)
            length_ft = arinc_runway.length
            runway_length_m = float(length_ft) * self.FT_TO_M if length_ft else float("nan")

            fixes = (faf_x, faf_y, if_x, if_y, rw_x, rw_y, rw_fix.altitude * self.FT_TO_M, runway_length_m, arinc_runway.bearing)
            self._approach_cache[runway_name] = fixes
            return fixes

        raise ValueError(f"no approach found for runway {runway_name!r} ({last_err})")

    def _approach_features(self, icao24: str, flight_id: int, rotation_deg: float = 0.0) -> np.ndarray:
        row = self.flight_dataset.get_runway_assignment(icao24, flight_id)
        faf_x, faf_y, if_x, if_y, rw_x, rw_y, _, runway_length_m, _ = self._approach_fixes_airport_frame(str(row["landing_runway"]))

        if rotation_deg != 0.0:
            faf_x, faf_y = self._rotate_xy(faf_x, faf_y, rotation_deg)
            if_x, if_y = self._rotate_xy(if_x, if_y, rotation_deg)

        return np.array([faf_x, faf_y, if_x, if_y, runway_length_m], dtype=np.float32)

    def _build_flight(
        self, flight_idx: int, rotation_deg: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        traj, dynamics, timestamps = self.processor.load(flight_idx, mode=self.mode)
        icao24, flight_id = self.flight_dataset.flights[flight_idx]

        if rotation_deg != 0.0:
            traj = self._rotate_trajectory_states(traj, rotation_deg)

        dyn_feat = np.array([getattr(dynamics, k) for k in self.DYNAMICS_KEYS], dtype=np.float32)
        limits_feat = np.array([getattr(dynamics, k) for k in self.LIMIT_KEYS], dtype=np.float32)
        rwy_feat = self._runway_features(icao24, flight_id, rotation_deg=rotation_deg)
        appr_feat = self._approach_features(icao24, flight_id, rotation_deg=rotation_deg)

        xs, ys = [], []
        for i in range(len(traj) - 1):
            dt = (timestamps[i + 1] - timestamps[i]).total_seconds()
            if dt <= 0 or dt > self.max_dt:
                continue

            env = AircraftEnvironmentCasadi(state=traj[i], goal=None, dt=dt)
            thrust, load_factor, roll = env.inverse_action(traj[i + 1])

            if self.cap_data:
                thrust, load_factor, roll = float(thrust), float(load_factor), float(roll)
                thrust = float(np.clip(thrust, 0.0, dynamics.max_thrust))
                load_factor = float(np.clip(load_factor, dynamics.min_load_factor, dynamics.max_load_factor))
                roll = float(np.clip(roll, -dynamics.max_roll_angle, dynamics.max_roll_angle))

            state_feat = np.array(
                [traj[i].x - self.airport_cx, traj[i].y - self.airport_cy,
                 traj[i].h - self.airport_elevation_m, traj[i].v, traj[i].gamma, traj[i].chi],
                dtype=np.float32,
            )

            xs.append(np.concatenate([dyn_feat, state_feat, rwy_feat]))
            ys.append([float(thrust), float(load_factor), float(roll)])

        X = np.asarray(xs, dtype=np.float32)
        Y = np.asarray(ys, dtype=np.float32)
        L = np.tile(limits_feat, (len(X), 1))   # constant per flight, tiled row-aligned with X/Y
        AP = np.tile(appr_feat, (len(X), 1))    # constant per flight, tiled row-aligned with X/Y
        return X, Y, L, AP

    def _build_index(self) -> None:
        next_aug_idx = len(self.flight_dataset)

        for flight_idx in range(len(self.flight_dataset)):
            try:
                X, Y, L, AP = self._build_flight(flight_idx)  # rotation_deg=0.0 -> original
            except Exception as e:
                print(f"[WARN] skipping flight {flight_idx}: {e}")
                continue
            if len(X) == 0:
                continue

            self._cache[flight_idx] = (X, Y, L, AP)
            self._source_flight[flight_idx] = flight_idx
            for step_idx in range(len(X)):
                self.samples.append((flight_idx, step_idx))
                self.flight_id_of_sample.append(flight_idx)

            for rotation_deg in self.augment_rotations_deg:
                try:
                    Xa, Ya, La, APa = self._build_flight(flight_idx, rotation_deg=rotation_deg)
                except Exception as e:
                    print(f"[WARN] skipping {rotation_deg}deg augmentation of flight {flight_idx}: {e}")
                    continue
                if len(Xa) == 0:
                    continue

                aug_idx = next_aug_idx
                next_aug_idx += 1

                self._cache[aug_idx] = (Xa, Ya, La, APa)
                self._source_flight[aug_idx] = flight_idx
                for step_idx in range(len(Xa)):
                    self.samples.append((aug_idx, step_idx))
                    self.flight_id_of_sample.append(aug_idx)

    @property
    def input_dim(self) -> int:
        return len(self.DYNAMICS_KEYS) + 6 + 4  # 15

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
        X, Y, L, AP = self._cache[flight_idx]

        x_raw = X[step_idx]
        y_raw = Y[step_idx]
        x = (x_raw - self.x_mean) / self.x_std
        y = (y_raw - self.y_mean) / self.y_std

        if not self.provide_raw_features:
            return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))

        packed = np.concatenate([x, x_raw, L[step_idx], AP[step_idx]]).astype(np.float32)  # (15+15+4+5,) = (39,)
        return torch.from_numpy(packed), torch.from_numpy(y.astype(np.float32))

    def flight_level_split(self, val_frac: float = 0.2, seed: int = 42) -> tuple[Subset, Subset]:
        # split at the ORIGINAL-flight level so augmented siblings always
        # follow their source flight into the same split
        rng = np.random.default_rng(seed)
        real_flight_indices = np.array(sorted(set(self._source_flight.values())))
        rng.shuffle(real_flight_indices)

        n_val = max(1, int(len(real_flight_indices) * val_frac))
        val_roots = set(real_flight_indices[:n_val].tolist())

        val_flights = {f for f, root in self._source_flight.items() if root in val_roots}
        train_flights = [f for f in self._cache.keys() if f not in val_flights]

        self.fit_normalization_stats(train_flights)

        train_idx = [i for i, f in enumerate(self.flight_id_of_sample) if f not in val_flights]
        val_idx = [i for i, f in enumerate(self.flight_id_of_sample) if f in val_flights]
        return Subset(self, train_idx), Subset(self, val_idx)

    def split_flights(self, val_frac: float = 0.2, seed: int = 42) -> tuple[list[int], list[int]]:
        rng = np.random.default_rng(seed)
        real_flight_indices = np.array(sorted(set(self._source_flight.values())))
        rng.shuffle(real_flight_indices)

        n_val = max(1, int(len(real_flight_indices) * val_frac))
        val_roots = set(real_flight_indices[:n_val].tolist())

        train_flights = [f for f, root in self._source_flight.items() if root not in val_roots]
        val_flights = [f for f, root in self._source_flight.items() if root in val_roots]
        return train_flights, val_flights

    def indices_for_flights(self, flights: set[int]) -> list[int]:
        return [i for i, f in enumerate(self.flight_id_of_sample) if f in flights]

    def set_normalization_stats(
            self, x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray
    ) -> None:
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    icao: str = "CYVR"

    base = LazyFlightParquetDataset(f"../../../data/{icao}/test/20260709_1207")

    airport: Airport = Airport.load_from_folder(base_path=Path("../../../data/"), icao=icao)

    dataset = LandingActionDataset(
        base, mode="bspline_state", max_dt=5.0,
        airport_center_latlon=(airport.lat, airport.lon),
        airport_elevation_m=airport.get_mean_runway_elevation_ft() * 0.3048,
    )

    print(f"Loaded {len(dataset)} transitions")
    print(f"Valid flights: {len(dataset._cache)}")
    print(f"Airport center UTM: ({dataset.airport_cx:.1f}, {dataset.airport_cy:.1f})")
    print(f"Airport elevation ref: {dataset.airport_elevation_m:.1f} m")
    print(f"Approach cache (runway -> faf x/y, if x/y, runway length [m]): {dataset._approach_cache}")

    fi = next(iter(dataset._cache))
    X, _, _, AP = dataset._cache[fi]
    print("example state x/y/h [m]:", X[0, 5], X[0, 6], X[0, 7])  # 5,6,7 = state x,y,h
    print("example FAF x/y [m]:", AP[0, 0], AP[0, 1])
    print("example IF  x/y [m]:", AP[0, 2], AP[0, 3])
    print("example runway length [m]:", AP[0, 4])

    def _heading_vector(chi_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chi = np.radians(chi_deg)
        return np.sin(chi), np.cos(chi)

    def plot_trajectory_with_headings(ax, X: np.ndarray, AP: np.ndarray, title: str, step: int = 10) -> None:
        xs, ys, chis = X[:, 5], X[:, 6], X[:, 10]
        fx, fy, bearing = float(X[0, 11]), float(X[0, 12]), float(X[0, 14])
        faf_x, faf_y, if_x, if_y = (float(v) for v in AP[0, :4])

        idx = np.arange(0, len(X), step)
        ax.scatter(xs[idx], ys[idx], s=15, c="tab:blue", label="trajectory (every 10th)", zorder=3)

        extent = max(xs.max() - xs.min(), ys.max() - ys.min(), 1.0)
        arrow_len = 0.04 * extent

        dx, dy = _heading_vector(chis[idx])
        ax.quiver(xs[idx], ys[idx], dx, dy, angles="xy", scale_units="xy",
                  scale=1 / arrow_len, color="tab:blue", width=0.004, label="heading (chi)")

        ax.plot(fx, fy, "kx", markersize=14, markeredgewidth=3, label="runway fix", zorder=4)
        ax.plot(faf_x, faf_y, "g^", markersize=12, label="FAF", zorder=4)
        ax.plot(if_x, if_y, "ms", markersize=10, label="IF", zorder=4)
        ax.plot([if_x, faf_x, fx], [if_y, faf_y, fy], "g--", linewidth=1.5, label="IF -> FAF -> fix", zorder=2)
        rdx, rdy = _heading_vector(np.array([bearing]))
        ax.quiver([fx], [fy], rdx, rdy, angles="xy", scale_units="xy",
                  scale=1 / (2 * arrow_len), color="tab:red", width=0.006, label="runway bearing")

        ax.set_title(title)
        ax.set_xlabel("x [m, airport-centered]"); ax.set_ylabel("y [m, airport-centered]")
        ax.axis("equal"); ax.grid(True); ax.legend(fontsize=8, loc="best")

    demo_flight_idx = fi
    X_orig, _, _, AP_orig = dataset._cache[demo_flight_idx]
    X_aug, _, _, AP_aug = dataset._build_flight(demo_flight_idx, rotation_deg=180.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    plot_trajectory_with_headings(axes[0], X_orig, AP_orig, "Original")
    plot_trajectory_with_headings(axes[1], X_aug, AP_aug, "Augmented (180 deg)")
    fig.suptitle(f"{icao}  flight index {demo_flight_idx}")
    fig.tight_layout()
    plt.show()