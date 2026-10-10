import time
from abc import abstractmethod, ABCMeta, ABC
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas import Series
from pyproj import Transformer
from torch import nn, Tensor
from typing_extensions import override

from ml.utils.data_loader.landing_action_dataset import LandingActionDataset
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from ml.utils.models.constraint_layer_cvxpy import AircraftCombinedQPLayer
from ml.utils.models.constraint_next_state_layer_cvxpy import ConstraintNextStateLayer
from engine.physics.aircraft_dynamics import AircraftDynamics
from engine.physics.aircraft_environment_casadi import AircraftEnvironmentCasadi
from engine.physics.aircraft_kinematics_state_casadi import AircraftKinematicsStateCasadi
from engine.physics.batch_result_store import already_processed, append_trajectory, append_status
from utils.airport import Airport

# todo: add round filter for starting positions
FT_TO_M = 0.3048
AIRPORTS = ["CYVR", "CYUL", "CYYZ"]  # "CYYC", "CYEG"
DATASET_NAME = "20261005_1406"       # 20261005_1406 -> new test set, 20260709_1207 -> old test set


def load_norm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[0].astype(np.float32), arr[1].astype(np.float32)


def _t(*values: float) -> Tensor:
    """scalars -> (1, n) float tensor"""
    return torch.tensor([values], dtype=torch.float32)


class RolloutEvaluator(metaclass=ABCMeta):
    name: str = "model"
    output_dim: int = 0

    def __init__(
        self,
        result_dir: str | Path,
        constrained: bool = False,
        dt: float = 1.0,
        max_steps: int = 2000,
        resume: bool = True,
        mode: str = "bspline_state",
        best_model_name: str = "best_model.pth",
        solver: None | str = None,
    ) -> None:
        result_dir: str | Path = Path(result_dir)
        self.constrained, self.dt, self.max_steps, self.resume, self.mode = constrained, dt, max_steps, resume, mode

        self.x_mean, self.x_std = load_norm(result_dir / "normalization_x.csv")
        self.y_mean, self.y_std = load_norm(result_dir / "normalization_y.csv")

        self.model: nn.Module = self.create_model(len(self.x_mean), self.output_dim)
        self.model.load_state_dict(torch.load(result_dir / best_model_name, map_location="cpu"))
        self.model.eval()
        self.safety_layer: nn.Module | None = None
        self.run_id: str = result_dir.name
        self.solver: None | str = solver

    @abstractmethod
    def create_model(self, input_dim: int, output_dim: int) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def build_safety_layer(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def safe_predict(self, x_norm: Tensor, state_feat: np.ndarray) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def step(self, state: AircraftKinematicsStateCasadi, pred: np.ndarray) -> dict:
        raise NotImplementedError

    def reset(self, state: AircraftKinematicsStateCasadi) -> None:
        pass

    def is_done(self, state: AircraftKinematicsStateCasadi) -> bool:
        return float(state.h) <= self.airport_elev - 50

    def model_input(self, state_feat: np.ndarray) -> Tensor:
        x = np.concatenate([self.dyn_feat, state_feat, self.rwy_feat])
        assert x.shape[0] == self.x_mean.shape[0], f"feature dim {x.shape[0]} != normalization dim {self.x_mean.shape[0]}"
        return torch.from_numpy(((x - self.x_mean) / self.x_std).astype(np.float32)).unsqueeze(0)

    def predict(self, state_feat: np.ndarray) -> np.ndarray:
        x_norm = self.model_input(state_feat)
        with torch.no_grad():
            if self.safety_layer is not None:
                return self.safe_predict(x_norm, state_feat).squeeze(0).detach().numpy()
            return self.model(x_norm).squeeze(0).numpy() * self.y_std + self.y_mean

    def state_feat(self, state: AircraftKinematicsStateCasadi) -> np.ndarray:
        return np.array(
            [float(state.x) - self.cx, float(state.y) - self.cy, float(state.h) - self.airport_elev,
             float(state.v), float(state.gamma), float(state.chi)], dtype=np.float32)

    def setup_flight(self, ds: LandingActionDataset, idx: int) -> None:
        X, _, L, AP = ds._cache[idx]
        self.dyn_feat = X[0, 0:5]                           # mass, C_D0, area, cl_max, k
        self.rwy_feat = X[0, 11:15]                         # fix x/y, elevation rel, bearing
        self.limits = L[0]                                  # max_thrust, min/max load factor, max roll
        self.approach = AP[0]                               # faf x/y, if x/y, runway length [m]
        self.h_thr = float(self.rwy_feat[2])                # runway fix height incl. TCH, centered (= x_raw[:, 13] in training)
        self.safety_layer = self.build_safety_layer() if self.constrained else None

    @staticmethod
    def row(icao24: str, flight_id: int, step: int, time_s: float, state) -> dict:
        return {
            "icao24": icao24,
            "flight_id": flight_id,
            "step": step,
            "time_s": time_s,
            "x": float(state.x),
            "y": float(state.y),
            "h": float(state.h),
            "v": float(state.v),
            "chi": np.radians(float(state.chi)),
            "gamma": np.radians(float(state.gamma))
        }

    def rollout_flight(self, ds: LandingActionDataset, idx: int) -> tuple[list[dict], bool, float]:
        if idx not in ds._cache:
            raise ValueError("flight skipped by LandingActionDataset (no approach / no dynamic model data)")
        flights = ds.flight_dataset
        icao24, flight_id = flights.flights[idx]

        self.setup_flight(ds, idx)
        _, self.dynamics = flights.get_aircraft(icao24)
        start_point, _ = flights.get_start_and_end_points(flights[idx])
        state = self.build_start_state(start_point, self.dynamics, self.to_utm)
        self.reset(state)

        rows: list[dict] = []
        t0 = time.perf_counter()
        for step in range(self.max_steps):
            pred = self.predict(self.state_feat(state))
            row = self.row(icao24, flight_id, step, step * self.dt, state)
            row.update(self.step(state, pred))
            rows.append(row)

            if self.is_done(state):
                rows.append(self.row(icao24, flight_id, step + 1, (step + 1) * self.dt, state))
                return rows, True, time.perf_counter() - t0
        return rows, False, time.perf_counter() - t0

    def run(self, ds: LandingActionDataset, output_dir: Path) -> None:
        self.to_utm, self.cx, self.cy, self.airport_elev = ds._utm, ds.airport_cx, ds.airport_cy, ds.airport_elevation_m

        output_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{self.name}{'_constrained' if self.constrained else ''}_rollout_{self.run_id}"
        traj_path, status_path = output_dir / f"{tag}.parquet", output_dir / f"{tag}_status.jsonl"

        flights = ds.flight_dataset.flights
        n_ok, n_total = 0, len(flights)
        for idx, (icao24, flight_id) in enumerate(flights):
            print(f"\n[{idx + 1}/{n_total}] {tag} {icao24} / flight_id={flight_id}")

            if self.resume and already_processed(status_path, icao24, flight_id):
                print("  [SKIP] already processed (resume=True)")
                n_ok += 1
                continue

            try:
                rows, landed, solve_time_s = self.rollout_flight(ds, idx)
                df_traj = pd.DataFrame(rows)
                append_trajectory(df_traj, traj_path)
                t_end = float(df_traj["time_s"].iloc[-1])
                append_status(status_path, {
                    "icao24": icao24,
                    "flight_id": flight_id,
                    "success": landed,
                    "T_opt": t_end,
                    "solve_time_s": solve_time_s,
                    "error": None if landed else "did not reach runway elevation within max_steps",
                })
                n_ok += landed
                print(f"  [{'OK' if landed else 'WARN'}] {'' if landed else 'not landed, '}T={t_end:.1f} s (saved)")
            except Exception as e:
                print(f"  [FAIL] {e}")
                append_status(status_path, {
                    "icao24": icao24,
                    "flight_id": flight_id,
                    "success": False,
                    "T_opt": None,
                    "error": str(e),
                })

        print(f"\n[DONE] {tag}: {n_ok}/{n_total} landed")
        print(f"  Trajectory: {traj_path}")
        print(f"  Status-Log:   {status_path}")

    def run_airports(
        self,
        output_folder: str,
        airports: list[str] = AIRPORTS,
        dataset_name: str = DATASET_NAME,
        data_root: str = "../data"
    ) -> None:
        for icao in airports:
            airport = Airport.load_from_folder(base_path=Path(data_root), icao=icao)
            ds = LandingActionDataset(
                LazyFlightParquetDataset(f"{data_root}/{icao}/test/{dataset_name}"),
                mode=self.mode,
                airport_center_latlon=(airport.lat, airport.lon),
                airport_elevation_m=airport.get_mean_runway_elevation_ft() * FT_TO_M,
            )
            self.run(ds, Path(data_root) / icao / output_folder)

    @staticmethod
    def build_start_state(
        start_point: Series,
        aircraft_dynamics: AircraftDynamics,
        transformer: Transformer,
    ) -> AircraftKinematicsStateCasadi:
        x0, y0 = transformer.transform(float(start_point["lon"]), float(start_point["lat"]))

        v0 = float(start_point["velocity"])
        h0 = float(start_point["geoaltitude"])

        heading0 = float(start_point["heading"]) if pd.notna(start_point["heading"]) else 0.0
        chi0 = heading0

        vertrate0 = float(start_point["vertrate"]) if pd.notna(start_point["vertrate"]) else 0.0
        gamma0 = float(np.degrees(np.arctan2(vertrate0, max(v0, 1e-3))))

        return AircraftKinematicsStateCasadi(
            x=x0,
            y=y0,
            h=h0,
            v=v0,
            gamma=gamma0,
            chi=chi0,
            aircraft_type=aircraft_dynamics,
        )


class ActionRollout(RolloutEvaluator, metaclass=ABCMeta):
    output_dim = 3

    @override
    def reset(self, state) -> None:
        self.env = AircraftEnvironmentCasadi(state=state, goal=None, dt=self.dt)

    @override
    def build_safety_layer(self) -> nn.Module:
        return AircraftCombinedQPLayer(
            base_model=self.model,
            y_mean=self.y_mean,
            y_std=self.y_std,
            debug=False,
            solver=self.solver,
        )

    @override
    def safe_predict(self, x_norm: Tensor, state_feat: np.ndarray) -> Tensor:
        x, y, h_rel, v, gamma, chi = state_feat.tolist()
        fx, fy, _, bearing = self.rwy_feat.tolist()
        faf_x, faf_y, if_x, if_y, rw_len = self.approach.tolist()
        return self.safety_layer(
            x_norm,
            aircraft_state=_t(h_rel, v, gamma),
            aircraft_dyn=torch.from_numpy(self.dyn_feat).unsqueeze(0),
            aircraft_limits=torch.from_numpy(self.limits).unsqueeze(0),
            aircraft_pos=_t(x, y, chi),
            runway_pos=_t(fx, fy, bearing),
            faf_pos=_t(faf_x, faf_y),
            if_pos=_t(if_x, if_y),
            runway_length_m=torch.tensor([rw_len], dtype=torch.float32),
            h_thr=self.h_thr
        )

    @override
    def step(self, state, pred: np.ndarray) -> dict:
        d = self.dynamics
        t = float(np.clip(pred[0], 0.0, d.max_thrust))
        n = float(np.clip(pred[1], d.min_load_factor, d.max_load_factor))
        phi = float(np.clip(pred[2], -d.max_roll_angle, d.max_roll_angle))
        self.env.apply_action(t, n, phi)
        return {"thrust": t, "load_factor": n, "roll": phi}


class NextStateRollout(RolloutEvaluator, metaclass=ABCMeta):
    output_dim = 6

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.was_before_fix: bool = False

    @override
    def build_safety_layer(self) -> nn.Module:
        return ConstraintNextStateLayer(
            base_model=self.model,
            y_mean=self.y_mean,
            y_std=self.y_std,
            debug=False,
            solver=self.solver,
        )

    @override
    def safe_predict(self, x_norm: Tensor, state_feat: np.ndarray) -> Tensor:
        fx, fy, _, bearing = self.rwy_feat.tolist()
        faf_x, faf_y, if_x, if_y, rw_len = self.approach.tolist()
        return self.safety_layer(
            x_norm,
            runway_pos=_t(fx, fy, bearing),
            faf_pos=_t(faf_x, faf_y),                                       # funnel active below FAF distance
            if_pos=_t(if_x, if_y),                                          # 30 deg intercept active below IF distance
            runway_length_m=torch.tensor([rw_len], dtype=torch.float32),      # LPV lateral geometry (GARP/LGSA)
            h_thr=self.h_thr
        )

    @override
    def step(self, state, pred: np.ndarray) -> dict:
        nx, ny, nh_rel, nv, ngamma, nchi = pred.tolist()
        state.x, state.y, state.h = nx + self.cx, ny + self.cy, nh_rel + self.airport_elev
        state.v, state.gamma, state.chi = nv, ngamma, nchi
        return {}

    @override
    def reset(self, state: AircraftKinematicsStateCasadi) -> None:
        self.was_before_fix = False

    @override
    def is_done(self, state: AircraftKinematicsStateCasadi, overshoot_m: float = 500.0, max_lateral_m: float = 3000.0) -> bool:
        x, y = float(state.x) - self.cx, float(state.y) - self.cy
        fx, fy, _, bearing = self.rwy_feat.tolist()
        b = np.radians(bearing)
        s = -((x - fx) * np.sin(b) + (y - fy) * np.cos(b))  # distance to the runway fix along the landing axis
        e_y = (x - fx) * np.cos(b) - (y - fy) * np.sin(b)   # lateral offset to the centerline
        self.was_before_fix |= s > 0.0
        passed = self.was_before_fix and s < -overshoot_m and abs(e_y) < max_lateral_m
        return super().is_done(state) or passed