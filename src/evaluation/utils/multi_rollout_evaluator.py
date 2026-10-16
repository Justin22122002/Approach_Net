import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from typing_extensions import override

from evaluation.utils.rollout_evaluator import ActionRollout, AIRPORTS, DATASET_NAME, FT_TO_M
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from ml.utils.data_loader.multi_aircraft_landing_action_dataset import MultiAircraftLandingActionDataset
from ml.utils.models.constraint_graph_attention_model import MultiAircraftConstrainedQPLayer
from ml.utils.models.graph_attention_model import MultiAircraftLandingGNN, infer_encoder_type
from engine.physics.aircraft_dynamics import AircraftDynamics
from engine.physics.aircraft_environment_casadi import AircraftEnvironmentCasadi
from engine.physics.aircraft_kinematics_state_casadi import AircraftKinematicsStateCasadi
from engine.physics.batch_result_store import append_trajectory, append_status
from utils.airport import Airport


@dataclass
class SceneAircraft:
    icao24: str
    flight_id: int
    dyn_feat: np.ndarray       # (5,)  mass, C_D0, area, cl_max, k
    rwy_feat: np.ndarray       # (4,)  fix x/y, height rel. incl. TCH, bearing
    limits: np.ndarray         # (4,)  max_thrust, min/max load factor, max roll
    approach: np.ndarray       # (5,)  faf x/y, if x/y, runway length [m]
    dynamics: AircraftDynamics
    state: AircraftKinematicsStateCasadi
    env: AircraftEnvironmentCasadi
    rows: list[dict] = field(default_factory=list)
    wall_s: float = 0.0        # share of the scene compute time


class MultiAircraftRollout(ActionRollout):
    name = "gnn"

    def __init__(
        self,
        result_dir: str | Path,
        constrained: bool = False,
        use_separation: bool = True,
        qp_kwargs: dict | None = None,
        solver: str | None = None,
        solver_args: dict | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        best_model_name: str | None = None,
        encoder_type: str | None = None,
        **kwargs,
    ) -> None:
        best_model_name = best_model_name or ("best_base_model.pth" if constrained else "best_model.pth")
        if encoder_type is None:
            sd = torch.load(Path(result_dir) / best_model_name, map_location="cpu")
            encoder_type = infer_encoder_type(sd)
        self.encoder_type: str = encoder_type  # must be set before super().__init__ -> create_model()
        print(f"[{self.name}] encoder_type={self.encoder_type}")

        super().__init__(result_dir, constrained=constrained, solver=solver, best_model_name=best_model_name, **kwargs)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.use_separation = use_separation
        self.qp_kwargs = qp_kwargs or {}
        self.solver_args = solver_args or {}

        self.model.to(self.device, self.dtype)
        self.safety_layer = self.build_safety_layer().to(self.device, self.dtype) if constrained else None
        if self.safety_layer is not None:
            self.safety_layer.eval()

    @override
    def create_model(self, input_dim: int, output_dim: int) -> nn.Module:
        return MultiAircraftLandingGNN(input_dim=input_dim, action_dim=output_dim, encoder_type=self.encoder_type)

    @override
    def build_safety_layer(self) -> nn.Module:
        return MultiAircraftConstrainedQPLayer(
            base_model=self.model,
            y_mean=self.y_mean,
            y_std=self.y_std,
            solver=self.solver,
            use_separation=self.use_separation,
            solver_args=self.solver_args,
            separation_kwargs={"solver_args": self.solver_args},
            debug=False,
            **self.qp_kwargs,
        )

    def _t(self, arr: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(arr), dtype=self.dtype, device=self.device)

    def predict_scene(self, active: list[SceneAircraft]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_raw = np.stack([np.concatenate([a.dyn_feat, self.state_feat(a.state), a.rwy_feat]) for a in active])
        x_norm = self._t((x_raw - self.x_mean) / self.x_std)
        batch_idx = torch.zeros(len(active), dtype=torch.long, device=self.device)

        with torch.no_grad():
            if self.safety_layer is None:
                raw_norm = self.model(x_norm, batch_idx)
                out = raw_norm * self._t(self.y_std) + self._t(self.y_mean)
                sep_active = np.zeros(len(active), dtype=bool)
            else:
                xr = self._t(x_raw)
                ap = self._t(np.stack([a.approach for a in active]))
                out = self.safety_layer(
                    x_norm,
                    aircraft_state=xr[:, 7:10],
                    aircraft_dyn=xr[:, 0:5],
                    aircraft_limits=self._t(np.stack([a.limits for a in active])),
                    aircraft_pos=xr[:, [5, 6, 10]],
                    runway_pos=xr[:, [11, 12, 14]],
                    batch_idx=batch_idx,
                    faf_pos=ap[:, 0:2],
                    if_pos=ap[:, 2:4],
                    runway_length_m=ap[:, 4],
                    h_thr=xr[:, 13],                 # runway fix height incl. TCH, centered
                )
                raw_norm = self.safety_layer.last_raw_norm
                mask = self.safety_layer.separation.last_active_mask if self.use_separation else None
                sep_active = mask.cpu().numpy() if mask is not None else np.zeros(len(active), dtype=bool)

        raw = raw_norm.cpu().numpy() * self.y_std + self.y_mean
        return out.cpu().numpy(), raw, sep_active

    def spawn(self, ds: MultiAircraftLandingActionDataset, idx: int) -> SceneAircraft:
        X, _, L, AP, _ = ds._cache[idx]
        flights = ds.flight_dataset
        icao24, flight_id = flights.flights[idx]
        _, dynamics = flights.get_aircraft(icao24)
        start_point, _ = flights.get_start_and_end_points(flights[idx])
        state = self.build_start_state(start_point, dynamics, self.to_utm)
        self.reset(state)
        return SceneAircraft(icao24, flight_id, X[0, 0:5], X[0, 11:15], L[0], AP[0], dynamics, state, self.env)

    @staticmethod
    def nearest_neighbour(active: list[SceneAircraft]) -> tuple[np.ndarray, np.ndarray]:
        p = np.array([[float(a.state.x), float(a.state.y), float(a.state.h)] for a in active])
        if len(p) < 2:
            return np.full(len(p), np.nan), np.full(len(p), np.nan)
        d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
        np.fill_diagonal(d, np.inf)
        j = d.argmin(axis=1)
        return d[np.arange(len(p)), j], np.abs(p[:, 2] - p[j, 2])

    def finish(self, a: SceneAircraft, landed: bool, traj_path: Path, status_path: Path) -> None:
        if landed:
            n = len(a.rows)
            a.rows.append(self.row(a.icao24, a.flight_id, n, n * self.dt, a.state))
        df = pd.DataFrame(a.rows)
        append_trajectory(df, traj_path)
        t_end = float(df["time_s"].iloc[-1])
        append_status(status_path, {
            "icao24": a.icao24,
            "flight_id": a.flight_id,
            "success": landed,
            "T_opt": t_end,
            "solve_time_s": a.wall_s,
            "error": None if landed else "did not reach runway elevation within max_steps",
        })
        print(f"  [{'OK' if landed else 'WARN'}] {a.icao24} / {a.flight_id}: {'' if landed else 'not landed, '}T={t_end:.1f} s")

    @override
    def run(self, ds: MultiAircraftLandingActionDataset, output_dir: Path) -> None:
        self.to_utm, self.cx, self.cy, self.airport_elev = ds._utm, ds.airport_cx, ds.airport_cy, ds.airport_elevation_m

        output_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{self.name}{'_constrained' if self.constrained else ''}_rollout_{self.run_id}"
        traj_path, status_path = output_dir / f"{tag}.parquet", output_dir / f"{tag}_status.jsonl"
        for p in (traj_path, status_path):
            p.unlink(missing_ok=True)  # joint scene -> always a fresh run

        flights = ds.flight_dataset.flights
        for idx, (icao24, flight_id) in enumerate(flights):
            if idx not in ds._cache:
                append_status(status_path, {
                    "icao24": icao24,
                    "flight_id": flight_id,
                    "success": False,
                    "T_opt": None,
                    "error": "flight skipped by MultiAircraftLandingActionDataset (no approach / no dynamic model data)"
                })

        start_of = {i: ds._flight_start_time[i].timestamp() for i in ds._cache}
        queue = sorted(start_of, key=start_of.get)
        active: list[SceneAircraft] = []
        t = start_of[queue[0]] if queue else 0.0
        n_done = 0
        print(f"\n[{tag}] {len(queue)}/{len(flights)} flights, device={self.device}, solver={self.solver}")

        while queue or active:
            if not active:
                t = max(t, start_of[queue[0]])
            while queue and start_of[queue[0]] <= t:
                idx = queue.pop(0)
                try:
                    active.append(self.spawn(ds, idx))
                except Exception as e:
                    icao24, flight_id = flights[idx]
                    print(f"  [FAIL] {icao24} / {flight_id}: {e}")
                    append_status(status_path, {
                        "icao24": icao24,
                        "flight_id": flight_id,
                        "success": False,
                        "T_opt": None,
                        "error": str(e)
                    })
            if not active:
                continue

            t0 = time.perf_counter()
            out, raw, sep_active = self.predict_scene(active)
            wall = (time.perf_counter() - t0) / len(active)
            dist_h, dh = self.nearest_neighbour(active)

            still: list[SceneAircraft] = []
            for i, a in enumerate(active):
                step = len(a.rows)
                row = self.row(a.icao24, a.flight_id, step, step * self.dt, a.state)
                row.update({
                    "t_sim": t,
                    "n_scene": len(active),
                    "min_dist_h_m": float(dist_h[i]),
                    "dh_nearest_m": float(dh[i]),
                    "sep_active": bool(sep_active[i]),
                    # note those values are the physical values before the QP layer
                    "thrust_raw": float(raw[i, 0]),
                    "load_factor_raw": float(raw[i, 1]),
                    "roll_raw": float(raw[i, 2]),
                })
                self.env, self.dynamics = a.env, a.dynamics
                row.update(self.step(a.state, out[i]))
                a.rows.append(row)
                a.wall_s += wall

                if self.is_done(a.state) or len(a.rows) >= self.max_steps:
                    self.finish(a, self.is_done(a.state), traj_path, status_path)
                    n_done += 1
                else:
                    still.append(a)
            active = still
            t += self.dt

        print(f"\n[DONE] {tag}: {n_done} flights flown")
        print(f"  Trajectory: {traj_path}")
        print(f"  Status-Log: {status_path}")

    @override
    def run_airports(
        self,
        output_folder: str,
        airports: list[str] = AIRPORTS,
        dataset_name: str = DATASET_NAME,
        data_root: str = "../data",
    ) -> None:
        for icao in airports:
            airport = Airport.load_from_folder(base_path=Path(data_root), icao=icao)
            ds = MultiAircraftLandingActionDataset(
                LazyFlightParquetDataset(f"{data_root}/{icao}/test/{dataset_name}"),
                mode=self.mode,
                airport_center_latlon=(airport.lat, airport.lon),
                airport_elevation_m=airport.get_mean_runway_elevation_ft() * FT_TO_M,
            )
            self.run(ds, Path(data_root) / icao / output_folder)