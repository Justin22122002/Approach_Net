from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pyproj import Transformer

from physics.aircraft_dynamics import AircraftDynamics
from physics.aircraft_kinematics_state_casadi import AircraftKinematicsStateCasadi
from physics.batch_result_store import load_status_log
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset


@dataclass
class FlightEvaluation:
    icao24: str
    flight_id: int
    converged: bool
    lateral_dev_m: float | None
    vertical_dev_m: float | None
    angle_dev_deg: float | None
    gamma_dev_deg: float | None
    cpa_step: int | None
    success: bool
    velocity_dev_ms: float | None = None
    approach_speed_ref_ms: float | None = None


class TrajectorySuccessEvaluator:
    def __init__(
        self,
        lateral_tol_m: float = 15.0,
        vertical_tol_up_m: float = 20.0,
        vertical_tol_down_m: float | None = None,
        angle_tol_deg: float = 10.0,
    ) -> None:
        self.lateral_tol_m: float = lateral_tol_m
        self.vertical_tol_up_m: float = vertical_tol_up_m
        self.vertical_tol_down_m: float = (
            vertical_tol_down_m if vertical_tol_down_m is not None else vertical_tol_up_m
        )
        self.angle_tol_deg: float = angle_tol_deg

        self.APPROACH_SPEED_FACTOR: float = 1.3
        self.STANDARD_GLIDE_PATH_DEG: float = -3.0

    @staticmethod
    def _project_goal(
        transformer: Transformer,
        runway_row: Series,
    ) -> tuple[float, float, float, float]:
        goal_x, goal_y = transformer.transform(
            float(runway_row["fix_longitude"]),
            float(runway_row["fix_latitude"]),
        )
        goal_h = float(runway_row["fix_elevation_m"])
        heading_goal_deg = float(runway_row["runway_bearing"])
        return goal_x, goal_y, goal_h, heading_goal_deg

    def evaluate_flight(
        self,
        df_traj_flight: DataFrame,
        converged: bool,
        runway_row: Series,
        aircraft_dynamics: AircraftDynamics,
        transformer: Transformer,
    ) -> FlightEvaluation:

        icao24 = str(df_traj_flight["icao24"].iloc[0])
        flight_id = int(df_traj_flight["flight_id"].iloc[0])

        df_sorted = df_traj_flight.sort_values("step")
        steps = df_sorted["step"].to_numpy()
        xs = df_sorted["x"].to_numpy()
        ys = df_sorted["y"].to_numpy()
        hs = df_sorted["h"].to_numpy()
        vs = df_sorted["v"].to_numpy()
        chis = df_sorted["chi"].to_numpy()      # rad
        gammas = df_sorted["gamma"].to_numpy()   # rad

        goal_x, goal_y, goal_h, heading_goal_deg = self._project_goal(transformer, runway_row)

        dx = xs - goal_x
        dy = ys - goal_y
        dh = hs - goal_h

        dist3d = np.sqrt(dx**2 + dy**2 + dh**2)
        idx = int(np.argmin(dist3d))

        bearing_rad = np.radians(heading_goal_deg)
        lateral_dev_m = float(abs(dx[idx] * np.cos(bearing_rad) - dy[idx] * np.sin(bearing_rad)))
        vertical_dev_m = float(dh[idx])

        e_chi = np.arctan2(
            np.sin(chis[idx] - bearing_rad),
            np.cos(chis[idx] - bearing_rad),
        )
        angle_dev_deg = float(abs(np.degrees(e_chi)))

        glide_goal_rad = np.radians(self.STANDARD_GLIDE_PATH_DEG)
        e_gam = np.arctan2(
            np.sin(gammas[idx] - glide_goal_rad),
            np.cos(gammas[idx] - glide_goal_rad),
        )
        gamma_dev_deg = float(abs(np.degrees(e_gam)))

        if vertical_dev_m >= 0:
            vertical_ok = vertical_dev_m <= self.vertical_tol_up_m
        else:
            vertical_ok = abs(vertical_dev_m) <= self.vertical_tol_down_m

        geometry_ok = (
            lateral_dev_m <= self.lateral_tol_m
            and vertical_ok
            and angle_dev_deg <= self.angle_tol_deg
        )

        success = converged and geometry_ok

        velocity_dev_ms: float | None = None
        approach_speed_ref_ms: float | None = None

        if success:
            dummy_state = AircraftKinematicsStateCasadi(
                x=0.0, y=0.0, h=goal_h, v=0.0, gamma=0.0, chi=0.0,
                aircraft_type=aircraft_dynamics,
            )
            v_stall = float(dummy_state.stall_speed(goal_h))
            approach_speed_ref_ms = v_stall * self.APPROACH_SPEED_FACTOR
            velocity_dev_ms = float(vs[idx] - approach_speed_ref_ms)

        return FlightEvaluation(
            icao24=icao24,
            flight_id=flight_id,
            converged=converged,
            lateral_dev_m=lateral_dev_m,
            vertical_dev_m=vertical_dev_m,
            angle_dev_deg=angle_dev_deg,
            gamma_dev_deg=gamma_dev_deg,
            cpa_step=int(steps[idx]),
            success=success,
            velocity_dev_ms=velocity_dev_ms,
            approach_speed_ref_ms=approach_speed_ref_ms,
        )

    def evaluate_batch(
        self,
        trajectories_path: Path,
        status_path: Path,
        dataset: LazyFlightParquetDataset,
        transformer: Transformer,
    ) -> DataFrame:
        df_traj = pd.read_parquet(trajectories_path)
        df_status = load_status_log(status_path)

        if df_status.empty:
            raise ValueError(f"Status log {status_path} ist leer oder existiert nicht.")

        results: list[FlightEvaluation] = []

        for (icao24, flight_id), group in df_traj.groupby(["icao24", "flight_id"]):
            status_row = df_status[
                (df_status["icao24"].astype(str) == str(icao24))
                & (df_status["flight_id"].astype(int) == int(flight_id))
            ]
            if status_row.empty:
                continue

            converged = bool(status_row["success"].iloc[0])

            try:
                runway_row = dataset.get_runway_assignment(str(icao24), int(flight_id))
            except KeyError:
                continue

            try:
                _, aircraft_dynamics = dataset.get_aircraft(str(icao24))
            except KeyError:
                continue

            evaluation = self.evaluate_flight(
                group, converged, runway_row, aircraft_dynamics, transformer
            )
            results.append(evaluation)

        return pd.DataFrame([vars(e) for e in results])

    @staticmethod
    def aggregate(eval_df: DataFrame) -> dict:
        if eval_df.empty:
            return {"n_total": 0, "n_converged": 0, "success_rate": None}

        n_total = len(eval_df)
        success_rate = float(eval_df["success"].mean())

        converged_df = eval_df[eval_df["converged"]]
        n_converged = len(converged_df)

        success_df = eval_df[eval_df["success"]]

        def _stats(df: DataFrame, col: str) -> dict:
            if df.empty:
                return {"mean": None, "median": None, "std": None, "max": None}
            s = df[col].abs()
            return {
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()) if len(s) > 1 else 0.0,
                "max": float(s.max()),
            }

        return {
            "n_total": n_total,
            "n_converged": n_converged,
            "success_rate": success_rate,
            "lateral_dev_m": _stats(converged_df, "lateral_dev_m"),
            "vertical_dev_m": _stats(converged_df, "vertical_dev_m"),
            "angle_dev_deg": _stats(converged_df, "angle_dev_deg"),
            "gamma_dev_deg": _stats(converged_df, "gamma_dev_deg"),
            "velocity_dev_ms": _stats(success_df, "velocity_dev_ms"),
        }


if __name__ == "__main__":
    from pyproj import Transformer as Trf
    from utils_project.geo_transformer import GeoTransformer

    base_path = "../data/CYVR/20260625_1508"
    dataset = LazyFlightParquetDataset(base_path)

    airport_lat, airport_lon = 49.1967, -123.1815
    crs_to = GeoTransformer.utm_epsg(airport_lat, airport_lon)
    transformer = Trf.from_crs("EPSG:4326", crs_to, always_xy=True)

    evaluator = TrajectorySuccessEvaluator(
        lateral_tol_m=15.0,
        vertical_tol_up_m=20.0,
        angle_tol_deg=10.0,
    )

    output_dir = Path("../data/CYVR/rnav_optimized")
    eval_df = evaluator.evaluate_batch(
        trajectories_path=output_dir / "rnav_trajectories.parquet",
        status_path=output_dir / "rnav_status.jsonl",
        dataset=dataset,
        transformer=transformer,
    )

    print(eval_df)

    stats = evaluator.aggregate(eval_df)

    for key, value in stats.items():
        print(f"  {key}: {value}")