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
    velocity_dev_pct: float | None = None
    landing_time_s: float | None = None
    s_needed: float = float("inf")           # strict tolerances -- drives `success` AND succ@1/succ@2/...
    s_needed_window: float = float("inf")     # wide window tolerances -- drives window_auc ONLY


class TrajectorySuccessEvaluator:
    def __init__(
        self,
        lateral_tol_m: float = 15.0,
        vertical_tol_up_m: float = 20.0,
        vertical_tol_down_m: float | None = None,
        angle_tol_deg: float = 10.0,
        gamma_ref_deg: float = -3.0,
        gamma_tol_deg: float = 3.0,
        velocity_tol_pct: float = 0.20,
        window_lateral_tol_m: float = 2500.0,
        window_vertical_tol_up_m: float = 1000.0,
        window_vertical_tol_down_m: float | None = None,
        window_angle_tol_deg: float = 10.0,
        window_gamma_tol_deg: float = 3.0,
        window_velocity_tol_pct: float = 2.5,
    ) -> None:
        self.lateral_tol_m: float = lateral_tol_m
        self.vertical_tol_up_m: float = vertical_tol_up_m
        self.vertical_tol_down_m: float = (
            vertical_tol_down_m if vertical_tol_down_m is not None else vertical_tol_up_m
        )
        self.angle_tol_deg: float = angle_tol_deg

        self.gamma_ref_deg: float = gamma_ref_deg
        self.gamma_tol_deg: float = gamma_tol_deg

        self.velocity_tol_pct: float = velocity_tol_pct

        self.APPROACH_SPEED_FACTOR: float = 1.3
        self.STANDARD_GLIDE_PATH_DEG: float = gamma_ref_deg

        self.window_lateral_tol_m: float = window_lateral_tol_m
        self.window_vertical_tol_up_m: float = window_vertical_tol_up_m
        self.window_vertical_tol_down_m: float = (
            window_vertical_tol_down_m if window_vertical_tol_down_m is not None else window_vertical_tol_up_m
        )
        self.window_angle_tol_deg: float = window_angle_tol_deg
        self.window_gamma_tol_deg: float = window_gamma_tol_deg
        self.window_velocity_tol_pct: float = window_velocity_tol_pct

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

    def _compute_s_needed(
        self,
        lateral_dev_m: float,
        vertical_dev_m: float,
        angle_dev_deg: float,
        gamma_dev_deg: float,
        velocity_dev_pct: float | None,
    ) -> float:
        ratios = []

        ratios.append(abs(lateral_dev_m) / self.lateral_tol_m if self.lateral_tol_m > 0 else float("inf"))

        if vertical_dev_m >= 0:
            vt = self.vertical_tol_up_m
        else:
            vt = self.vertical_tol_down_m
        ratios.append(abs(vertical_dev_m) / vt if vt > 0 else float("inf"))

        ratios.append(abs(angle_dev_deg) / self.angle_tol_deg if self.angle_tol_deg > 0 else float("inf"))
        ratios.append(abs(gamma_dev_deg) / self.gamma_tol_deg if self.gamma_tol_deg > 0 else float("inf"))

        if velocity_dev_pct is None or self.velocity_tol_pct <= 0:
            ratios.append(float("inf"))
        else:
            ratios.append(abs(velocity_dev_pct) / self.velocity_tol_pct)

        return float(max(ratios))

    def _compute_s_needed_window(
        self,
        lateral_dev_m: float,
        vertical_dev_m: float,
        angle_dev_deg: float,
        gamma_dev_deg: float,
        velocity_dev_pct: float | None,
    ) -> float:
        ratios = []

        ratios.append(
            abs(lateral_dev_m) / self.window_lateral_tol_m if self.window_lateral_tol_m > 0 else float("inf")
        )

        if vertical_dev_m >= 0:
            vt = self.window_vertical_tol_up_m
        else:
            vt = self.window_vertical_tol_down_m
        ratios.append(abs(vertical_dev_m) / vt if vt > 0 else float("inf"))

        ratios.append(
            abs(angle_dev_deg) / self.window_angle_tol_deg if self.window_angle_tol_deg > 0 else float("inf")
        )
        ratios.append(
            abs(gamma_dev_deg) / self.window_gamma_tol_deg if self.window_gamma_tol_deg > 0 else float("inf")
        )

        if velocity_dev_pct is None or self.window_velocity_tol_pct <= 0:
            ratios.append(float("inf"))
        else:
            ratios.append(abs(velocity_dev_pct) / self.window_velocity_tol_pct)

        return float(max(ratios))

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
        landing_time_s = float(df_sorted["time_s"].iloc[-1])
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

        glide_goal_rad = np.radians(self.gamma_ref_deg)
        e_gam = np.arctan2(
            np.sin(gammas[idx] - glide_goal_rad),
            np.cos(gammas[idx] - glide_goal_rad),
        )
        gamma_dev_deg = float(abs(np.degrees(e_gam)))
        gamma_ok = gamma_dev_deg <= self.gamma_tol_deg

        if vertical_dev_m >= 0:
            vertical_ok = vertical_dev_m <= self.vertical_tol_up_m
        else:
            vertical_ok = abs(vertical_dev_m) <= self.vertical_tol_down_m

        dummy_state = AircraftKinematicsStateCasadi(
            x=0.0, y=0.0, h=goal_h, v=0.0, gamma=0.0, chi=0.0,
            aircraft_type=aircraft_dynamics,
        )
        v_stall = float(dummy_state.stall_speed(goal_h))
        approach_speed_ref_ms = v_stall * self.APPROACH_SPEED_FACTOR
        velocity_dev_ms = float(vs[idx] - approach_speed_ref_ms)
        velocity_dev_pct = float(velocity_dev_ms / approach_speed_ref_ms) if approach_speed_ref_ms > 0 else None
        velocity_ok = (
            velocity_dev_pct is not None
            and abs(velocity_dev_pct) <= self.velocity_tol_pct
        )

        geometry_ok = (
            lateral_dev_m <= self.lateral_tol_m
            and vertical_ok
            and angle_dev_deg <= self.angle_tol_deg
            and gamma_ok
            and velocity_ok
        )

        success = converged and geometry_ok

        s_needed = self._compute_s_needed(
            lateral_dev_m, vertical_dev_m, angle_dev_deg, gamma_dev_deg, velocity_dev_pct
        )
        s_needed_window = self._compute_s_needed_window(
            lateral_dev_m, vertical_dev_m, angle_dev_deg, gamma_dev_deg, velocity_dev_pct
        )

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
            velocity_dev_pct=velocity_dev_pct,
            landing_time_s=landing_time_s,
            s_needed=s_needed,
            s_needed_window=s_needed_window,
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
            raise ValueError(f"Status log {status_path} is empty or does not exist.")

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
    def _stats(df: DataFrame, col: str) -> dict:
        if df.empty or col not in df:
            return {"mean": None, "median": None, "std": None, "max": None}
        s = df[col].dropna().abs()
        if s.empty:
            return {"mean": None, "median": None, "std": None, "max": None}
        return {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()) if len(s) > 1 else 0.0,
            "max": float(s.max()),
        }

    @classmethod
    def _metric_block(cls, df: DataFrame) -> dict:
        return {
            "n": len(df),
            "lateral_dev_m": cls._stats(df, "lateral_dev_m"),
            "vertical_dev_m": cls._stats(df, "vertical_dev_m"),
            "angle_dev_deg": cls._stats(df, "angle_dev_deg"),
            "gamma_dev_deg": cls._stats(df, "gamma_dev_deg"),
            "velocity_dev_ms": cls._stats(df, "velocity_dev_ms"),
            "velocity_dev_pct": cls._stats(df, "velocity_dev_pct"),
            "landing_time_s": cls._stats(df, "landing_time_s"),
        }

    @staticmethod
    def _window_auc(df: DataFrame) -> float | None:
        if df.empty or "s_needed_window" not in df or "converged" not in df:
            return None
        s_needed = df["s_needed_window"].to_numpy(dtype=float)
        converged = df["converged"].to_numpy(dtype=bool)
        contribution = np.where(converged, np.clip(1.0 - np.minimum(s_needed, 1.0), 0.0, 1.0), 0.0)
        return float(contribution.mean())

    S_CHECKPOINTS: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 10.0)

    @staticmethod
    def _success_rate_at_s(df: DataFrame, s: float) -> float | None:
        if df.empty or "s_needed" not in df or "converged" not in df:
            return None
        s_needed = df["s_needed"].to_numpy(dtype=float)
        converged = df["converged"].to_numpy(dtype=bool)
        return float(np.mean(converged & (s_needed <= s)))

    @classmethod
    def _success_rate_by_s(cls, df: DataFrame) -> dict:
        return {s: cls._success_rate_at_s(df, s) for s in cls.S_CHECKPOINTS}

    @classmethod
    def aggregate(cls, eval_df: DataFrame) -> dict:
        if eval_df.empty:
            return {"n_total": 0, "n_converged": 0, "success_rate": None, "window_auc": None}

        n_total = len(eval_df)
        success_rate = float(eval_df["success"].mean())
        n_converged = int(eval_df["converged"].sum())
        window_auc = cls._window_auc(eval_df)
        success_rate_by_s = cls._success_rate_by_s(eval_df)
        s_needed_stats = cls._stats(eval_df, "s_needed")
        s_needed_window_stats = cls._stats(eval_df, "s_needed_window")

        success_df = eval_df[eval_df["success"]]

        return {
            "n_total": n_total,
            "n_converged": n_converged,
            "success_rate": success_rate,
            "window_auc": window_auc,
            "success_rate_by_s": success_rate_by_s,
            "s_needed_stats": s_needed_stats,
            "s_needed_window_stats": s_needed_window_stats,
            "all": cls._metric_block(eval_df),
            "success_only": cls._metric_block(success_df),
        }