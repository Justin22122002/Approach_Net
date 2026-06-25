from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pyproj import Transformer

from physics.aircraft_dynamics import AircraftDynamics
from physics.aircraft_kinematics_state_casadi import AircraftKinematicsStateCasadi
from physics.landing_optimizer import LandingTrajectoryOptimizer, state_to_casadi
from utils_project.geo_transformer import GeoTransformer
from src.utils.data_loader.lazy_flight_parquet_dataset import  LazyFlightParquetDataset


def build_start_state(
    start_point: Series,
    aircraft_dynamics: AircraftDynamics,
    transformer: Transformer,
) -> AircraftKinematicsStateCasadi:
    """
    vertrate: Vertical speed in m/s (positive = climbing).
    velocity: Speed over ground in metres per second. -> m/s per second
    """
    x0, y0 = transformer.transform(float(start_point["lon"]), float(start_point["lat"]))

    v0 = float(start_point["velocity"])
    h0 = float(start_point["geoaltitude"])

    heading0 = float(start_point["heading"]) if pd.notna(start_point["heading"]) else 0.0
    chi0 = -heading0

    vertrate0 = float(start_point["vertrate"]) if pd.notna(start_point["vertrate"]) else 0.0
    gamma0 = float(np.degrees(np.arctan2(vertrate0, max(v0, 1e-3))))

    return AircraftKinematicsStateCasadi(
        x=x0, y=y0, h=h0,
        v=v0, gamma=gamma0, chi=chi0,
        aircraft_type=aircraft_dynamics,
    )


def solve_single_flight(
    icao24: str,
    flight_id: int,
    df_flight: DataFrame,
    dataset: LazyFlightParquetDataset,
    transformer: Transformer,
    N: int,
    T_min: float,
    T_max: float,
) -> dict:
    start_point, _ = dataset.get_start_and_end_points(df_flight)
    _, aircraft_dynamics = dataset.get_aircraft(icao24)
    runway_row = dataset.get_runway_assignment(icao24, flight_id)

    state = build_start_state(start_point, aircraft_dynamics, transformer)

    goal_x, goal_y = transformer.transform(
        float(runway_row["fix_longitude"]), float(runway_row["fix_latitude"])
    )
    goal_h = float(runway_row["fix_elevation_m"])
    goal_chi_deg = float(runway_row["runway_bearing"])
    goal = np.array([goal_x, goal_y, goal_h])

    optimizer = LandingTrajectoryOptimizer(state=state, N=N, T_min=T_min, T_max=T_max)
    x0_vec = state_to_casadi(state)

    X_opt, U_opt, T_opt = optimizer.solve(x0_vec, goal, goal_chi_deg)

    return {
        "icao24": icao24, "flight_id": flight_id,
        "X_opt": X_opt, "U_opt": U_opt, "T_opt": T_opt, "N": N,
    }


def trajectory_to_dataframe(result: dict) -> DataFrame:
    """X_opt = [x, y, h, v, chi, gamma] je Knotenpunkt, U_opt = [thrust, load_factor, roll]."""
    X_opt, U_opt, T_opt, N = result["X_opt"], result["U_opt"], result["T_opt"], result["N"]
    dt = T_opt / N

    rows = []
    for k in range(N + 1):
        rows.append({
            "icao24": result["icao24"],
            "flight_id": result["flight_id"],
            "step": k,
            "time_s": k * dt,
            "x": float(X_opt[0, k]), "y": float(X_opt[1, k]), "h": float(X_opt[2, k]),
            "v": float(X_opt[3, k]), "chi": float(X_opt[4, k]), "gamma": float(X_opt[5, k]),
            "thrust": float(U_opt[0, k]) if k < N else None,
            "load_factor": float(U_opt[1, k]) if k < N else None,
            "roll": float(U_opt[2, k]) if k < N else None,
        })
    return pd.DataFrame(rows)


# todo: get airport name and load whole airport, to get lat and lon etc. ...
def run_batch_optimization(
    dataset: LazyFlightParquetDataset,
    airport_lat: float,
    airport_lon: float,
    output_dir: Path,
    N: int = 100,
    T_min: float = 60.0,
    T_max: float = 1800.0,
) -> None:
    crs_to = GeoTransformer.utm_epsg(airport_lat, airport_lon)
    transformer = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_frames: list[DataFrame] = []
    status_rows: list[dict] = []

    for idx in range(len(dataset)):
        icao24, flight_id = dataset.flights[idx]
        print(f"\n[{idx + 1}/{len(dataset)}] Optimiere {icao24} / flight_id={flight_id} ...")

        try:
            df_flight = dataset[idx]
            if df_flight.empty:
                raise ValueError("DataFrame ist leer.")

            result = solve_single_flight(icao24, flight_id, df_flight, dataset, transformer, N, T_min, T_max)
            trajectory_frames.append(trajectory_to_dataframe(result))
            status_rows.append({"icao24": icao24, "flight_id": flight_id, "success": True, "T_opt": result["T_opt"], "error": None})
            print(f"  [OK] T_opt={result['T_opt']:.1f} s")

        except KeyError as e:
            print(f"  [SKIP] Fehlende Stammdaten: {e}")
            status_rows.append({"icao24": icao24, "flight_id": flight_id, "success": False, "T_opt": None, "error": str(e)})
        except Exception as e:
            print(f"  [FAIL] Optimierung fehlgeschlagen: {e}")
            status_rows.append({"icao24": icao24, "flight_id": flight_id, "success": False, "T_opt": None, "error": str(e)})

    suffix = datetime.now(UTC).strftime("%Y%m%d_%H%M")

    if trajectory_frames:
        pd.concat(trajectory_frames, ignore_index=True).to_parquet(
            output_dir / f"optimized_trajectories_{suffix}.parquet"
        )
    else:
        print("[WARN] Keine Trajektorie konnte optimiert werden.")

    df_status = pd.DataFrame(status_rows)
    df_status.to_parquet(output_dir / f"optimization_status_{suffix}.parquet")

    n_ok = int(df_status["success"].sum())
    print(f"\n[DONE] {n_ok}/{len(df_status)} Flüge erfolgreich optimiert.")


if __name__ == "__main__":
    base_path = "../data/CYVR/20260622_1510"
    dataset = LazyFlightParquetDataset(base_path)

    run_batch_optimization(
        dataset=dataset,
        airport_lat=49.1967,
        airport_lon=-123.1815,
        output_dir=Path("../../data/CYVR/optimized"),
        N=100, T_min=60.0, T_max=1800.0,
    )