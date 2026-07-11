from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from pyproj import Transformer

from evaluation.main_data_optimizer import build_start_state
from physics.aircraft_dynamics import AircraftDynamics
from physics.aircraft_environment_casadi import AircraftEnvironmentCasadi
from physics.batch_result_store import already_processed, append_trajectory, append_status
from utils.airport import Airport
from utils_project.geo_transformer import GeoTransformer
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from ml.utils.data_loader.landing_action_dataset import LandingActionDataset
from ml.utils.models.resnet_model import LandingActionResNet


def load_norm(path: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[0].astype(np.float32), arr[1].astype(np.float32)


def predict_action(
    model: LandingActionResNet,
    dyn_feat: np.ndarray,
    rwy_feat: np.ndarray,
    state_feat: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    aircraft_dynamics: AircraftDynamics,
) -> tuple[float, float, float]:
    x = np.concatenate([dyn_feat, state_feat, rwy_feat])
    x_norm = (x - x_mean) / x_std

    with torch.no_grad():
        y_norm = model(torch.from_numpy(x_norm).unsqueeze(0)).squeeze(0).numpy()

    t, n, phi = (y_norm * y_std + y_mean).tolist()

    # capping of the actions (poor-man's safety filter until the CBF-QP layer)
    t = float(np.clip(t, 0.0, aircraft_dynamics.max_thrust))
    n = float(np.clip(n, aircraft_dynamics.min_load_factor, aircraft_dynamics.max_load_factor))
    phi = float(np.clip(phi, -aircraft_dynamics.max_roll_angle, aircraft_dynamics.max_roll_angle))

    return t, n, phi


def rollout_single_flight(
    icao24: str,
    flight_id: int,
    df_flight: DataFrame,
    dataset: LazyFlightParquetDataset,
    to_utm: Transformer,
    airport_cx: float,
    airport_cy: float,
    airport_elev: float,
    model: LandingActionResNet,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    dt: float = 1.0,
    max_steps: int = 2000,
) -> tuple[list[dict], bool]:

    start_point, _ = dataset.get_start_and_end_points(df_flight)
    _, aircraft_dynamics = dataset.get_aircraft(icao24)
    runway_row = dataset.get_runway_assignment(icao24, flight_id)
    runway_elevation_m = float(runway_row["fix_elevation_m"])

    state = build_start_state(start_point, aircraft_dynamics, to_utm)
    env = AircraftEnvironmentCasadi(state=state, goal=None, dt=dt)

    dyn_feat = np.array(
        [getattr(aircraft_dynamics, k) for k in LandingActionDataset.DYNAMICS_KEYS],
        dtype=np.float32,
    )

    # runway fix in airport-centered UTM frame, elevation RELATIVE (matches training)
    fix_x, fix_y = to_utm.transform(
        float(runway_row["fix_longitude"]), float(runway_row["fix_latitude"])
    )
    rwy_feat = np.array([
        fix_x - airport_cx, fix_y - airport_cy,
        runway_elevation_m - airport_elev, float(runway_row["runway_bearing"]),
    ], dtype=np.float32)

    rows: list[dict] = []
    landed = False

    for step in range(max_steps):
        # state.x/state.y are already UTM -> recenter; h RELATIVE for the base-model features
        state_feat = np.array(
            [float(state.x) - airport_cx, float(state.y) - airport_cy,
             float(state.h) - airport_elev, float(state.v), float(state.gamma), float(state.chi)],
            dtype=np.float32,
        )

        t, n, phi = predict_action(
            model, dyn_feat, rwy_feat, state_feat, x_mean, x_std, y_mean, y_std, aircraft_dynamics
        )

        rows.append({
            "icao24": icao24, "flight_id": flight_id, "step": step, "time_s": step * dt,
            "x": float(state.x), "y": float(state.y), "h": float(state.h), "v": float(state.v),
            "chi": float(state.chi), "gamma": float(state.gamma),
            "thrust": t, "load_factor": n, "roll": phi,
        })

        env.apply_action(t, n, phi)

        if float(state.h) <= runway_elevation_m:
            landed = True
            step += 1
            rows.append({
                "icao24": icao24, "flight_id": flight_id, "step": step, "time_s": step * dt,
                "x": float(state.x), "y": float(state.y), "h": float(state.h), "v": float(state.v),
                "chi": float(state.chi), "gamma": float(state.gamma),
                "thrust": None, "load_factor": None, "roll": None,
            })
            break

    return rows, landed


def run_batch_rollout(
    dataset: LazyFlightParquetDataset,
    airport_center_latlon: tuple[float, float],
    airport_elevation_m: float,
    model_path: str,
    norm_x_path: str,
    norm_y_path: str,
    output_dir: Path,
    dt: float = 1.0,
    max_steps: int = 2000,
    resume: bool = True,
) -> None:

    alat, alon = airport_center_latlon
    crs_to = GeoTransformer.utm_epsg(alat, alon)
    to_utm = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)

    # airport center in UTM -> origin of the airport-centered frame (matches dataset)
    airport_cx, airport_cy = to_utm.transform(alon, alat)

    model = LandingActionResNet(
        input_dim=15,  # dynamics(5) + state(6) + runway(4)
        hidden_dim=256,
        n_blocks=4,
        output_dim=3,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    x_mean, x_std = load_norm(norm_x_path)
    y_mean, y_std = load_norm(norm_y_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = output_dir / "resnet_rollout_status.parquet"
    status_path = output_dir / "resnet_rollout_status.jsonl"

    n_ok = 0
    n_total = len(dataset)

    for idx in range(n_total):
        icao24, flight_id = dataset.flights[idx]
        print(f"\n[{idx + 1}/{n_total}] rollout {icao24} / flight_id={flight_id}")

        if resume and already_processed(status_path, icao24, flight_id):
            print("  [SKIP] already processed (resume=True)")
            n_ok += 1
            continue

        try:
            df_flight = dataset[idx]
            if df_flight.empty:
                raise ValueError("Empty flight data")

            rows, landed = rollout_single_flight(
                icao24, flight_id, df_flight, dataset,
                to_utm, airport_cx, airport_cy, airport_elevation_m,
                model, x_mean, x_std, y_mean, y_std,
                dt=dt, max_steps=max_steps,
            )

            df_traj = pd.DataFrame(rows)
            append_trajectory(df_traj, trajectories_path)
            t_opt = float(df_traj["time_s"].iloc[-1])

            append_status(status_path, {
                "icao24": icao24, "flight_id": flight_id, "success": landed, "T_opt": t_opt,
                "error": None if landed else "did not reach runway elevation within max_steps",
            })
            if landed:
                n_ok += 1
                print(f"  [OK] T={t_opt:.1f} s (saved)")
            else:
                print(f"  [WARN] not landed, T={t_opt:.1f} s (saved)")

        except Exception as e:
            print(f"  [FAIL] {e}")
            append_status(status_path, {
                "icao24": icao24, "flight_id": flight_id, "success": False,
                "T_opt": None, "error": str(e),
            })

    print(f"\n[DONE] {n_ok}/{n_total} landed")
    print(f"  Trajektorien: {trajectories_path}")
    print(f"  Status-Log:   {status_path}")


if __name__ == "__main__":
    icao = "CYVR"
    airport: Airport = Airport.load_from_folder(base_path=Path("../data/"), icao=icao)
    base_path = f"../data/{icao}/20260625_1508"
    dataset = LazyFlightParquetDataset(base_path)

    TIME = "2026-07-07_21-23-13"
    MODEL_PATH = f"../ml/results/landing_action_resnet/{TIME}/best_model.pth"
    NORM_X_PATH = f"../ml/results/landing_action_resnet/{TIME}/normalization_x.csv"
    NORM_Y_PATH = f"../ml/results/landing_action_resnet/{TIME}/normalization_y.csv"

    run_batch_rollout(
        dataset=dataset,
        airport_center_latlon=(airport.lat, airport.lon),
        airport_elevation_m=airport.arinc_424_runway[0].runway_altitude * 0.3048,
        model_path=MODEL_PATH,
        norm_x_path=NORM_X_PATH,
        norm_y_path=NORM_Y_PATH,
        output_dir=Path("../data/CYVR/resnet_rollout"),
        dt=1.0,
        max_steps=2000,
    )