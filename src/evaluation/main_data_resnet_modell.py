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
from utils_project.geo_transformer import GeoTransformer
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from ml.utils.data_loader.landing_action_dataset import LandingActionDataset
from ml.utils.models.mlp_model import LandingActionMLP, LandingActionResNet


def load_norm(path: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[0].astype(np.float32), arr[1].astype(np.float32)


def predict_action(
    model: LandingActionMLP,
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

    # capping of the actions
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
    to_latlon: Transformer,
    model: LandingActionMLP,
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
    rwy_feat = np.array([
        runway_row["fix_latitude"], runway_row["fix_longitude"],
        runway_row["fix_elevation_m"], runway_row["runway_bearing"],
    ], dtype=np.float32)

    rows: list[dict] = []
    landed = False
    step = 0

    for step in range(max_steps):
        lon, lat = to_latlon.transform(float(state.x), float(state.y))
        state_feat = np.array(
            [lat, lon, float(state.h), float(state.v), float(state.gamma), float(state.chi)],
            dtype=np.float32,
        )

        t, n, phi = predict_action(
            model, dyn_feat, rwy_feat, state_feat, x_mean, x_std, y_mean, y_std, aircraft_dynamics
        )

        rows.append({
            "icao24": icao24,
            "flight_id": flight_id,
            "step": step,
            "time_s": step * dt,
            "x": float(state.x),
            "y": float(state.y),
            "h": float(state.h),
            "v": float(state.v),
            "chi": float(state.chi),
            "gamma": float(state.gamma),
            "thrust": t,
            "load_factor": n,
            "roll": phi,
        })

        env.apply_action(t, n, phi)

        if float(state.h) <= runway_elevation_m:
            landed = True
            step += 1
            rows.append({
                "icao24": icao24,
                "flight_id": flight_id,
                "step": step,
                "time_s": step * dt,
                "x": float(state.x),
                "y": float(state.y),
                "h": float(state.h),
                "v": float(state.v),
                "chi": float(state.chi),
                "gamma": float(state.gamma),
                "thrust": None,
                "load_factor": None,
                "roll": None,
            })
            break

    return rows, landed


def run_batch_rollout(
    dataset: LazyFlightParquetDataset,
    airport_lat: float,
    airport_lon: float,
    model_path: str,
    norm_x_path: str,
    norm_y_path: str,
    output_dir: Path,
    dt: float = 1.0,
    max_steps: int = 2000,
    resume: bool = True,
) -> None:

    crs_to = GeoTransformer.utm_epsg(airport_lat, airport_lon)
    to_utm = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)
    to_latlon = Transformer.from_crs(crs_to, "EPSG:4326", always_xy=True)

    model = LandingActionResNet(
        input_dim=15, # see dataset for reference
        hidden_dim=256,
        n_blocks=4,
        output_dim=3
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

        print(f"\n[{idx + 1}/{n_total}] MLP rollout {icao24} / flight_id={flight_id}")

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
                to_utm, to_latlon, model, x_mean, x_std, y_mean, y_std,
                dt=dt, max_steps=max_steps,
            )

            df_traj = pd.DataFrame(rows)
            append_trajectory(df_traj, trajectories_path)

            t_opt = float(df_traj["time_s"].iloc[-1])

            append_status(status_path, {
                "icao24": icao24,
                "flight_id": flight_id,
                "success": landed,
                "T_opt": t_opt,
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
                "icao24": icao24,
                "flight_id": flight_id,
                "success": False,
                "T_opt": None,
                "error": str(e),
            })

    print(f"\n[DONE] {n_ok}/{n_total} landed")
    print(f"  Trajektorien: {trajectories_path}")
    print(f"  Status-Log:   {status_path}")


if __name__ == "__main__":
    base_path = "../data/CYVR/20260625_1508"
    dataset = LazyFlightParquetDataset(base_path)

    TIME = "2026-07-01_21-56-02"
    MODEL_PATH = f"../ml/results/landing_action_resnet/{TIME}/best_model.pth"
    NORM_X_PATH = f"../ml/results/landing_action_resnet/{TIME}/normalization_x.csv"
    NORM_Y_PATH = f"../ml/results/landing_action_resnet/{TIME}/normalization_y.csv"

    run_batch_rollout(
        dataset=dataset,
        airport_lat=49.1967,
        airport_lon=-123.1815,
        model_path=MODEL_PATH,
        norm_x_path=NORM_X_PATH,
        norm_y_path=NORM_Y_PATH,
        output_dir=Path("../data/CYVR/resnet_rollout"),
        dt=1.0,
        max_steps=2000,
    )