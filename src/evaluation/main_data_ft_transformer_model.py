from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from pyproj import Transformer

from evaluation.main_data_optimizer import build_start_state
from ml.utils.models.landing_action_ft_transformer import LandingActionFTTransformer
from physics.aircraft_dynamics import AircraftDynamics
from physics.aircraft_environment_casadi import AircraftEnvironmentCasadi
from physics.batch_result_store import already_processed, append_trajectory, append_status
from utils.airport import Airport
from utils_project.geo_transformer import GeoTransformer
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from ml.utils.data_loader.landing_action_dataset import LandingActionDataset


def load_norm(path: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[0].astype(np.float32), arr[1].astype(np.float32)


def predict_action(
    model: LandingActionFTTransformer,
    dyn_feat: np.ndarray,
    rwy_feat: np.ndarray,          # airport-centered, elevation RELATIVE: [fix_x, fix_y, elev_rel, bearing]
    state_feat: np.ndarray,        # airport-centered, h RELATIVE: [x, y, h_rel, v, gamma, chi]
    aircraft_dynamics: AircraftDynamics,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> tuple[float, float, float]:
    x = np.concatenate([dyn_feat, state_feat, rwy_feat])
    assert x.shape[0] == x_mean.shape[0], f"feature dim {x.shape[0]} != normalization dim {x_mean.shape[0]}"
    x_norm = torch.from_numpy((x - x_mean) / x_std).unsqueeze(0)

    with torch.no_grad():
        pred_norm = model(x_norm).squeeze(0).numpy()
    t, n, phi = (pred_norm * y_std + y_mean).tolist()

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
    model: LandingActionFTTransformer,
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
        state_feat = np.array(
            [float(state.x) - airport_cx, float(state.y) - airport_cy,
             float(state.h) - airport_elev, float(state.v), float(state.gamma), float(state.chi)],
            dtype=np.float32,
        )

        t, n, phi = predict_action(
            model, dyn_feat, rwy_feat, state_feat, aircraft_dynamics, x_mean, x_std, y_mean, y_std,
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
            "chi": np.radians(float(state.chi)),
            "gamma": np.radians(float(state.gamma)),
            "thrust": t,
            "load_factor": n,
            "roll": phi,
        })

        env.apply_action(t, n, phi)

        # todo: this can be tricky, add if the runway threshold is passed
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
                "chi": np.radians(float(state.chi)),
                "gamma": np.radians(float(state.gamma)),
                "thrust": None,
                "load_factor": None,
                "roll": None,
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

    airport_cx, airport_cy = to_utm.transform(alon, alat)

    x_mean, x_std = load_norm(norm_x_path)
    y_mean, y_std = load_norm(norm_y_path)

    model = LandingActionFTTransformer(input_dim=len(x_mean), action_dim=3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    n_total = len(dataset)

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = output_dir / "ft_transformer_rollout.parquet"
    status_path = output_dir / "ft_transformer_rollout_status.jsonl"

    n_ok = 0

    for idx in range(n_total):
        icao24, flight_id = dataset.flights[idx]

        print(f"\n[{idx + 1}/{n_total}] FT-Transformer rollout (no constraint) {icao24} / flight_id={flight_id}")

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
    TIME = "2026-08-05_21-33-34"  # 2026-08-06_07-08-10,  2026-08-05_21-33-34 -> flip augmentation
    MODEL_PATH = f"../ml/results/multiple_airports_landing_action_ft_transformer_augmented/{TIME}/best_model.pth"
    NORM_X_PATH = f"../ml/results/multiple_airports_landing_action_ft_transformer_augmented/{TIME}/normalization_x.csv"
    NORM_Y_PATH = f"../ml/results/multiple_airports_landing_action_ft_transformer_augmented/{TIME}/normalization_y.csv"

    airport_list_icao: list[str] = [
        "CYVR",
        "CYUL",
        "CYYZ",
        # "CYYC",
        # "CYEG",
    ]

    dataset_name: str = "20260709_1207" # note: arrivals_20260802_0920 -> wrong, 20260802_0920 -> right

    for airport_icao in airport_list_icao:
        icao = airport_icao
        base_path = f"../data/{icao}/test/{dataset_name}"
        dataset = LazyFlightParquetDataset(base_path)

        airport: Airport = Airport.load_from_folder(base_path=Path("../data/"), icao=icao)

        run_batch_rollout(
            dataset=dataset,
            airport_center_latlon=(airport.lat, airport.lon),
            airport_elevation_m=airport.get_mean_runway_elevation_ft() * 0.3048,
            model_path=MODEL_PATH,
            norm_x_path=NORM_X_PATH,
            norm_y_path=NORM_Y_PATH,
            output_dir=Path(f"../data/{icao}/multiple_airports_ft_transformer_rollout"),
            dt=1.0,
            max_steps=2000,
        )