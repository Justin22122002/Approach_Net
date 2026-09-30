from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from pandas import DataFrame
from pyproj import Transformer
from rtdl_revisiting_models import FTTransformer

from evaluation.main_data_optimizer import build_start_state
from ml.utils.models.landing_action_ft_transformer import LandingActionFTTransformer
from physics.aircraft_dynamics import AircraftDynamics
from physics.aircraft_environment_casadi import AircraftEnvironmentCasadi
from physics.batch_result_store import already_processed, append_trajectory, append_status
from utils.airport import Airport
from utils_project.geo_transformer import GeoTransformer
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from ml.utils.data_loader.landing_action_dataset import LandingActionDataset
from ml.utils.models.constraint_layer_cvxpy import (
    AircraftEnvelopeQPLayer, AircraftFunnelQPLayer, AircraftCombinedQPLayer,
)

LayerMode = Literal["envelope", "funnel", "combined"]
GuidedModel = AircraftEnvelopeQPLayer | AircraftFunnelQPLayer | AircraftCombinedQPLayer


def load_norm(path: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[0].astype(np.float32), arr[1].astype(np.float32)


def build_guided_model(
    layer_mode: LayerMode,
    model: LandingActionFTTransformer,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    runway_elevation_m: float,
    stall_margin: float,
    alpha_stall: float,
) -> GuidedModel:
    kwargs = dict(base_model=model, y_mean=y_mean, y_std=y_std, debug=False)

    if layer_mode == "envelope":
        return AircraftEnvelopeQPLayer(**kwargs, stall_margin=stall_margin, alpha_stall=alpha_stall)
    elif layer_mode == "funnel":
        return AircraftFunnelQPLayer(**kwargs, h_thr=runway_elevation_m)
    elif layer_mode == "combined":
        return AircraftCombinedQPLayer(
            **kwargs, stall_margin=stall_margin, alpha_stall=alpha_stall, h_thr=runway_elevation_m,
        )
    else:
        raise ValueError(f"unknown layer_mode: {layer_mode!r}")


def predict_action(
    layer_mode: LayerMode,
    guided_model: GuidedModel,
    dyn_feat: np.ndarray,
    rwy_feat: np.ndarray,          # airport-centered, elevation RELATIVE: [fix_x, fix_y, elev_rel, bearing]
    state_feat: np.ndarray,        # airport-centered, h RELATIVE: [x, y, h_rel, v, gamma, chi]
    runway_elevation_m: float,     # ABSOLUTE runway elevation, used as h_thr in the QP layer
    aircraft_dynamics: AircraftDynamics,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    airport_elev: float,
) -> tuple[float, float, float]:
    x_c, y_c, h_rel, v, gamma, chi = state_feat.tolist()
    fix_x_c, fix_y_c, fix_elev_rel, bearing = rwy_feat.tolist()

    x = np.concatenate([dyn_feat, state_feat, rwy_feat])
    assert x.shape[0] == x_mean.shape[0], f"feature dim {x.shape[0]} != normalization dim {x_mean.shape[0]}"
    x_norm = (x - x_mean) / x_std

    h_abs = h_rel + airport_elev

    aircraft_state_t = torch.tensor([[h_abs, v, gamma]], dtype=torch.float32)
    aircraft_dyn_t = torch.from_numpy(dyn_feat).unsqueeze(0)  # mass, C_D0, area, cl_max, k
    aircraft_limits_t = torch.tensor([[
        aircraft_dynamics.max_thrust, aircraft_dynamics.min_load_factor,
        aircraft_dynamics.max_load_factor, aircraft_dynamics.max_roll_angle,
    ]], dtype=torch.float32)

    x_norm_t = torch.from_numpy(x_norm).unsqueeze(0)

    if layer_mode == "envelope":
        out = guided_model(x_norm_t, aircraft_state_t, aircraft_dyn_t, aircraft_limits_t)
    else:
        aircraft_pos_t = torch.tensor([[x_c, y_c, chi]], dtype=torch.float32)
        runway_pos_t = torch.tensor([[fix_x_c, fix_y_c, bearing]], dtype=torch.float32)
        out = guided_model(
            x_norm_t, aircraft_state_t, aircraft_dyn_t, aircraft_limits_t,
            aircraft_pos=aircraft_pos_t, runway_pos=runway_pos_t,
        )

    t, n, phi = out.squeeze(0).tolist()

    t = float(np.clip(t, 0.0, aircraft_dynamics.max_thrust))
    n = float(np.clip(n, aircraft_dynamics.min_load_factor, aircraft_dynamics.max_load_factor))
    phi = float(np.clip(phi, -aircraft_dynamics.max_roll_angle, aircraft_dynamics.max_roll_angle))

    return t, n, phi


def rollout_single_flight(
    layer_mode: LayerMode,
    icao24: str,
    flight_id: int,
    df_flight: DataFrame,
    dataset: LazyFlightParquetDataset,
    to_utm: Transformer,
    airport_cx: float,
    airport_cy: float,
    airport_elev: float,
    guided_model: GuidedModel,
    x_mean: np.ndarray,
    x_std: np.ndarray,
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
            layer_mode, guided_model, dyn_feat, rwy_feat, state_feat, runway_elevation_m,
            aircraft_dynamics, x_mean, x_std, airport_elev,
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
    layer_mode: LayerMode,
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
    stall_margin: float = 1.15,
    alpha_stall: float = 1.0,
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
    trajectories_path = output_dir / f"{layer_mode}_qp_rollout.parquet"
    status_path = output_dir / f"{layer_mode}_qp_rollout_status.jsonl"

    n_ok = 0

    for idx in range(n_total):
        icao24, flight_id = dataset.flights[idx]

        print(f"\n[{idx + 1}/{n_total}] {layer_mode}-QP FT-Transformer rollout {icao24} / flight_id={flight_id}")

        if resume and already_processed(status_path, icao24, flight_id):
            print("  [SKIP] already processed (resume=True)")
            n_ok += 1
            continue

        try:
            df_flight = dataset[idx]
            if df_flight.empty:
                raise ValueError("Empty flight data")

            runway_row = dataset.get_runway_assignment(icao24, flight_id)
            runway_elevation_m = float(runway_row["fix_elevation_m"])

            guided_model = build_guided_model(
                layer_mode, model, y_mean, y_std, runway_elevation_m, stall_margin, alpha_stall,
            )

            rows, landed = rollout_single_flight(
                layer_mode, icao24, flight_id, df_flight, dataset,
                to_utm, airport_cx, airport_cy, airport_elevation_m,
                guided_model, x_mean, x_std,
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

    print(f"\n[DONE] {layer_mode}: {n_ok}/{n_total} landed")
    print(f"  Trajektorien: {trajectories_path}")
    print(f"  Status-Log:   {status_path}")


if __name__ == "__main__":
    TIME = "2026-08-05_21-33-34"  # 2026-08-06_07-08-10,  2026-08-05_21-33-34 -> flip augmentation
    MODEL_PATH = f"../ml/results/multiple_airports_landing_action_ft_transformer_augmented/{TIME}/best_model.pth"
    NORM_X_PATH = f"../ml/results/multiple_airports_landing_action_ft_transformer_augmented/{TIME}/normalization_x.csv"
    NORM_Y_PATH = f"../ml/results/multiple_airports_landing_action_ft_transformer_augmented/{TIME}/normalization_y.csv"

    LAYER_MODES: list[LayerMode] = ["combined"]  # ["envelope", "funnel", "combined"]

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

        for layer_mode in LAYER_MODES:
            run_batch_rollout(
                layer_mode=layer_mode,
                dataset=dataset,
                airport_center_latlon=(airport.lat, airport.lon),
                airport_elevation_m=airport.get_mean_runway_elevation_ft() * 0.3048,
                model_path=MODEL_PATH,
                norm_x_path=NORM_X_PATH,
                norm_y_path=NORM_Y_PATH,
                output_dir=Path(f"../data/{icao}/multiple_airports_ft_transformer_rollout_constraint"),
                dt=1.0,
                max_steps=2000,
                stall_margin=1.15,
                alpha_stall=1.0,
            )