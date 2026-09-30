import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from pyproj import Transformer

from evaluation.main_data_optimizer import build_start_state, solve_single_flight
from ml.utils.data_loader.landing_action_dataset import LandingActionDataset
from ml.utils.data_loader.landing_next_state_dataset import LandingNextStateDataset
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from ml.utils.models.constraint_layer_cvxpy import AircraftCombinedQPLayer
from ml.utils.models.ils_funnel_state_layer_cvxpy import ILSFunnelStateLayer
from ml.utils.models.landing_action_ft_transformer import LandingActionFTTransformer
from ml.utils.models.landing_resnet_101 import LandingActionResNet101
from physics.aircraft_environment_casadi import AircraftEnvironmentCasadi
from utils.airport import Airport
from utils_project.geo_transformer import GeoTransformer

ICAO = "CYVR"
TEST_DATA_PATH = f"../data/{ICAO}/test/20260709_1207"
DATA_ROOT = Path("../data/")

DT = 1.0
MAX_STEPS = 2000
N_FLIGHTS = 10
MIN_DURATION_S = 200.0

STALL_MARGIN = 1.15
ALPHA_STALL = 1.0

FT_ACTION_TIME = "2026-08-05_21-33-34"
FT_ACTION_MODEL_PATH = f"../ml/results/multiple_airports_landing_action_ft_transformer_augmented/{FT_ACTION_TIME}/best_model.pth"
FT_ACTION_NORM_X = f"../ml/results/multiple_airports_landing_action_ft_transformer_augmented/{FT_ACTION_TIME}/normalization_x.csv"
FT_ACTION_NORM_Y = f"../ml/results/multiple_airports_landing_action_ft_transformer_augmented/{FT_ACTION_TIME}/normalization_y.csv"

FT_NEXT_TIME = "2026-08-06_13-05-37"
FT_NEXT_MODEL_PATH = f"../ml/results/multiple_airports_landing_next_state_ft_transformer/{FT_NEXT_TIME}/best_model.pth"
FT_NEXT_NORM_X = f"../ml/results/multiple_airports_landing_next_state_ft_transformer/{FT_NEXT_TIME}/normalization_x.csv"
FT_NEXT_NORM_Y = f"../ml/results/multiple_airports_landing_next_state_ft_transformer/{FT_NEXT_TIME}/normalization_y.csv"

RESNET_ACTION_TIME = "2026-08-05_14-54-31"
RESNET_ACTION_MODEL_PATH = f"../ml/results/multiple_airports_landing_action_resnet101_torchvision_augmented/{RESNET_ACTION_TIME}/best_model.pth"
RESNET_ACTION_NORM_X = f"../ml/results/multiple_airports_landing_action_resnet101_torchvision_augmented/{RESNET_ACTION_TIME}/normalization_x.csv"
RESNET_ACTION_NORM_Y = f"../ml/results/multiple_airports_landing_action_resnet101_torchvision_augmented/{RESNET_ACTION_TIME}/normalization_y.csv"

RESNET_NEXT_TIME = "2026-08-06_13-05-41"
RESNET_NEXT_MODEL_PATH = f"../ml/results/multiple_airports_landing_next_state_resnet101_torchvision/{RESNET_NEXT_TIME}/best_model.pth"
RESNET_NEXT_NORM_X = f"../ml/results/multiple_airports_landing_next_state_resnet101_torchvision/{RESNET_NEXT_TIME}/normalization_x.csv"
RESNET_NEXT_NORM_Y = f"../ml/results/multiple_airports_landing_next_state_resnet101_torchvision/{RESNET_NEXT_TIME}/normalization_y.csv"

OUTPUT_DIR = DATA_ROOT / ICAO
PLOT_PATH = OUTPUT_DIR / "benchmark_selected_flights.png"
CSV_PATH = OUTPUT_DIR / "benchmark_method_comparison.csv"
RAW_CSV_PATH = OUTPUT_DIR / "benchmark_method_comparison_raw.csv"


@dataclass
class ModelBundle:
    model: torch.nn.Module
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


def load_norm(path: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[0].astype(np.float32), arr[1].astype(np.float32)


def load_bundle(
    model_factory: Callable[[int], torch.nn.Module],
    model_path: str,
    norm_x_path: str,
    norm_y_path: str,
) -> ModelBundle:
    x_mean, x_std = load_norm(norm_x_path)
    y_mean, y_std = load_norm(norm_y_path)
    model = model_factory(len(x_mean))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return ModelBundle(model, x_mean, x_std, y_mean, y_std)


def compute_duration_s(df_flight: DataFrame) -> float:
    if "time" not in df_flight.columns or df_flight.empty:
        return float(len(df_flight) - 1)
    delta = df_flight["time"].max() - df_flight["time"].min()
    if hasattr(delta, "total_seconds"):
        return float(delta.total_seconds())
    return float(delta)


def select_flights(
    dataset: LazyFlightParquetDataset, n_flights: int, min_duration_s: float
) -> list[tuple[float, str, int, DataFrame]]:
    candidates: list[tuple[float, str, int, DataFrame]] = []

    for idx in range(len(dataset)):
        icao24, flight_id = dataset.flights[idx]
        df_flight = dataset[idx]
        if df_flight.empty:
            continue
        duration_s = compute_duration_s(df_flight)
        if duration_s >= min_duration_s:
            candidates.append((duration_s, icao24, flight_id, df_flight))

    candidates.sort(key=lambda c: c[0])
    return candidates[:n_flights]


def plot_selected_flights(selected: list[tuple[float, str, int, DataFrame]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for duration_s, icao24, flight_id, df_flight in selected:
        df_sorted = df_flight.sort_values("time") if "time" in df_flight.columns else df_flight
        ax.plot(
            df_sorted["lon"], df_sorted["lat"], marker=".", markersize=2, linewidth=1,
            label=f"{icao24}/{flight_id} ({duration_s:.0f}s)",
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{ICAO}: 10 Testflüge, kürzeste Dauer (>= {MIN_DURATION_S:.0f}s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _runway_feat(runway_row, to_utm: Transformer, airport_cx: float, airport_cy: float, airport_elev: float) -> np.ndarray:
    fix_x, fix_y = to_utm.transform(float(runway_row["fix_longitude"]), float(runway_row["fix_latitude"]))
    runway_elevation_m = float(runway_row["fix_elevation_m"])
    return np.array([
        fix_x - airport_cx, fix_y - airport_cy,
        runway_elevation_m - airport_elev, float(runway_row["runway_bearing"]),
    ], dtype=np.float32)


def _state_feat(state, airport_cx: float, airport_cy: float, airport_elev: float) -> np.ndarray:
    return np.array([
        float(state.x) - airport_cx, float(state.y) - airport_cy,
        float(state.h) - airport_elev, float(state.v), float(state.gamma), float(state.chi),
    ], dtype=np.float32)


def solve_rnav(
    icao24: str, flight_id: int, df_flight: DataFrame, dataset: LazyFlightParquetDataset,
    to_utm: Transformer, N: int = 200, T_min: float = 60.0, T_max: float = 1800.0,
) -> tuple[float, bool]:
    result = solve_single_flight(icao24, flight_id, df_flight, dataset, to_utm, N, T_min, T_max)
    return float(result["T_opt"]), bool(result["success"])


def _predict_action(
    model: torch.nn.Module, dyn_feat: np.ndarray, rwy_feat: np.ndarray, state_feat: np.ndarray,
    aircraft_dynamics, x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray,
) -> tuple[float, float, float]:
    x = np.concatenate([dyn_feat, state_feat, rwy_feat])
    x_norm = torch.from_numpy((x - x_mean) / x_std).unsqueeze(0)
    with torch.no_grad():
        pred_norm = model(x_norm).squeeze(0).numpy()
    t, n, phi = (pred_norm * y_std + y_mean).tolist()
    t = float(np.clip(t, 0.0, aircraft_dynamics.max_thrust))
    n = float(np.clip(n, aircraft_dynamics.min_load_factor, aircraft_dynamics.max_load_factor))
    phi = float(np.clip(phi, -aircraft_dynamics.max_roll_angle, aircraft_dynamics.max_roll_angle))
    return t, n, phi


def rollout_action_unconstrained(
    bundle: ModelBundle, icao24: str, flight_id: int, df_flight: DataFrame,
    dataset: LazyFlightParquetDataset, to_utm: Transformer,
    airport_cx: float, airport_cy: float, airport_elev: float,
    dt: float = DT, max_steps: int = MAX_STEPS,
) -> tuple[float, bool]:
    start_point, _ = dataset.get_start_and_end_points(df_flight)
    _, aircraft_dynamics = dataset.get_aircraft(icao24)
    runway_row = dataset.get_runway_assignment(icao24, flight_id)
    runway_elevation_m = float(runway_row["fix_elevation_m"])

    state = build_start_state(start_point, aircraft_dynamics, to_utm)
    env = AircraftEnvironmentCasadi(state=state, goal=None, dt=dt)

    dyn_feat = np.array([getattr(aircraft_dynamics, k) for k in LandingActionDataset.DYNAMICS_KEYS], dtype=np.float32)
    rwy_feat = _runway_feat(runway_row, to_utm, airport_cx, airport_cy, airport_elev)

    landed = False
    final_step = 0
    for step in range(max_steps):
        state_feat = _state_feat(state, airport_cx, airport_cy, airport_elev)
        t, n, phi = _predict_action(
            bundle.model, dyn_feat, rwy_feat, state_feat, aircraft_dynamics,
            bundle.x_mean, bundle.x_std, bundle.y_mean, bundle.y_std,
        )
        env.apply_action(t, n, phi)

        final_step = step
        if float(state.h) <= runway_elevation_m:
            landed = True
            final_step = step + 1
            break

    return final_step * dt, landed


def _predict_action_combined(
    guided_model: AircraftCombinedQPLayer, dyn_feat: np.ndarray, rwy_feat: np.ndarray, state_feat: np.ndarray,
    aircraft_dynamics, x_mean: np.ndarray, x_std: np.ndarray, airport_elev: float,
) -> tuple[float, float, float]:
    x_c, y_c, h_rel, v, gamma, chi = state_feat.tolist()
    fix_x_c, fix_y_c, _fix_elev_rel, bearing = rwy_feat.tolist()

    x = np.concatenate([dyn_feat, state_feat, rwy_feat])
    x_norm = (x - x_mean) / x_std
    h_abs = h_rel + airport_elev

    aircraft_state_t = torch.tensor([[h_abs, v, gamma]], dtype=torch.float32)
    aircraft_dyn_t = torch.from_numpy(dyn_feat).unsqueeze(0)
    aircraft_limits_t = torch.tensor([[
        aircraft_dynamics.max_thrust, aircraft_dynamics.min_load_factor,
        aircraft_dynamics.max_load_factor, aircraft_dynamics.max_roll_angle,
    ]], dtype=torch.float32)
    x_norm_t = torch.from_numpy(x_norm).unsqueeze(0)

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


def rollout_action_combined_qp(
    bundle: ModelBundle, icao24: str, flight_id: int, df_flight: DataFrame,
    dataset: LazyFlightParquetDataset, to_utm: Transformer,
    airport_cx: float, airport_cy: float, airport_elev: float,
    stall_margin: float = STALL_MARGIN, alpha_stall: float = ALPHA_STALL,
    dt: float = DT, max_steps: int = MAX_STEPS,
) -> tuple[float, bool]:
    start_point, _ = dataset.get_start_and_end_points(df_flight)
    _, aircraft_dynamics = dataset.get_aircraft(icao24)
    runway_row = dataset.get_runway_assignment(icao24, flight_id)
    runway_elevation_m = float(runway_row["fix_elevation_m"])

    guided_model = AircraftCombinedQPLayer(
        base_model=bundle.model, y_mean=bundle.y_mean, y_std=bundle.y_std,
        stall_margin=stall_margin, alpha_stall=alpha_stall, h_thr=runway_elevation_m, debug=False,
    )

    state = build_start_state(start_point, aircraft_dynamics, to_utm)
    env = AircraftEnvironmentCasadi(state=state, goal=None, dt=dt)

    dyn_feat = np.array([getattr(aircraft_dynamics, k) for k in LandingActionDataset.DYNAMICS_KEYS], dtype=np.float32)
    rwy_feat = _runway_feat(runway_row, to_utm, airport_cx, airport_cy, airport_elev)

    landed = False
    final_step = 0
    for step in range(max_steps):
        state_feat = _state_feat(state, airport_cx, airport_cy, airport_elev)
        t, n, phi = _predict_action_combined(
            guided_model, dyn_feat, rwy_feat, state_feat, aircraft_dynamics,
            bundle.x_mean, bundle.x_std, airport_elev,
        )
        env.apply_action(t, n, phi)

        final_step = step
        if float(state.h) <= runway_elevation_m:
            landed = True
            final_step = step + 1
            break

    return final_step * dt, landed


def _predict_next_state(
    model: torch.nn.Module, state_feat: np.ndarray, dyn_feat: np.ndarray, rwy_feat: np.ndarray,
    x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray,
) -> np.ndarray:
    x = np.concatenate([dyn_feat, state_feat, rwy_feat])
    x_norm_t = torch.from_numpy(((x - x_mean) / x_std).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        out_norm = model(x_norm_t)
    return out_norm.squeeze(0).numpy() * y_std + y_mean


def rollout_next_state_unconstrained(
    bundle: ModelBundle, icao24: str, flight_id: int, df_flight: DataFrame,
    dataset: LazyFlightParquetDataset, to_utm: Transformer,
    airport_cx: float, airport_cy: float, airport_elev: float,
    dt: float = DT, max_steps: int = MAX_STEPS,
) -> tuple[float, bool]:
    start_point, _ = dataset.get_start_and_end_points(df_flight)
    _, aircraft_dynamics = dataset.get_aircraft(icao24)
    runway_row = dataset.get_runway_assignment(icao24, flight_id)
    runway_elevation_m = float(runway_row["fix_elevation_m"])

    state = build_start_state(start_point, aircraft_dynamics, to_utm)

    dyn_feat = np.array([getattr(aircraft_dynamics, k) for k in LandingNextStateDataset.DYNAMICS_KEYS], dtype=np.float32)
    rwy_feat = _runway_feat(runway_row, to_utm, airport_cx, airport_cy, airport_elev)

    landed = False
    final_step = 0
    for step in range(max_steps):
        state_feat = _state_feat(state, airport_cx, airport_cy, airport_elev)
        nx, ny, nh_rel, nv, ngamma, nchi = _predict_next_state(
            bundle.model, state_feat, dyn_feat, rwy_feat, bundle.x_mean, bundle.x_std, bundle.y_mean, bundle.y_std,
        ).tolist()

        state.x = nx + airport_cx
        state.y = ny + airport_cy
        state.h = nh_rel + airport_elev
        state.v = nv
        state.gamma = ngamma
        state.chi = nchi

        final_step = step
        if float(state.h) <= runway_elevation_m:
            landed = True
            final_step = step + 1
            break

    return final_step * dt, landed


def _predict_funnel_next_state(
    guided_model: ILSFunnelStateLayer, state_feat: np.ndarray, dyn_feat: np.ndarray, rwy_feat: np.ndarray,
    x_mean: np.ndarray, x_std: np.ndarray, runway_pos_t: torch.Tensor,
) -> np.ndarray:
    x = np.concatenate([dyn_feat, state_feat, rwy_feat])
    x_norm_t = torch.from_numpy(((x - x_mean) / x_std).astype(np.float32)).unsqueeze(0)
    out = guided_model(x_norm_t, runway_pos_t)
    return out.squeeze(0).numpy()


def rollout_next_state_ils_funnel(
    bundle: ModelBundle, icao24: str, flight_id: int, df_flight: DataFrame,
    dataset: LazyFlightParquetDataset, to_utm: Transformer,
    airport_cx: float, airport_cy: float, airport_elev: float,
    dt: float = DT, max_steps: int = MAX_STEPS,
) -> tuple[float, bool]:
    start_point, _ = dataset.get_start_and_end_points(df_flight)
    _, aircraft_dynamics = dataset.get_aircraft(icao24)
    runway_row = dataset.get_runway_assignment(icao24, flight_id)
    runway_elevation_m = float(runway_row["fix_elevation_m"])

    guided_model = ILSFunnelStateLayer(
        base_model=bundle.model, y_mean=bundle.y_mean, y_std=bundle.y_std,
        h_thr=runway_elevation_m, debug=False,
    )

    state = build_start_state(start_point, aircraft_dynamics, to_utm)

    dyn_feat = np.array([getattr(aircraft_dynamics, k) for k in LandingNextStateDataset.DYNAMICS_KEYS], dtype=np.float32)
    rwy_feat = _runway_feat(runway_row, to_utm, airport_cx, airport_cy, airport_elev)
    fix_x, fix_y = to_utm.transform(float(runway_row["fix_longitude"]), float(runway_row["fix_latitude"]))
    runway_pos_t = torch.tensor(
        [[fix_x - airport_cx, fix_y - airport_cy, float(runway_row["runway_bearing"])]], dtype=torch.float32,
    )

    landed = False
    final_step = 0
    for step in range(max_steps):
        state_feat = _state_feat(state, airport_cx, airport_cy, airport_elev)
        nx, ny, nh_rel, nv, ngamma, nchi = _predict_funnel_next_state(
            guided_model, state_feat, dyn_feat, rwy_feat, bundle.x_mean, bundle.x_std, runway_pos_t,
        ).tolist()

        state.x = nx + airport_cx
        state.y = ny + airport_cy
        state.h = nh_rel + airport_elev
        state.v = nv
        state.gamma = ngamma
        state.chi = nchi

        final_step = step
        if float(state.h) <= runway_elevation_m:
            landed = True
            final_step = step + 1
            break

    return final_step * dt, landed


def main() -> None:
    dataset = LazyFlightParquetDataset(TEST_DATA_PATH)
    airport: Airport = Airport.load_from_folder(base_path=DATA_ROOT, icao=ICAO)

    alat, alon = airport.lat, airport.lon
    crs_to = GeoTransformer.utm_epsg(alat, alon)
    to_utm = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)
    airport_cx, airport_cy = to_utm.transform(alon, alat)
    airport_elev = airport.get_mean_runway_elevation_ft() * 0.3048

    print(f"[SELECT] Suche {N_FLIGHTS} Flüge mit Dauer >= {MIN_DURATION_S:.0f}s (kürzeste zuerst) an {ICAO} ...")
    selected = select_flights(dataset, N_FLIGHTS, MIN_DURATION_S)
    print(f"[SELECT] {len(selected)} Flüge gefunden:")
    for duration_s, icao24, flight_id, df_flight in selected:
        print(f"    {icao24}/{flight_id}: {duration_s:.0f}s, {len(df_flight)} Punkte")

    plot_selected_flights(selected, PLOT_PATH)
    print(f"[PLOT] gespeichert: {PLOT_PATH}")

    print("[MODELS] lade FT-Transformer & ResNet101 (je Action + Next-State) ...")
    ft_action = load_bundle(
        lambda d: LandingActionFTTransformer(input_dim=d, action_dim=3),
        FT_ACTION_MODEL_PATH, FT_ACTION_NORM_X, FT_ACTION_NORM_Y,
    )
    ft_next_state = load_bundle(
        lambda d: LandingActionFTTransformer(input_dim=d, action_dim=6),
        FT_NEXT_MODEL_PATH, FT_NEXT_NORM_X, FT_NEXT_NORM_Y,
    )
    resnet_action = load_bundle(
        lambda d: LandingActionResNet101(input_dim=d, output_dim=3),
        RESNET_ACTION_MODEL_PATH, RESNET_ACTION_NORM_X, RESNET_ACTION_NORM_Y,
    )
    resnet_next_state = load_bundle(
        lambda d: LandingActionResNet101(input_dim=d, output_dim=6),
        RESNET_NEXT_MODEL_PATH, RESNET_NEXT_NORM_X, RESNET_NEXT_NORM_Y,
    )

    methods: dict[str, Callable[[str, int, DataFrame], tuple[float, bool]]] = {
        "rnav_optimizer": lambda icao24, fid, df: solve_rnav(icao24, fid, df, dataset, to_utm),
        "ft_transformer_action_baseline": lambda icao24, fid, df: rollout_action_unconstrained(
            ft_action, icao24, fid, df, dataset, to_utm, airport_cx, airport_cy, airport_elev),
        "ft_transformer_action_constraint": lambda icao24, fid, df: rollout_action_combined_qp(
            ft_action, icao24, fid, df, dataset, to_utm, airport_cx, airport_cy, airport_elev),
        "ft_transformer_next_state_baseline": lambda icao24, fid, df: rollout_next_state_unconstrained(
            ft_next_state, icao24, fid, df, dataset, to_utm, airport_cx, airport_cy, airport_elev),
        "ft_transformer_next_state_ils_funnel": lambda icao24, fid, df: rollout_next_state_ils_funnel(
            ft_next_state, icao24, fid, df, dataset, to_utm, airport_cx, airport_cy, airport_elev),
        "resnet101_action_baseline": lambda icao24, fid, df: rollout_action_unconstrained(
            resnet_action, icao24, fid, df, dataset, to_utm, airport_cx, airport_cy, airport_elev),
        "resnet101_action_constraint": lambda icao24, fid, df: rollout_action_combined_qp(
            resnet_action, icao24, fid, df, dataset, to_utm, airport_cx, airport_cy, airport_elev),
        "resnet101_next_state_baseline": lambda icao24, fid, df: rollout_next_state_unconstrained(
            resnet_next_state, icao24, fid, df, dataset, to_utm, airport_cx, airport_cy, airport_elev),
        "resnet101_next_state_ils_funnel": lambda icao24, fid, df: rollout_next_state_ils_funnel(
            resnet_next_state, icao24, fid, df, dataset, to_utm, airport_cx, airport_cy, airport_elev),
    }

    per_method_records: dict[str, list[dict]] = {name: [] for name in methods}

    for duration_s, icao24, flight_id, df_flight in selected:
        print(f"\n=== Flug {icao24}/{flight_id} (Dauer {duration_s:.0f}s) ===")
        for method_name, method_fn in methods.items():
            t0 = time.perf_counter()
            try:
                T_opt, landed = method_fn(icao24, flight_id, df_flight)
                solve_time_s = time.perf_counter() - t0
                per_method_records[method_name].append({
                    "icao24": icao24, "flight_id": flight_id,
                    "T_opt": T_opt, "solve_time_s": solve_time_s, "landed": landed,
                })
                status = "OK" if landed else "WARN(nicht gelandet/konvergiert, aber gezählt)"
                print(f"  [{status}] {method_name}: T_opt={T_opt:.1f}s, solve_time={solve_time_s:.3f}s")
            except Exception as e:
                per_method_records[method_name].append({
                    "icao24": icao24, "flight_id": flight_id,
                    "T_opt": None, "solve_time_s": None, "landed": False,
                })
                print(f"  [FAIL] {method_name}: {e}")

    raw_rows = [
        {"model": method_name, **record}
        for method_name, records in per_method_records.items()
        for record in records
    ]
    df_raw = pd.DataFrame(raw_rows)
    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(RAW_CSV_PATH, index=False)

    rows = []
    for method_name, records in per_method_records.items():
        valid = [r for r in records if r["T_opt"] is not None]
        n_landed = sum(1 for r in records if r["landed"])
        avg_T_opt = float(np.mean([r["T_opt"] for r in valid])) if valid else None
        avg_solve_time = float(np.mean([r["solve_time_s"] for r in valid])) if valid else None
        rows.append({
            "model": method_name,
            "avg_landing_time_s": avg_T_opt,
            "avg_solving_time_s": avg_solve_time,
            "n_landed": n_landed,
            "n_success": len(valid),
            "n_flights": len(records),
        })

    df_summary = pd.DataFrame(rows)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(CSV_PATH, index=False)

    print(f"\n[DONE] Zusammenfassung gespeichert: {CSV_PATH}")
    print(f"[DONE] Rohdaten (pro Flug/Methode) gespeichert: {RAW_CSV_PATH}")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()