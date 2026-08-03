from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyproj import Transformer

from evaluation.main_data_resnet_modell_constraint_cvxpy import LayerMode
from evaluation.utils.trajectory_evaluator import TrajectorySuccessEvaluator
from utils.airport import Airport
from utils_project.geo_transformer import GeoTransformer
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset


def plot_sample_trajectories(
    evaluator: TrajectorySuccessEvaluator,
    dataset: LazyFlightParquetDataset,
    transformer: Transformer,
    trajectories_path: Path,
    n_samples: int = 6,
    seed: int = 42,
    label: str = "",
) -> None:
    df_traj = pd.read_parquet(trajectories_path)

    keys = df_traj[["icao24", "flight_id"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(keys), size=min(n_samples, len(keys)), replace=False)

    n = len(sample_idx)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for i, ki in enumerate(sample_idx):
        icao24, flight_id = keys.iloc[ki]
        ax = axes[i // cols][i % cols]

        group = df_traj[
            (df_traj["icao24"] == icao24) & (df_traj["flight_id"] == flight_id)
        ].sort_values("step")

        ax.plot(group["x"], group["y"], "b-", label="trajectory")
        ax.scatter(group["x"].iloc[0], group["y"].iloc[0], c="green", s=60, label="start")
        ax.scatter(group["x"].iloc[-1], group["y"].iloc[-1], c="red", s=60, label="end")

        try:
            runway_row = dataset.get_runway_assignment(str(icao24), int(flight_id))
            goal_x, goal_y, _, _ = evaluator._project_goal(transformer, runway_row)
            ax.scatter(goal_x, goal_y, c="black", marker="x", s=80, label="runway fix")
        except KeyError:
            pass

        ax.set_title(f"{icao24} / {flight_id}")
        ax.axis("equal")
        ax.grid(True)
        ax.legend(fontsize=7)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(f"{label} sample trajectories (ground track)")
    fig.tight_layout()
    plt.show()


def flatten_stats(icao: str, layer_mode: str, stats: dict) -> dict:
    row: dict = {
        "icao": icao,
        "layer_mode": layer_mode,
        "n_total": stats.get("n_total"),
        "n_converged": stats.get("n_converged"),
        "success_rate": stats.get("success_rate"),
    }

    for block_name in ("all", "success_only"):
        block = stats.get(block_name)
        if not block:
            continue
        row[f"{block_name}_n"] = block.get("n")
        for metric_name, metric_stats in block.items():
            if metric_name == "n" or not isinstance(metric_stats, dict):
                continue
            for stat_name, stat_value in metric_stats.items():
                row[f"{block_name}_{metric_name}_{stat_name}"] = stat_value

    return row


def evaluate_dir(
    evaluator: TrajectorySuccessEvaluator,
    dataset: LazyFlightParquetDataset,
    transformer: Transformer,
    output_dir: Path,
    traj_filename: str,
    status_filename: str,
    label: str,
    icao: str,
    layer_mode: str,
    n_samples: int = 6,
) -> dict:
    print(f"\n {label} ({output_dir})")

    eval_df = evaluator.evaluate_batch(
        trajectories_path=output_dir / traj_filename,
        status_path=output_dir / status_filename,
        dataset=dataset,
        transformer=transformer,
    )

    eval_df.to_csv(output_dir / f"{label.lower()}_evaluation.csv", index=False)
    print(eval_df)

    stats = evaluator.aggregate(eval_df)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    plot_sample_trajectories(
        evaluator, dataset, transformer,
        trajectories_path=output_dir / traj_filename,
        n_samples=n_samples,
        label=label,
    )

    return flatten_stats(icao, layer_mode, stats)


if __name__ == "__main__":
    evaluator = TrajectorySuccessEvaluator(
        lateral_tol_m=15.0,
        vertical_tol_up_m=20.0,
        angle_tol_deg=10.0,
        gamma_tol_deg=3.0,
        velocity_tol_pct=0.50,
    )

    LAYER_MODES: list[LayerMode] = ["envelope", "funnel", "combined"]

    airport_list_icao: list[str] = [
        "CYVR",
        "CYUL",
        "CYYZ",
        # "CYYC", data problem
        # "CYEG", data problem
    ]

    summary_rows: list[dict] = []

    for airport_icao in airport_list_icao:
        test_base_path = f"../data/{airport_icao}/test/20260709_1207"
        airport_dataset = LazyFlightParquetDataset(test_base_path)

        airport_obj: Airport = Airport.load_from_folder(base_path=Path("../data/"), icao=airport_icao)
        crs_to = GeoTransformer.utm_epsg(airport_obj.lat, airport_obj.lon)
        airport_transformer = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)

        baseline_row = evaluate_dir(
            evaluator, airport_dataset, airport_transformer,
            output_dir=Path(f"../data/{airport_icao}/resnet_rollout"),
            traj_filename="resnet_rollout_status.parquet",
            status_filename="resnet_rollout_status.jsonl",
            label="RESNET_BASELINE",
            icao=airport_icao,
            layer_mode="baseline",
            n_samples=6,
        )
        summary_rows.append(baseline_row)

        for layer_mode in LAYER_MODES:
            row = evaluate_dir(
                evaluator, airport_dataset, airport_transformer,
                output_dir=Path(f"../data/{airport_icao}/multiple_airports_resnet_qp_rollout"),
                traj_filename=f"{layer_mode}_qp_rollout.parquet",
                status_filename=f"{layer_mode}_qp_rollout_status.jsonl",
                label=f"RESNET_CONSTRAINT_QP_{layer_mode.upper()}",
                icao=airport_icao,
                layer_mode=layer_mode,
                n_samples=6,
            )
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path("../data/qp_layer_mode_summary_old.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SUMMARY] {summary_path}")
    print(summary_df)