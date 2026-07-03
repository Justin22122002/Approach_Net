from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyproj import Transformer

from evaluation.utils.trajectory_evaluator import TrajectorySuccessEvaluator
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


def evaluate_dir(
    evaluator: TrajectorySuccessEvaluator,
    dataset: LazyFlightParquetDataset,
    transformer: Transformer,
    output_dir: Path,
    traj_filename: str,
    status_filename: str,
    label: str,
    n_samples: int = 6,
) -> None:
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


if __name__ == "__main__":
    base_path = "../data/CYVR/20260625_1508"
    dataset = LazyFlightParquetDataset(base_path)

    # todo: load airport dynamically
    airport_lat, airport_lon = 49.1967, -123.1815
    crs_to = GeoTransformer.utm_epsg(airport_lat, airport_lon)
    transformer = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)

    evaluator = TrajectorySuccessEvaluator(
        lateral_tol_m=15.0,
        vertical_tol_up_m=20.0,
        angle_tol_deg=10.0,
    )

    evaluate_dir(
        evaluator, dataset, transformer,
        output_dir=Path("../data/CYVR/rnav_optimized"),
        traj_filename="rnav_trajectories.parquet",
        status_filename="rnav_status.jsonl",
        label="RNAV",
        n_samples=6,
    )

    evaluate_dir(
        evaluator, dataset, transformer,
        output_dir=Path("../data/CYVR/mlp_rollout"),
        traj_filename="mlp_trajectories.parquet",
        status_filename="mlp_status.jsonl",
        label="MLP",
        n_samples=6,
    )

    evaluate_dir(
        evaluator, dataset, transformer,
        output_dir=Path("../data/CYVR/resnet_rollout"),
        traj_filename="resnet_rollout_trajectories.parquet",
        status_filename="resnet_rollout_status.jsonl",
        label="MLP",
        n_samples=6,
    )