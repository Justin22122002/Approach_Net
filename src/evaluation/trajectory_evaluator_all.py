from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evaluation.utils.trajectory_evaluator import TrajectorySuccessEvaluator
from utils.airport import Airport
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from ml.utils.data_loader.landing_action_dataset import LandingActionDataset

FT_TO_M = 0.3048


def plot_sample_trajectories(
    evaluator: TrajectorySuccessEvaluator,
    ds: LandingActionDataset,
    trajectories_path: Path,
    eval_df: pd.DataFrame,
    n_samples: int = 6,
    seed: int = 42,
    label: str = "",
) -> None:
    df_traj = pd.read_parquet(trajectories_path)

    # (icao24, flight_id) -> success from the TrajectorySuccessEvaluator; missing -> not evaluated
    success_of: dict[tuple[str, int], bool] = {
        (str(r.icao24), int(r.flight_id)): bool(r.success) for r in eval_df.itertuples()
    } if not eval_df.empty else {}

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

        success = success_of.get((str(icao24), int(flight_id)))
        color = "tab:gray" if success is None else ("tab:green" if success else "tab:red")
        status = "NOT EVALUATED" if success is None else ("SUCCESS" if success else "FAIL")

        ax.plot(group["x"], group["y"], "-", color="tab:blue", label="trajectory")
        ax.scatter(group["x"].iloc[0], group["y"].iloc[0], c="tab:blue", marker="o", s=60, label="start")
        ax.scatter(group["x"].iloc[-1], group["y"].iloc[-1], c="tab:orange", marker="s", s=60, label="end")

        try:
            goal_x, goal_y, _, _ = evaluator._project_goal(ds, str(icao24), int(flight_id))
            ax.scatter(goal_x, goal_y, c="black", marker="x", s=80, label="runway fix")
        except (KeyError, ValueError):
            pass

        ax.set_title(f"{icao24} / {flight_id}  -  {status}", color=color, fontweight="bold")
        for spine in ax.spines.values():  # colored frame: green = success, red = fail
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
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
        "window_auc": stats.get("window_auc"),
    }

    for s, rate in (stats.get("success_rate_by_s") or {}).items():
        row[f"succ_rate_s{s:g}"] = rate

    s_needed_stats = stats.get("s_needed_stats") or {}
    for stat_name, stat_value in s_needed_stats.items():
        row[f"s_needed_{stat_name}"] = stat_value

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
    ds: LandingActionDataset,
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
        ds=ds,
    )

    eval_df.to_csv(output_dir / f"{Path(traj_filename).stem}_evaluation.csv", index=False)
    print(eval_df)

    stats = evaluator.aggregate(eval_df)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    plot_sample_trajectories(
        evaluator, ds,
        trajectories_path=output_dir / traj_filename,
        eval_df=eval_df,
        n_samples=n_samples,
        label=label,
    )

    return flatten_stats(icao, layer_mode, stats)


def rollout_experiment(label: str, output_dir_name: str, name: str, run_id: str, constrained: bool = False) -> dict:
    """file names exactly like RolloutEvaluator.run(): {name}[_constrained]_rollout_{run_id}"""
    tag = f"{name}{'_constrained' if constrained else ''}_rollout_{run_id}"
    return dict(
        label=label,
        layer_mode=label.lower(),
        output_dir_name=output_dir_name,
        traj_filename=f"{tag}.parquet",
        status_filename=f"{tag}_status.jsonl",
    )


RUN_FT_ACTION = "2026-10-10_20-33-57"
RUN_FT_ACTION_CONSTRAINT_TRAINING = "2026-10-11_06-09-41"

RUN_RESNET_ACTION = "2026-10-10_20-02-35"
RUN_RESNET_ACTION_CONSTRAINT_TRAINING = "2026-10-11_05-34-36"

RUN_FT_NEXT_STATE = "2026-10-11_16-49-30"
RUN_RESNET_NEXT_STATE = "2026-10-11_18-23-44"

# 2026-10-10_23-39-39, 2026-10-12_11-33-15, 2026-10-16_06-08-04
RUN_FT_GNN = "2026-10-10_23-39-39"
# 2026-10-11_17-13-54, 2026-10-12_18-10-46, 2026-10-16_20-10-17
RUN_FT_GNN_CONSTRAINT_TRAINING = "2026-10-11_17-13-54"

# 2026-10-12_21-14-14, 2026-10-16_06-21-55
RUN_RESNET_GNN = "2026-10-12_21-14-14"
# 2026-10-13_12-57-59, 2026-10-16_22-17-12
RUN_RESNET_GNN_CONSTRAINT_TRAINING = "2026-10-13_12-57-59"

EXPERIMENTS: list[dict] = [
    rollout_experiment(
        "RESNET101_ACTION_BASELINE",
        "multiple_airports_resnet101_rollout",
        "resnet101",
        RUN_RESNET_ACTION
    ),
    rollout_experiment(
        "RESNET101_ACTION_CONSTRAINT",
        "multiple_airports_resnet101_rollout_constraint",
        "resnet101",
        RUN_RESNET_ACTION,
        constrained=True
    ),
    rollout_experiment(
        "RESNET101_ACTION_CONSTRAINT_TRAINING",
        "multiple_airports_resnet101_rollout_constraint_training",
         "resnet101",
        RUN_RESNET_ACTION_CONSTRAINT_TRAINING,
        constrained=True
    ),
    rollout_experiment(
        "FT_TRANSFORMER_ACTION_BASELINE",
        "multiple_airports_ft_transformer_rollout",
        "ft_transformer",
        RUN_FT_ACTION
    ),
    rollout_experiment(
        "FT_TRANSFORMER_ACTION_CONSTRAINT",
        "multiple_airports_ft_transformer_rollout_constraint",
        "ft_transformer",
        RUN_FT_ACTION,
        constrained=True
    ),
    rollout_experiment(
        "FT_TRANSFORMER_ACTION_CONSTRAINT_TRAINING",
        "multiple_airports_ft_transformer_rollout_constraint_training",
        "ft_transformer",
        RUN_FT_ACTION_CONSTRAINT_TRAINING,
        constrained=True
    ),
    rollout_experiment(
        "RESNET101_NEXT_STATE_BASELINE",
        "multiple_airports_next_state_resnet101_rollout",
        "resnet101_next_state",
        RUN_RESNET_NEXT_STATE
    ),
    rollout_experiment(
        "RESNET101_NEXT_STATE_CONSTRAINT",
        "multiple_airports_next_state_resnet101_rollout_constraint",
        "resnet101_next_state",
        RUN_RESNET_NEXT_STATE,
        constrained=True
    ),
    rollout_experiment(
        "FT_TRANSFORMER_NEXT_STATE_BASELINE",
        "multiple_airports_ft_transformer_next_state_rollout",
        "ft_transformer_next_state",
        RUN_FT_NEXT_STATE
    ),
    rollout_experiment(
        "FT_TRANSFORMER_NEXT_STATE_CONSTRAINT",
        "multiple_airports_ft_transformer_next_state_rollout_constraint",
        "ft_transformer_next_state",
        RUN_FT_NEXT_STATE,
        constrained=True
    ),
    rollout_experiment(
        "RNAV_OPTIMIZER",
        "rnav_optimized",
        "rnav_optimizer",
        "20261005_1406",
    ),
    rollout_experiment(
        "FT_GNN_BASELINE",
        "multiple_airports_ft_transformer_gnn_rollout",
        "gnn",
        RUN_FT_GNN
    ),
    rollout_experiment(
        "FT_GNN_CONSTRAINT",
        "multiple_airports_ft_transformer_gnn_rollout_constraint",
        "gnn",
        RUN_FT_GNN,
        constrained=True
    ),
    rollout_experiment(
        "FT_GNN_CONSTRAINT_TRAINING",
        "multiple_airports_ft_transformer_gnn_rollout_constraint_training",
        "gnn",
        RUN_FT_GNN_CONSTRAINT_TRAINING,
        constrained=True
    ),
    rollout_experiment(
        "RESNET_GNN_BASELINE",
        "multiple_airports_resnet_101_gnn_rollout",
        "gnn",
        RUN_RESNET_GNN
    ),
    rollout_experiment(
        "RESNET_GNN_CONSTRAINT",
        "multiple_airports_resnet_101_gnn_rollout_constraint",
        "gnn",
        RUN_RESNET_GNN,
        constrained=True
    ),
    rollout_experiment(
        "RESNET_GNN_CONSTRAINT_TRAINING",
        "multiple_airports_resnet_101_gnn_rollout_constraint_training",
        "gnn",
        RUN_RESNET_GNN_CONSTRAINT_TRAINING,
        constrained=True
    ),
]


if __name__ == "__main__":
    evaluator = TrajectorySuccessEvaluator(
        lateral_tol_m=15.0,
        vertical_tol_up_m=20.0,
        vertical_tol_down_m=20.0,
        angle_tol_deg=10.0,
        gamma_tol_deg=3.0,
        velocity_tol_pct=0.50,
        window_lateral_tol_m=2500.0,
        window_vertical_tol_up_m=1000.0,
        window_angle_tol_deg=10.0,
        window_gamma_tol_deg=3.0,
        window_velocity_tol_pct=2.5,
    )

    airport_list_icao: list[str] = [
        "CYVR",
        "CYUL",
        "CYYZ",
        # "CYYC",
        # "CYEG",
    ]

    dataset_name: str = "20261005_1406" # 20261005_1406 -> new test set, 20260709_1207 -> old test set

    summary_rows: list[dict] = []

    for airport_icao in airport_list_icao:
        airport_obj: Airport = Airport.load_from_folder(base_path=Path("../data/"), icao=airport_icao)
        ds = LandingActionDataset(
            LazyFlightParquetDataset(f"../data/{airport_icao}/test/{dataset_name}"),
            mode="bspline_state",
            airport_center_latlon=(airport_obj.lat, airport_obj.lon),
            airport_elevation_m=airport_obj.get_mean_runway_elevation_ft() * FT_TO_M,
        )

        for exp in EXPERIMENTS:
            try:
                row = evaluate_dir(
                    evaluator, ds,
                    output_dir=Path(f"../data/{airport_icao}/{exp['output_dir_name']}"),
                    traj_filename=exp["traj_filename"],
                    status_filename=exp["status_filename"],
                    label=exp["label"],
                    icao=airport_icao,
                    layer_mode=exp["layer_mode"],
                    n_samples=6,
                )
                summary_rows.append(row)
            except Exception as e:
                print(f"[WARN] skipping {exp['label']} @ {airport_icao}: {e}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path("../data/summary_experiments.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SUMMARY] {summary_path}")
    print(summary_df)