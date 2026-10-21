from pathlib import Path

import cvxpy as cp
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader

from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from ml.utils.data_loader.multi_aircraft_landing_action_dataset import (
    MultiAircraftLandingActionDataset,
    MultiAircraftTimestepView,
)
from ml.utils.models.constraint_graph_attention_model import MultiAircraftConstrainedQPLayer
from ml.utils.models.graph_attention_model import MultiAircraftLandingGNN
from ml.utils.trainer.constrained_multi_aircraft_gnn_trainer import ConstrainedMultiAircraftGNNTrainer
from ml.utils.trainer.multi_aircraft_GNN_trainer import graph_collate_fn
from utils.airport import Airport


def load_norm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[0].astype(np.float32), arr[1].astype(np.float32)


if __name__ == "__main__":
    ConstrainedMultiAircraftGNNTrainer.set_seed()

    dataset_name: str = "20260802_0920"

    airport_list_icao: list[str] = [
        "CYVR",
        "CYUL",
        "CYYZ",
        # "CYYC",
        # "CYEG",
    ]

    # --- warm start: complete pretrained MultiAircraftLandingGNN (encoder + GNN layers + head) ---
    # results folder of an unconstrained multi_aircraft_gnn run (BaseTrainer -> best_model.pth = gnn.state_dict()),
    # a constrained run works too with GNN_CHECKPOINT_NAME = "best_base_model.pth"
    GNN_TIME: str = "2026-10-16_06-08-04" # 2026-10-10_23-39-39 2026-10-12_11-33-15 2026-10-16_06-08-04
    PRETRAINED_GNN_DIR: Path | None = Path(f"ml/results/multiple_airports_landing_action_ft_transformer_gnn/{GNN_TIME}")
    GNN_CHECKPOINT_NAME: str = "best_model.pth"

    per_airport: list[tuple[MultiAircraftLandingActionDataset, list[int], list[int]]] = []
    all_train_X, all_train_Y = [], []

    for icao in airport_list_icao:
        airport = Airport.load_from_folder(base_path=Path("./data/"), icao=icao)
        flight_dataset = LazyFlightParquetDataset(f"./data/{icao}/train/{dataset_name}")

        dataset = MultiAircraftLandingActionDataset(
            flight_dataset=flight_dataset,
            mode="bspline_state",
            max_dt=5.0,
            airport_center_latlon=(airport.lat, airport.lon),
            airport_elevation_m=airport.get_mean_runway_elevation_ft() * 0.3048,
            provide_raw_features=True,  # packed (n_t, 39) -> needed by the QP layers
        )
        print(f"[{icao}] valid flights: {len(dataset._cache)}, input_dim: {dataset.input_dim}")

        train_flights, val_flights = dataset.split_flights_by_arrivals(val_frac=0.2)
        per_airport.append((dataset, train_flights, val_flights))

        for f in train_flights:
            X, Y, _, _, _ = dataset._cache[f]  # (n_pts, 15), (n_pts, 3), limits, approach, timestamps
            all_train_X.append(X)
            all_train_Y.append(Y)

    all_X = np.concatenate(all_train_X, axis=0)
    all_Y = np.concatenate(all_train_Y, axis=0)
    x_mean, x_std = all_X.mean(0), all_X.std(0) + 1e-8
    y_mean, y_std = all_Y.mean(0), all_Y.std(0) + 1e-8

    train_views: list[MultiAircraftTimestepView] = []
    val_views: list[MultiAircraftTimestepView] = []
    for dataset, train_flights, val_flights in per_airport:
        dataset.set_normalization_stats(x_mean, x_std, y_mean, y_std)
        train_views.append(MultiAircraftTimestepView(dataset, train_flights))
        val_views.append(MultiAircraftTimestepView(dataset, val_flights))

    final_train_set = ConcatDataset(train_views)
    final_val_set = ConcatDataset(val_views)
    print(f"total train timesteps: {len(final_train_set)}, val timesteps: {len(final_val_set)}")

    train_loader = DataLoader(final_train_set, batch_size=512, shuffle=True, collate_fn=graph_collate_fn)
    val_loader = DataLoader(final_val_set, batch_size=512, shuffle=False, collate_fn=graph_collate_fn)

    device = ConstrainedMultiAircraftGNNTrainer.get_device()

    gnn = MultiAircraftLandingGNN(input_dim=per_airport[0][0].input_dim, action_dim=3)
    gnn_ckpt = PRETRAINED_GNN_DIR / GNN_CHECKPOINT_NAME
    gnn.load_state_dict(torch.load(gnn_ckpt, map_location="cpu"), strict=True)
    print(f"gnn warm start of {gnn_ckpt}")

    model = MultiAircraftConstrainedQPLayer(
        base_model=gnn,
        y_mean=y_mean,
        y_std=y_std,
        solver=cp.MOREAU,  # cp.SCS
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)  # small lr -> fine-tuning of the pretrained GNN

    trainer = ConstrainedMultiAircraftGNNTrainer(
        model=model,
        optimizer=optimizer,
        loss=nn.MSELoss(),
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        results_path=Path("ml/results/multiple_airports_landing_action_constraint_ft_transformer_gnn"),
        patience=5,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3),
        tau=2.7,
    )

    trainer.save_normalization(x_mean, x_std, y_mean, y_std)
    trainer.train(train_loader, val_loader, epochs=200)

    trainer.plot_metrics(group_by_type=True)
    trainer.save_metrics()
    trainer.save_last_model()