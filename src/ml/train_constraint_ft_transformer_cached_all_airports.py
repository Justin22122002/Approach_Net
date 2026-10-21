import os
from pathlib import Path

import cvxpy as cp
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, Subset

from ml.pre_process_datasets import load_or_build_dataset
from ml.utils.models.landing_action_ft_transformer import LandingActionFTTransformer
from ml.utils.models.constraint_layer_cvxpy import AircraftCombinedQPLayer
from ml.utils.trainer.constrained_landing_action_trainer import ConstrainedLandingActionTrainer
from ml.utils.run_cli import exit_if_test, parse_run_args, report_results_folder

if __name__ == "__main__":
    args = parse_run_args()

    ConstrainedLandingActionTrainer.set_seed()

    dataset_name: str = "20260802_0920"  # 20260709_1207 20260802_0920
    airport_list_icao: list[str] = [
        "CYVR",
        "CYUL",
        "CYYZ",
        # "CYYC",
        # "CYEG",
    ]

    cap_data: bool = False # True

    folder_path: Path = Path("./data/")  # "../data/"

    TIME_STAMP: str = args.pretrained_dir or "2026-10-10_20-33-57"

    PRETRAINED_MODEL_PATH: Path | None = Path(
        f"ml/results/multiple_airports_landing_action_ft_transformer/{TIME_STAMP}/best_model.pth"
    )
    USE_PRETRAINED_MODEL: bool = True

    exit_if_test(args, __file__, TIME_STAMP=TIME_STAMP, PRETRAINED_MODEL_PATH=PRETRAINED_MODEL_PATH)

    all_datasets, per_airport_split, train_X, train_Y = [], [], [], []

    for airport_icao in airport_list_icao:
        dataset = load_or_build_dataset(airport_icao, folder_path, dataset_name, cap_data=cap_data)
        print(f"[{airport_icao}] samples: {len(dataset)}, input_dim: {dataset.input_dim}")
        all_datasets.append(dataset)

        train_flights, val_flights = dataset.split_flights(val_frac=0.2, seed=42)
        per_airport_split.append((dataset, train_flights, val_flights))

        for f in train_flights:
            X, Y, _, _ = dataset._cache[f]
            train_X.append(X)
            train_Y.append(Y)

    all_X = np.concatenate(train_X, axis=0)
    all_Y = np.concatenate(train_Y, axis=0)

    x_mean, x_std = all_X.mean(0), all_X.std(0) + 1e-8
    y_mean, y_std = all_Y.mean(0), all_Y.std(0) + 1e-8

    train_sets, val_sets = [], []
    for dataset, train_flights, val_flights in per_airport_split:
        dataset.set_normalization_stats(x_mean, x_std, y_mean, y_std)
        train_idx = dataset.indices_for_flights(set(train_flights))
        val_idx = dataset.indices_for_flights(set(val_flights))
        train_sets.append(Subset(dataset, train_idx))
        val_sets.append(Subset(dataset, val_idx))

    final_train_set = ConcatDataset(train_sets)
    final_val_set = ConcatDataset(val_sets)

    print(f"total train samples: {len(final_train_set)}, val samples: {len(final_val_set)}")

    # drop_last is needed, cause the solver can have problems with indexing the batches with different sizes
    train_loader = DataLoader(final_train_set, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(final_val_set, batch_size=256, shuffle=False, drop_last=True)

    device = ConstrainedLandingActionTrainer.get_device()

    base_model = LandingActionFTTransformer(input_dim=all_datasets[0].input_dim, action_dim=3)

    if USE_PRETRAINED_MODEL and PRETRAINED_MODEL_PATH is not None:
        base_model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
        print(f"base_model warm start of {PRETRAINED_MODEL_PATH}")
    else:
        print("base_model complete training (no warm start)")

    qp_layer = AircraftCombinedQPLayer(
        base_model=base_model,
        y_mean=y_mean,
        y_std=y_std,
        debug=False,
        solver=cp.MOREAU,  # cp.SCS
        # parameter for training
    )

    optimizer = optim.AdamW(qp_layer.parameters(), lr=1e-5, weight_decay=1e-5)  # lr=1e-4

    trainer = ConstrainedLandingActionTrainer(
        model=qp_layer,
        optimizer=optimizer,
        loss=nn.MSELoss(),
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        results_path=Path("ml/results/multiple_airports_landing_action_constraint_ft_transformer"),
        patience=5,  # 5 10
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3), # 2
        tau=2.7,
    )

    report_results_folder(trainer.results_folder_name, args.result_file)
    trainer.save_normalization(x_mean, x_std, y_mean, y_std)
    trainer.train(train_loader, val_loader, epochs=200)  # 30

    trainer.plot_metrics(group_by_type=True)
    trainer.save_metrics()
    trainer.save_last_model()

