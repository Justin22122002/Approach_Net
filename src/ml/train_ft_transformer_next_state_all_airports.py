from pathlib import Path

import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, Subset

from ml.utils.data_loader.landing_next_state_dataset import LandingNextStateDataset
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from ml.utils.models.landing_action_ft_transformer import LandingActionFTTransformer
from ml.utils.trainer.mlp_trainer import LandingActionTrainer
from utils.airport import Airport


if __name__ == "__main__":
    LandingActionTrainer.set_seed()

    dataset_name: str = "20260802_0920"

    airport_list_icao: list[str] = [
        "CYVR",
        "CYUL",
        "CYYZ",
        # "CYYC",
        # "CYEG",
    ]

    all_datasets: list = []
    per_airport_split: list = []
    train_X, train_Y = [], []

    for airport_icao in airport_list_icao:
        airport = Airport.load_from_folder(base_path=Path("./data/"), icao=airport_icao)

        flight_dataset = LazyFlightParquetDataset(f"./data/{airport_icao}/train/{dataset_name}")
        dataset = LandingNextStateDataset(
            flight_dataset=flight_dataset,
            mode="bspline_state",
            max_dt=5.0,
            airport_center_latlon=(airport.lat, airport.lon),
            airport_elevation_m=airport.get_mean_runway_elevation_ft() * 0.3048,
        )
        print(f"[{airport_icao}] samples: {len(dataset)}, input_dim: {dataset.input_dim}")
        all_datasets.append(dataset)

        train_flights, val_flights = dataset.split_flights(val_frac=0.2, seed=42)
        per_airport_split.append((dataset, train_flights, val_flights))

        for f in train_flights:
            X, Y = dataset._cache[f]
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

    train_loader = DataLoader(final_train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(final_val_set, batch_size=256, shuffle=False)

    device = LandingActionTrainer.get_device()

    model = LandingActionFTTransformer(
        input_dim=all_datasets[0].input_dim,
        action_dim=6,   # x, y, h, v, gamma, chi (next state)
    )

    trainer = LandingActionTrainer(
        model=model,
        optimizer=optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5),
        loss=nn.MSELoss(),
        device=device,
        results_path=Path("ml/results/multiple_airports_landing_next_state_ft_transformer"),
        patience=5,
    )

    trainer.train(train_loader, val_loader, epochs=200)

    trainer.save_normalization(x_mean, x_std, y_mean, y_std)

    trainer.plot_metrics(group_by_type=True)
    trainer.save_metrics()
    trainer.save_last_model()