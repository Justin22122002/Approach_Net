from pathlib import Path

import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset, Subset

from ml.utils.data_loader.landing_state_forecast_dataset import LandingActionWindowDataset
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from ml.utils.models.landing_action_itransformer import LandingActionITransformer
from ml.utils.trainer.landing_forecast_trainer import LandingActionITransformerTrainer
from utils.airport import Airport

WINDOW_LEN = 16

if __name__ == "__main__":
    LandingActionITransformerTrainer.set_seed()

    airport_list_icao: list[str] = [
        "CYVR",
        "CYUL",
        "CYYZ",
        # "CYYC", data problem
        # "CYEG", data problem
    ]

    all_datasets: list[LandingActionWindowDataset] = []
    per_airport_split = []
    train_x_series, train_y_series = [], []

    for airport_icao in airport_list_icao:
        airport = Airport.load_from_folder(base_path=Path("../data/"), icao=airport_icao)

        flight_dataset = LazyFlightParquetDataset(f"../data/{airport_icao}/train/20260708_1247")
        dataset = LandingActionWindowDataset(
            flight_dataset=flight_dataset,
            mode="bspline_state",
            airport_center_latlon=(airport.lat, airport.lon),
            airport_elevation_m=airport.arinc_424_runway[0].runway_altitude * 0.3048,
            window_len=WINDOW_LEN,
        )
        print(f"[{airport_icao}] flights: {len(dataset._cache)}, windows: {len(dataset)}")
        all_datasets.append(dataset)

        train_flights, val_flights = dataset.split_flights(val_frac=0.2, seed=42)
        per_airport_split.append((dataset, train_flights, val_flights))

        for f in train_flights:
            states, actions = dataset._cache[f]
            static_tiled = np.tile(dataset._static_cache[f], (len(states), 1))
            train_x_series.append(np.concatenate([states, static_tiled], axis=1))
            train_y_series.append(actions)

    all_x = np.concatenate(train_x_series, axis=0)
    all_y = np.concatenate(train_y_series, axis=0)
    x_mean, x_std = all_x.mean(axis=0), all_x.std(axis=0) + 1e-8
    y_mean, y_std = all_y.mean(axis=0), all_y.std(axis=0) + 1e-8

    train_sets, val_sets = [], []
    for dataset, train_flights, val_flights in per_airport_split:
        dataset.set_normalization_stats(x_mean, x_std, y_mean, y_std)
        train_idx = dataset.indices_for_flights(set(train_flights))
        val_idx = dataset.indices_for_flights(set(val_flights))
        train_sets.append(Subset(dataset, train_idx))
        val_sets.append(Subset(dataset, val_idx))

    final_train_set = ConcatDataset(train_sets)
    final_val_set = ConcatDataset(val_sets)

    print(f"total train windows: {len(final_train_set)}, val windows: {len(final_val_set)}")

    train_loader = DataLoader(final_train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(final_val_set, batch_size=32, shuffle=False)

    device = LandingActionITransformerTrainer.get_device()

    model = LandingActionITransformer(
        num_variates=all_datasets[0].num_variates,  # 15
        lookback_len=WINDOW_LEN,
        dim=128, depth=2, heads=8, dim_head=32,
        action_dim=3,
    ).to(device)

    trainer = LandingActionITransformerTrainer(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=1e-4),
        loss=nn.MSELoss(),
        device=device,
        results_path=Path("results/multiple_airports_landing_action_itransformer"),
        patience=20,
    )

    trainer.train(train_loader, val_loader, epochs=1000)

    trainer.save_normalization(x_mean, x_std, y_mean, y_std)

    trainer.plot_metrics(group_by_type=True)
    trainer.save_metrics()
    trainer.save_last_model()