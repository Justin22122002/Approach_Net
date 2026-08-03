from pathlib import Path

from torch import nn, optim
from torch.utils.data import DataLoader

from ml.utils.data_loader.landing_state_forecast_dataset import LandingActionWindowDataset
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from ml.utils.models.landing_action_itransformer import LandingActionITransformer
from ml.utils.trainer.landing_forecast_trainer import LandingActionITransformerTrainer
from utils.airport import Airport

WINDOW_LEN = 16

if __name__ == "__main__":
    LandingActionITransformerTrainer.set_seed()

    icao = "CYVR"
    airport = Airport.load_from_folder(base_path=Path("../data/"), icao=icao)
    flight_dataset = LazyFlightParquetDataset(f"../data/{icao}/train/20260701_1614")

    dataset = LandingActionWindowDataset(
        flight_dataset,
        mode="bspline_state",
        airport_center_latlon=(airport.lat, airport.lon),
        airport_elevation_m=airport.arinc_424_runway[0].runway_altitude * 0.3048,
        window_len=WINDOW_LEN,
    )
    print(f"windows: {len(dataset)}, flights: {len(dataset._cache)}")

    train_set, val_set = dataset.flight_level_split(val_frac=0.2, seed=42)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    device = LandingActionITransformerTrainer.get_device()

    model = LandingActionITransformer(
        num_variates=dataset.num_variates,  # 15
        lookback_len=WINDOW_LEN,
        dim=128, depth=2, heads=8, dim_head=32,
        action_dim=3,
    ).to(device)

    trainer = LandingActionITransformerTrainer(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=1e-4),
        loss=nn.MSELoss(),
        device=device,
        results_path=Path("results/landing_action_itransformer"),
        patience=20,
    )

    trainer.train(train_loader, val_loader, epochs=1000)

    assert dataset.x_mean is not None and dataset.y_mean is not None
    trainer.save_normalization(dataset.x_mean, dataset.x_std, dataset.y_mean, dataset.y_std)

    trainer.plot_metrics(group_by_type=True)
    trainer.save_metrics()
    trainer.save_last_model()