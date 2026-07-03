from pathlib import Path
from torch import nn, optim
from torch.utils.data import DataLoader

from ml.utils.data_loader.landing_action_dataset import LandingActionDataset
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from ml.utils.models.mlp_model import LandingActionResNet
from ml.utils.trainer.mlp_trainer import LandingActionTrainer

if __name__ == "__main__":
    LandingActionTrainer.set_seed()

    flight_dataset = LazyFlightParquetDataset("../data/CYVR/20260628_1254") # "../data/CYVR/train/20260701_1614" using the train data
    dataset = LandingActionDataset(
        flight_dataset=flight_dataset,
        mode="bspline_state",
        max_dt=5.0
    )

    print(f"Samples: {len(dataset)}, input_dim: {dataset.input_dim}")

    # Train/Val Split
    train_set, val_set = dataset.flight_level_split(val_frac=0.2, seed=42)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

    device = LandingActionTrainer.get_device()

    model = LandingActionResNet(
        input_dim=dataset.input_dim,
        hidden_dim=256,
        n_blocks=4,
        output_dim=3
    )

    trainer = LandingActionTrainer(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss=nn.MSELoss(),
        device=device,
        results_path=Path("results/landing_action_resnet"),
    )

    trainer.train(train_loader, val_loader, epochs=50)

    trainer.save_normalization(dataset.x_mean, dataset.x_std, dataset.y_mean, dataset.y_std)
