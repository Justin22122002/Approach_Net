import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

from ml.utils.trainer.base_trainer import BaseTrainer, TrainingMetrics


class LandingActionITransformerTrainer(BaseTrainer[Tensor, Tensor, Tensor, TrainingMetrics]):
    @override
    def forward_pass(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)  # (B, 3)

    @override
    def calculate_loss(self, outputs: Tensor, labels: Tensor) -> Tensor:
        return self.loss(outputs, labels)

    @override
    def calculate_accuracy(self, outputs: Tensor, labels: Tensor) -> float:
        with torch.no_grad():
            return torch.mean(torch.abs(outputs - labels)).item()

    def save_normalization(
        self,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ) -> None:
        self.results_path.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            self.results_path / "normalization_x.csv",
            np.vstack([x_mean, x_std]),
            delimiter=",", header="x_mean,x_std", comments="",
        )
        np.savetxt(
            self.results_path / "normalization_y.csv",
            np.vstack([y_mean, y_std]),
            delimiter=",", header="y_mean,y_std", comments="",
        )

    @staticmethod
    def load_normalization(x_path, y_path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.loadtxt(x_path, delimiter=",", skiprows=1)
        y = np.loadtxt(y_path, delimiter=",", skiprows=1)
        return (
            x[0].astype(np.float32), x[1].astype(np.float32),
            y[0].astype(np.float32), y[1].astype(np.float32),
        )