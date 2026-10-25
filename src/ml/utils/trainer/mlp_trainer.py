import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

from ml.utils.trainer.base_trainer import BaseTrainer, TrainingMetrics


class LandingActionTrainer(BaseTrainer[Tensor, Tensor, Tensor, TrainingMetrics]):
    @override
    def forward_pass(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)

    @override
    def calculate_loss(self, outputs: Tensor, labels: Tensor) -> Tensor:
        return self.loss(outputs, labels)

    @override
    def calculate_accuracy(self, outputs: Tensor, labels: Tensor) -> float:
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
            self.results_path / f"normalization_x.csv",
            np.vstack([x_mean, x_std]),
            delimiter=",", header="x_mean,x_std", comments="",
        )
        np.savetxt(
            self.results_path / f"normalization_y.csv",
            np.vstack([y_mean, y_std]),
            delimiter=",", header="y_mean,y_std", comments="",
        )
