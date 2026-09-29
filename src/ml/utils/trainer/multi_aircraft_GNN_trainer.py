import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

from ml.utils.data_loader.multi_aircraft_landing_action_dataset import MultiAircraftTimestepView
from ml.utils.trainer.base_trainer import BaseTrainer, TrainingMetrics


def graph_collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """
    BaseTrainer's generic bound is INPUTS: Tensor (it calls inputs.to(device)
    directly), so batch_idx can't be passed as a second object alongside x
    -> it is packed as an extra last column of x instead.

    x: (N, input_dim), y: (N, action_dim), batch_idx: (N,)
    -> inputs: (N, input_dim + 1)   [:, :-1] = x   [:, -1] = batch_idx (as float)
    -> labels: (N, action_dim)
    """
    x, y, batch_idx = MultiAircraftTimestepView.collate_fn(batch)
    inputs = torch.cat([x, batch_idx.float().unsqueeze(1)], dim=1)  # (N, input_dim+1)
    return inputs, y


class MultiAircraftGNNTrainer(BaseTrainer[Tensor, Tensor, Tensor, TrainingMetrics]):
    @override
    def forward_pass(self, inputs: Tensor) -> Tensor:
        x, batch_idx = inputs[:, :-1], inputs[:, -1].long()  # (N, input_dim), (N,)
        return self.model(x, batch_idx)  # (N, action_dim)

    @override
    def calculate_loss(self, outputs: Tensor, labels: Tensor) -> Tensor:
        return self.loss(outputs, labels)

    @override
    def calculate_accuracy(self, outputs: Tensor, labels: Tensor) -> float:
        return torch.mean(torch.abs(outputs - labels)).item()  # MAE, same as LandingActionTrainer

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