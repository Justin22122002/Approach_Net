import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing_extensions import override
import torch.nn.functional as F

from ml.utils.models.constraint_layer_cvxpy import AircraftEnvelopeQPLayer, _QPLayerBase, masked_mean_loss
from ml.utils.trainer.mlp_trainer import LandingActionTrainer


class ConstrainedLandingActionTrainer(LandingActionTrainer):
    # todo: override trainer to also save the base model without the solver layer extra
    def __init__(
            self,
            *args,
            y_mean: np.ndarray,
            y_std: np.ndarray,
            tau: float = 0.05,
            input_dim: int = 15,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.model, _QPLayerBase), "self.model must be a CBF-QP safety layer"
        self.input_dim: int = input_dim
        self.y_mean_t: Tensor = torch.as_tensor(y_mean, dtype=torch.float32, device=self.device)
        self.y_std_t: Tensor = torch.as_tensor(y_std, dtype=torch.float32, device=self.device)
        self.debug_worst_samples: bool = False
        self.tau: float = tau
        self._w_sum: float = 0.0
        self._w_count: int = 0

    def _unpack(self, packed: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        d = self.input_dim  # 15
        x_norm = packed[:, :d]
        x_raw = packed[:, d:2 * d]
        limits = packed[:, 2 * d:2 * d + 4]

        aircraft_dyn = x_raw[:, 0:5]
        aircraft_pos = x_raw[:, [5, 6, 10]]
        aircraft_state = x_raw[:, 7:10]
        runway_pos = x_raw[:, [11, 12, 14]]
        return x_norm, aircraft_state, aircraft_dyn, limits, aircraft_pos, runway_pos

    @override
    def forward_pass(self, inputs: Tensor) -> Tensor:
        x_norm, aircraft_state, aircraft_dyn, aircraft_limits, aircraft_pos, runway_pos = self._unpack(inputs)
        out_phys = self.model(x_norm, aircraft_state, aircraft_dyn, aircraft_limits, aircraft_pos, runway_pos)
        return (out_phys - self.y_mean_t) / self.y_std_t

    #@override
    #def calculate_loss(self, outputs: Tensor, labels: Tensor) -> Tensor:
    #    per_sample = (outputs - labels).pow(2).mean(dim=-1)
    #    loss = masked_mean_loss(per_sample, self.model)
    #    if loss is None:
    #        return torch.zeros((), device=outputs.device, dtype=outputs.dtype, requires_grad=True)
    #    return loss

    def _raw_in_label_space(self) -> Tensor:
        # last_raw_norm is normalised with the layer buffer,
        raw_phys = self.model.last_raw_norm * self.model.y_std + self.model.y_mean
        return (raw_phys - self.y_mean_t) / self.y_std_t

    @override
    def calculate_loss(self, outputs: Tensor, labels: Tensor) -> Tensor:
        # Validation / Eval: pure mse loss
        if not self.model.training:
            per_sample = (outputs - labels).pow(2).mean(dim=-1)
            loss = masked_mean_loss(per_sample, self.model)
            if loss is None:
                return torch.zeros((), device=outputs.device, dtype=outputs.dtype, requires_grad=True)
            return loss

        # Training
        raw = self._raw_in_label_space()
        mask = self.model.last_solved_mask
        solved = mask.float() if mask is not None else torch.ones(outputs.shape[0], device=outputs.device)

        with torch.no_grad():
            shift = (outputs - raw).pow(2).mean(dim=-1)
            w = torch.exp(-shift / self.tau) * solved

        mse_qp = (outputs - labels).pow(2).mean(dim=-1)
        mse_raw = (raw - labels).pow(2).mean(dim=-1)
        # Fallback-Samples where w=0 -> only raw loss: mse_qp will be erased (non zero gradients)
        mse_qp = torch.where(solved > 0, mse_qp, torch.zeros_like(mse_qp))
        loss = (w * mse_qp + (1.0 - w) * mse_raw).mean()

        #print(f"w_mean={w.mean():.3f}  w<0.1={(w < 0.1).float().mean():.1%}  "
        #      f"median_shift={shift[shift > 1e-4].median() if (shift > 1e-4).any() else 0:.4f}")

        self._w_sum += w.sum().item()
        self._w_count += w.numel()

        return loss

    # todo: like mean square error
    #@override
    #def calculate_loss(self, outputs: Tensor, labels: Tensor) -> Tensor:
    #    # zulaessiger QP-Korrektur) dominieren den Batch-Loss nicht mehr so stark.
    #    per_sample = F.huber_loss(outputs, labels, delta=1.0, reduction="none").mean(dim=-1)
    #
    #    if self.debug_worst_samples and per_sample.numel() > 0:
    #        worst_idx = per_sample.argsort(descending=True)[:5]
    #        mask = self.model.last_solved_mask
    #        for i in worst_idx.tolist():
    #            slack_str = "n/a"
    #            if mask is not None:
    #                slack_str = "solved" if mask[i] else "FALLBACK"
    #            print(f"[worst-sample] loss={per_sample[i].item():.3f}  status={slack_str}")
    #
    #    loss = masked_mean_loss(per_sample, self.model)
    #    if loss is None:
    #        return torch.zeros((), device=outputs.device, dtype=outputs.dtype, requires_grad=True)
    #    return loss

    @override
    def calculate_accuracy(self, outputs: Tensor, labels: Tensor) -> float:
        mask = self.model.last_solved_mask
        diff = torch.abs(outputs - labels)
        if mask is not None and mask.any():
            diff = diff[mask]
        return diff.mean().item()

    @override
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        if self._w_count > 0:
            print(f"[w] mean over all samples per epoch: {self._w_sum / self._w_count:.4f}")
            self._w_sum, self._w_count = 0.0, 0
        return super().evaluate(loader)