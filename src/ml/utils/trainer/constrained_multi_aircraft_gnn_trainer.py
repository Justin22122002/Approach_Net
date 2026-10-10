import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing_extensions import override

from ml.utils.models.constraint_graph_attention_model import MultiAircraftConstrainedQPLayer
from ml.utils.models.constraint_layer_cvxpy import masked_mean_loss
from ml.utils.trainer.multi_aircraft_GNN_trainer import MultiAircraftGNNTrainer


class ConstrainedMultiAircraftGNNTrainer(MultiAircraftGNNTrainer):
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
        assert isinstance(self.model, MultiAircraftConstrainedQPLayer), \
            "self.model must be a MultiAircraftConstrainedQPLayer"
        self.input_dim: int = input_dim
        self.y_mean_t: Tensor = torch.as_tensor(y_mean, dtype=torch.float32, device=self.device)
        self.y_std_t: Tensor = torch.as_tensor(y_std, dtype=torch.float32, device=self.device)
        self.tau: float = tau
        self._w_sum: float = 0.0
        self._w_count: int = 0
        self._sep_stats: dict[str, int] = {}

    def _unpack(self, packed: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        d = self.input_dim
        x_norm = packed[:, :d]
        x_raw = packed[:, d:2 * d]
        limits = packed[:, 2 * d:2 * d + 4]
        faf_pos = packed[:, 2 * d + 4:2 * d + 6]
        if_pos = packed[:, 2 * d + 6:2 * d + 8]
        runway_length_m = packed[:, 2 * d + 8]

        aircraft_dyn = x_raw[:, 0:5]
        aircraft_pos = x_raw[:, [5, 6, 10]]
        aircraft_state = x_raw[:, 7:10]
        runway_pos = x_raw[:, [11, 12, 14]]
        h_thr = x_raw[:, 13]
        return x_norm, aircraft_state, aircraft_dyn, limits, aircraft_pos, runway_pos, faf_pos, if_pos, runway_length_m, h_thr

    @override
    def forward_pass(self, inputs: Tensor) -> Tensor:
        packed, batch_idx = inputs[:, :-1], inputs[:, -1].long()  # (N, 39), (N,)
        (x_norm, aircraft_state, aircraft_dyn, aircraft_limits, aircraft_pos, runway_pos,
         faf_pos, if_pos, runway_length_m, h_thr) = self._unpack(packed)
        out_phys = self.model(
            x_norm=x_norm,
            aircraft_state=aircraft_state,
            aircraft_dyn=aircraft_dyn,
            aircraft_limits=aircraft_limits,
            aircraft_pos=aircraft_pos,
            runway_pos=runway_pos,
            batch_idx=batch_idx,
            faf_pos=faf_pos,
            if_pos=if_pos,
            runway_length_m=runway_length_m,
            h_thr=h_thr,
        )
        if self.model.training:
            for key, val in self.model.separation.last_stats.items():
                self._sep_stats[key] = self._sep_stats.get(key, 0) + val
        return (out_phys - self.y_mean_t) / self.y_std_t

    def _raw_in_label_space(self) -> Tensor:
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

        self._w_sum += w.sum().item()
        self._w_count += w.numel()

        return loss

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
            print(f"[w] mean over all aircraft per epoch: {self._w_sum / self._w_count:.4f}")
            self._w_sum, self._w_count = 0.0, 0
        if self._sep_stats:
            s = self._sep_stats
            print(f"[sep] scenes={s.get('scenes', 0)}  multi-aircraft={s.get('scenes_multi', 0)}  "
                  f"active={s.get('scenes_active', 0)}  failed={s.get('scenes_failed', 0)}")
            self._sep_stats = {}
        return super().evaluate(loader)

    @override
    def save_best_model(self, val_loss: float) -> None:
        self.results_path.mkdir(parents=True, exist_ok=True)

        if val_loss < self.best_val_loss:
            torch.save(self.model.base_model.state_dict(), self.results_path / "best_base_model.pth")
            torch.save(self.model.state_dict(), self.results_path / "best_constrained_model.pth")
            print(f"Best model updated with validation loss {val_loss:.4f}.")

    @override
    def save_last_model(self) -> None:
        self.results_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.base_model.state_dict(), self.results_path / "last_base_model.pth")
        torch.save(self.model.state_dict(), self.results_path / "last_constrained_model.pth")
        print("Last model saved.")