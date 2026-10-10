import torch
from torch import nn, Tensor

from ml.utils.models.constraint_joint_speration_layer_cvxpy import AircraftSceneSeparationQPLayer
from ml.utils.models.constraint_layer_cvxpy import AircraftCombinedQPLayer


class MultiAircraftConstrainedQPLayer(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        y_mean: float | Tensor,
        y_std: float | Tensor,
        solver: str | None,
        use_separation: bool = True,
        keep_aircraft_rows: bool = True,
        separation_kwargs: dict | None = None,
        **combined_kwargs,
    ) -> None:
        super().__init__()
        self.base_model: nn.Module = base_model
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))

        combined_kwargs.setdefault("debug", False)
        self.combined: AircraftCombinedQPLayer = AircraftCombinedQPLayer(
            base_model=nn.Identity(),
            y_mean=y_mean,
            y_std=y_std,
            solver=solver,
            **combined_kwargs,
        )
        self.separation: AircraftSceneSeparationQPLayer = AircraftSceneSeparationQPLayer(
            solver=solver,
            **(separation_kwargs or {}),
        )
        self.use_separation: bool = use_separation
        self.keep_aircraft_rows: bool = keep_aircraft_rows

        self.last_raw_norm: Tensor | None = None
        self.last_solved_mask: Tensor | None = None
        self.last_combined_out: Tensor | None = None

    def forward(
        self,
        x_norm: Tensor,
        aircraft_state: Tensor,
        aircraft_dyn: Tensor,
        aircraft_limits: Tensor,
        aircraft_pos: Tensor,
        runway_pos: Tensor,
        batch_idx: Tensor | None = None,
        faf_pos: Tensor | None = None,
        if_pos: Tensor | None = None,
        runway_length_m: Tensor | None = None,
        h_thr: Tensor | None = None,
        use_separation: bool | None = None,
    ) -> Tensor:
        if batch_idx is None:
            batch_idx = torch.zeros(x_norm.shape[0], dtype=torch.long, device=x_norm.device)

        raw_norm = self.base_model(x_norm, batch_idx)
        self.last_raw_norm = raw_norm

        out = self.combined(
            x_norm=x_norm,
            aircraft_state=aircraft_state,
            aircraft_dyn=aircraft_dyn,
            aircraft_limits=aircraft_limits,
            aircraft_pos=aircraft_pos,
            runway_pos=runway_pos,
            faf_pos=faf_pos,
            if_pos=if_pos,
            runway_length_m=runway_length_m,
            raw_norm=raw_norm,
            h_thr=h_thr,
        )
        self.last_combined_out = out
        self.last_solved_mask = self.combined.last_solved_mask

        if self.use_separation if use_separation is None else use_separation:
            out = self.separation(
                out, aircraft_state, aircraft_dyn, aircraft_limits, aircraft_pos, batch_idx=batch_idx,
                A_air=self.combined.last_A if self.keep_aircraft_rows else None,
                b_air=self.combined.last_b if self.keep_aircraft_rows else None,
            )
        return out
