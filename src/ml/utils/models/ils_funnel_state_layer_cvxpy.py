import math

import numpy as np
import torch
from torch import nn, Tensor

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

DEG2RAD = math.pi / 180.0


def _funnel_row_values(
    z: Tensor, fix_x: Tensor, fix_y: Tensor, sin_r: Tensor, cos_r: Tensor,
    h_thr: float, tan_gs: float, c_y0: float, kappa_y: float, c_z0: float, kappa_z: float,
) -> Tensor:
    x, y, h = z[:, 0], z[:, 1], z[:, 2]
    x_rel, y_rel = x - fix_x, y - fix_y
    s_ = -(x_rel * sin_r + y_rel * cos_r)
    e_y = x_rel * cos_r - y_rel * sin_r
    e_z = h - h_thr - s_ * tan_gs
    cy = c_y0 + kappa_y * s_
    cz = c_z0 + kappa_z * s_
    return torch.stack([cy - e_y, cy + e_y, cz - e_z, cz + e_z], dim=-1)  # (B, 4)


class ILSFunnelStateLayer(nn.Module):
    def __init__(
        self, base_model, y_mean, y_std,
        h_thr: float = 0.0, glideslope_deg: float = 3.0,
        c_y0: float = 15.0, kappa_y_deg: float = 2.5, c_z0: float = 15.0, kappa_z_deg: float = 1.0,
        s_activate: float = 15_000.0, s_min: float = 150.0, heading_gate_deg: float = 90.0,
        w_pos=(1.0, 1.0, 2.0), rho_slack: float = 1e4,
        solver_args: dict | None = None, debug: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))

        self.h_thr = h_thr
        self.tan_gs = math.tan(glideslope_deg * DEG2RAD)
        self.c_y0 = c_y0
        self.kappa_y = math.tan(kappa_y_deg * DEG2RAD)
        self.c_z0 = c_z0
        self.kappa_z = math.tan(kappa_z_deg * DEG2RAD)
        self.s_activate = s_activate
        self.s_min = s_min
        self.cos_gate = math.cos(heading_gate_deg * DEG2RAD)
        self.w_pos = w_pos
        self.rho_slack = rho_slack
        self.solver_args = solver_args or {}
        self.debug = debug

        self.layer = self._build_qp()

    def _build_qp(self) -> CvxpyLayer:
        z = cp.Variable(3)                 # x, y, h
        s = cp.Variable(4, nonneg=True)

        z_nn = cp.Parameter(3)
        A = cp.Parameter((4, 3))
        b = cp.Parameter(4)

        w = [math.sqrt(wi) for wi in self.w_pos]
        constraints = [A @ z + b + s >= 0.0]
        objective = cp.Minimize(cp.sum_squares(cp.multiply(w, z - z_nn)) + self.rho_slack * cp.sum_squares(s))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp(), "funnel QP is not DPP conform"

        return CvxpyLayer(problem, parameters=[z_nn, A, b], variables=[z, s])

    @staticmethod
    def _linear_rows(f, z0: Tensor) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            z0v = z0.detach().clone().requires_grad_(True)
            out0 = f(z0v)
            B, m = out0.shape
            k = z0v.shape[1]
            A = torch.zeros(B, m, k)
            for j in range(m):
                grad = torch.autograd.grad(out0[:, j].sum(), z0v, retain_graph=(j < m - 1))[0]
                A[:, j, :] = grad.detach()
            b = out0.detach() - torch.einsum("bmk,bk->bm", A, z0v.detach())
        return A, b

    def _decode_next_state(self, x_norm: Tensor) -> Tensor:
        raw_norm = self.base_model(x_norm)
        return raw_norm * self.y_std + self.y_mean  # (B, 6): x, y, h, v, gamma, chi

    @torch.no_grad()
    def forward(self, x_norm: Tensor, runway_pos: Tensor) -> Tensor:
        raw = self._decode_next_state(x_norm)
        x_nn, y_nn, h_nn, v_nn, gamma_nn, chi_nn = raw.unbind(dim=-1)

        fix_x, fix_y, chi_r_deg = runway_pos.unbind(dim=-1)
        chi_r = torch.deg2rad(chi_r_deg)
        sin_r, cos_r = torch.sin(chi_r), torch.cos(chi_r)

        z_nn = torch.stack([x_nn, y_nn, h_nn], dim=-1)

        x_rel_nn, y_rel_nn = x_nn - fix_x, y_nn - fix_y
        s_nn = -(x_rel_nn * sin_r + y_rel_nn * cos_r)
        chi_nn_rad = torch.deg2rad(chi_nn)
        delta = torch.atan2(torch.sin(chi_nn_rad - chi_r), torch.cos(chi_nn_rad - chi_r))
        heading_ok = (torch.cos(delta) > self.cos_gate).float()
        active = ((s_nn > self.s_min) & (s_nn < self.s_activate)).float() * heading_ok

        f = lambda z: _funnel_row_values(
            z, fix_x, fix_y, sin_r, cos_r,
            self.h_thr, self.tan_gs, self.c_y0, self.kappa_y, self.c_z0, self.kappa_z,
        )
        A, b = self._linear_rows(f, z_nn)
        gate = active.view(-1, 1, 1)
        A = A * gate
        b = torch.where(gate.view(-1, 1) > 0, b, torch.ones_like(b))

        try:
            z_opt, s_opt = self.layer(z_nn, A, b, solver_args=self.solver_args)
        except Exception as e:
            print(f"[ils-funnel] WARN: Solver failed ({e}) -> pass raw prediction through unprojected")
            z_opt, s_opt = z_nn, None

        out = raw.clone()
        out[:, 0], out[:, 1], out[:, 2] = z_opt[:, 0], z_opt[:, 1], z_opt[:, 2]

        if self.debug:
            for i in range(out.shape[0]):
                slack = f"{s_opt[i].max():.4f}" if s_opt is not None else "fallback"
                print(f"[ils-funnel] s={s_nn[i]:8.1f}  active={bool(active[i])}  "
                      f"(x,y,h)_nn=({x_nn[i]:.1f},{y_nn[i]:.1f},{h_nn[i]:.1f}) -> "
                      f"({out[i,0]:.1f},{out[i,1]:.1f},{out[i,2]:.1f})  slack={slack}")
        return out