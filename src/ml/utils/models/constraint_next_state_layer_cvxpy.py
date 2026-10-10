import math

import numpy as np
import torch
from torch import nn, Tensor

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from ml.utils.models.constraint_layer_cvxpy import check_gradient_flow, AircraftFunnelQPLayer

DEG2RAD = math.pi / 180.0


def _wrap(a: Tensor) -> Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


class ConstraintNextStateLayer(nn.Module):
    """
    Kinematic (no aerodynamics) safety projection of the predicted next state (x, y, h, v, gamma, chi).
    Mirrors AircraftFunnelQPLayer / AircraftCombinedQPLayer:
      - RNAV (GPS) LPV funnel as hard (slacked) constraints on the next position
          lateral : |e_y| <= tan(LGSA) * (s + d_GARP)                      [FAA 8260.58D, Formula 3-1-1]
          vertical: |e_z| <= clamp(tan(0.25*GPA) * s, fsd_v_min, fsd_v_max) [RTCA DO-229]
        active for s_min < s < s_activate_funnel (FAF distance if faf_pos given) + heading gate
      - soft tracking cost (never infeasible), active for s < s_activate_track (IF distance if if_pos given)
          lateral : heading follows the 30 deg localizer-style intercept vector field
                    delta_des = -chi_int * (2/pi) * atan(k_vf * e_y)
          vertical: pull e_z towards the glide path
    QP variable d = (dx, dy, dh, dchi[rad]) is the deviation from the NN prediction
    -> local coordinates, well conditioned (no 1e4 m offsets in the QP).
    v and gamma are passed through unchanged.
    """

    def __init__(
        self,
        base_model: nn.Module,
        y_mean: float | Tensor,
        y_std: float | Tensor,
        solver: str | None = None,
        h_thr: float = 0.0,
        glideslope_deg: float = 3.0,
        thr_half_width_lat_m: float = 106.68,   # 350 ft floor, Order 8260.58D Formula 3-1-1
        fsd_v_ratio: float = 0.25,              # RTCA DO-229: vertical FSD = 0.25 * GPA
        fsd_v_min_m: float = 45.0,
        fsd_v_max_m: float = 150.0,
        runway_length_m: float | None = None,   # fallback, else 9023 ft FPAP floor
        lgsa_deg: float | None = None,
        intercept_angle_deg: float = 30.0,      # max. intercept angle onto the centerline
        k_vf: float = 1e-3,                     # 1/m, vector-field gain
        w_track: tuple[float, float] = (100.0, 0.05),  # (lateral heading [1/rad^2], vertical e_z [1/m^2])
        s_activate_funnel: float = 10_000.0,    # fallback if no faf_pos is passed to forward()
        s_activate_track: float = 20_000.0,     # fallback if no if_pos is passed to forward()
        s_min: float = 150.0,
        heading_gate_deg: float = 90.0,
        w_state: tuple[float, float, float, float] = (1.0, 1.0, 2.0, 100.0),  # x, y, h [1/m^2], chi [1/rad^2]
        rho_slack: float = 1e4,
        rho_track: float = 1.0,
        solver_args: dict | None = None,
        debug: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))
        self.solver = solver
        self.solver_args = solver_args or {}
        self.debug = debug
        self.fail_rate_warn = 0.1
        self.last_solved_mask: Tensor | None = None

        self.h_thr = h_thr  # 0.0 cause the data is centered around 0.0 of the airport
        self.gs = glideslope_deg * DEG2RAD
        self.tan_gs = math.tan(self.gs)
        self.thr_half_width_lat_m = thr_half_width_lat_m
        self.lgsa_deg = lgsa_deg

        # fallback LPV lateral geometry (same formulas as the per-sample version)
        rl = torch.tensor([runway_length_m if runway_length_m is not None else float("nan")])
        d_garp, tan_lgsa = AircraftFunnelQPLayer.lpv_lateral_geometry(rl, thr_half_width_lat_m, lgsa_deg)
        self.d_garp = d_garp.item()
        self.tan_lgsa = tan_lgsa.item()

        self.fsd_v_min = fsd_v_min_m
        self.fsd_v_max = fsd_v_max_m
        self.tan_fsdv = math.tan(fsd_v_ratio * self.gs)

        self.intercept_angle_deg = intercept_angle_deg
        self.k_vf = k_vf
        self.w_track = w_track
        self.s_activate_funnel = s_activate_funnel
        self.s_activate_track = s_activate_track
        self.s_min = s_min
        self.cos_gate = math.cos(heading_gate_deg * DEG2RAD)
        self.w_state = w_state
        self.rho_slack = rho_slack
        self.rho_track = rho_track

        self.layer = self._build_qp(n_rows=4, n_track_rows=2)

    def _build_qp(self, n_rows: int, n_track_rows: int) -> CvxpyLayer:
        d = cp.Variable(4)                    # dx, dy, dh, dchi
        s = cp.Variable(n_rows, nonneg=True)

        A = cp.Parameter((n_rows, 4))
        b = cp.Parameter(n_rows)
        A_track = cp.Parameter((n_track_rows, 4))
        b_track = cp.Parameter(n_track_rows)

        w = np.sqrt(np.asarray(self.w_state, dtype=float))
        constraints = [A @ d + b + s >= 0.0]
        objective = cp.Minimize(
            cp.sum_squares(cp.multiply(w, d))
            + self.rho_slack * cp.sum_squares(s)
            + self.rho_track * cp.sum_squares(A_track @ d + b_track)
        )
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp(), "next-state funnel QP is not DPP conform"
        return CvxpyLayer(problem, parameters=[A, b, A_track, b_track], variables=[d, s], solver=self.solver)

    @staticmethod
    def compute_rows(
        z0: Tensor,                          # (B, 3) predicted x, y, h
        delta0: Tensor,                      # (B,) predicted heading relative to runway [rad], wrapped
        runway_pos: Tensor,                  # (B, 3) fix_x, fix_y, chi_r_deg
        h_thr: float | Tensor,
        tan_gs: float,
        tan_lgsa: float | Tensor,
        d_garp: float | Tensor,
        tan_fsdv: float,
        fsd_v_min: float,
        fsd_v_max: float,
        s_activate_funnel: float | Tensor,
        s_activate_track: float | Tensor,
        s_min: float,
        cos_gate: float,
        intercept_angle_deg: float,
        k_vf: float,
        w_track: tuple[float, float],
    ):
        x, y, h = z0.unbind(dim=-1)
        fix_x, fix_y, chi_r_deg = runway_pos.unbind(dim=-1)
        chi_r = torch.deg2rad(chi_r_deg)
        sin_r, cos_r = torch.sin(chi_r), torch.cos(chi_r)
        zeros, ones = torch.zeros_like(x), torch.ones_like(x)

        x_rel, y_rel = x - fix_x, y - fix_y
        s_ = -(x_rel * sin_r + y_rel * cos_r)          # distance to runway fix along landing axis
        e_y = x_rel * cos_r - y_rel * sin_r            # lateral error
        e_z = h - h_thr - s_ * tan_gs                  # vertical error to the glide path

        # gradients w.r.t. d = (dx, dy, dh, dchi) -> all exactly linear except the cz clamp
        g_s = torch.stack([-sin_r, -cos_r, zeros, zeros], dim=1)
        g_ey = torch.stack([cos_r, -sin_r, zeros, zeros], dim=1)
        g_ez = torch.stack([zeros, zeros, ones, zeros], dim=1) - tan_gs * g_s

        # lateral LPV half-width, linear in s
        tan_lgsa_t = torch.as_tensor(tan_lgsa, dtype=x.dtype, device=x.device).expand_as(x)
        cy = tan_lgsa_t * (s_ + d_garp)
        g_cy = tan_lgsa_t.unsqueeze(1) * g_s

        # vertical FSD, clamped ramp -> piecewise linear, linearized at the prediction
        raw_wz = tan_fsdv * s_
        ramp_active = (raw_wz >= fsd_v_min) & (raw_wz <= fsd_v_max)
        cz = torch.clamp(raw_wz, min=fsd_v_min, max=fsd_v_max)
        dcz_ds = torch.where(ramp_active, torch.full_like(s_, tan_fsdv), zeros)
        g_cz = dcz_ds.unsqueeze(1) * g_s

        # funnel rows (>= 0), values at the prediction (b) + gradients (A)
        b = torch.stack([cy - e_y, cy + e_y, cz - e_z, cz + e_z], dim=1)
        A = torch.stack([g_cy - g_ey, g_cy + g_ey, g_cz - g_ez, g_cz + g_ez], dim=1)

        heading_ok = (torch.cos(delta0) > cos_gate).float()
        active = ((s_ > s_min) & (s_ < s_activate_funnel)).float() * heading_ok
        A = A * active.view(-1, 1, 1)
        b = torch.where(active.view(-1, 1) > 0, b, torch.ones_like(b))  # neutral row: 0*d + 1 + s >= 0

        # --- soft tracking cost ---
        # lateral: heading should follow the 30 deg intercept vector field
        chi_int = intercept_angle_deg * DEG2RAD
        delta_des = -chi_int * (2.0 / math.pi) * torch.atan(k_vf * e_y)
        ddes_dey = -chi_int * (2.0 / math.pi) * k_vf / (1.0 + (k_vf * e_y) ** 2)
        r_y = _wrap(delta0 - delta_des)
        g_ry = torch.stack([zeros, zeros, zeros, ones], dim=1) - ddes_dey.unsqueeze(1) * g_ey
        # vertical: pull e_z onto the glide path
        r_z = e_z
        g_rz = g_ez

        wl, wv = math.sqrt(w_track[0]), math.sqrt(w_track[1])
        track_active = (s_ < s_activate_track).float() * heading_ok
        A_track = torch.stack([wl * g_ry, wv * g_rz], dim=1) * track_active.view(-1, 1, 1)
        b_track = torch.stack([wl * r_y, wv * r_z], dim=1) * track_active.view(-1, 1)

        return A, b, A_track, b_track, s_, e_y, e_z, active, track_active

    def _solve(self, A: Tensor, b: Tensor, A_track: Tensor, b_track: Tensor) -> tuple[Tensor, Tensor]:
        B = A.shape[0]
        dev, dt = A.device, A.dtype
        args = [A, b, A_track, b_track]
        args_cpu = [t.cpu() for t in args] if dev.type != "cpu" else args

        finite = torch.stack([torch.isfinite(t).flatten(1).all(dim=1) for t in args_cpu], dim=0).all(dim=0)

        d_out = torch.zeros(B, 4, dtype=dt, device=dev)
        s_out = torch.zeros(B, A.shape[1], dtype=dt, device=dev)
        solved = torch.zeros(B, dtype=torch.bool, device=dev)

        idx = finite.nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            try:
                d_sub, s_sub = self.layer(*[t[idx] for t in args_cpu], solver_args=self.solver_args)
                d_out[idx] = d_sub.to(dev, dtype=dt)
                s_out[idx] = s_sub.to(dev, dtype=dt)
                solved[idx] = True
            except Exception as e:
                print(f"[ns-funnel] WARN: batched solve over {idx.numel()} samples failed ({e}) -> passing raw prediction through")

        # fallback: unprojected prediction (d = 0), slack = constraint violation
        un = (~solved).nonzero(as_tuple=True)[0]
        if un.numel() > 0:
            s_out[un] = torch.clamp(-torch.nan_to_num(b[un]), min=0.0)

        self.last_solved_mask = solved
        fail_rate = 1.0 - solved.float().mean().item()
        if fail_rate > self.fail_rate_warn:
            print(f"[ns-funnel] WARN: high fail rate {fail_rate:.1%} ({un.numel()}/{B} samples unprojected)")
        return d_out, s_out

    def _decode_next_state(self, x_norm: Tensor, raw_norm: Tensor | None = None) -> Tensor:
        if raw_norm is None:
            raw_norm = self.base_model(x_norm)
        self.last_raw_norm = raw_norm
        return raw_norm * self.y_std + self.y_mean  # (B, 6): x, y, h, v, gamma, chi

    def forward(
        self,
        x_norm: Tensor,
        runway_pos: Tensor,                     # (B, 3) fix_x, fix_y, chi_r_deg
        faf_pos: Tensor | None = None,          # (B, 2) -> funnel activation distance; None -> s_activate_funnel
        if_pos: Tensor | None = None,           # (B, 2) -> tracking activation distance; None -> s_activate_track
        runway_length_m: Tensor | None = None,  # (B,) NaN -> 9023 ft floor; None -> constructor value
        raw_norm: Tensor | None = None,         # (B, 6) precomputed normalized prediction
        h_thr: Tensor | None = None,            # (B,) runway fix height incl. TCH; None -> constructor value
    ) -> Tensor:
        raw = self._decode_next_state(x_norm, raw_norm)
        x_nn, y_nn, h_nn, v_nn, gamma_nn, chi_nn = raw.unbind(dim=-1)

        chi_r = torch.deg2rad(runway_pos[:, 2])
        delta0 = _wrap(torch.deg2rad(chi_nn) - chi_r)
        z0 = torch.stack([x_nn, y_nn, h_nn], dim=-1)

        s_act_funnel = AircraftFunnelQPLayer.fix_distance(faf_pos, runway_pos) if faf_pos is not None else self.s_activate_funnel
        s_act_track = AircraftFunnelQPLayer.fix_distance(if_pos, runway_pos) if if_pos is not None else self.s_activate_track
        if runway_length_m is not None:
            d_garp, tan_lgsa = AircraftFunnelQPLayer.lpv_lateral_geometry(
                runway_length_m, self.thr_half_width_lat_m, self.lgsa_deg)
        else:
            d_garp, tan_lgsa = self.d_garp, self.tan_lgsa

        A, b, A_track, b_track, s_, e_y, e_z, active, track_active = self.compute_rows(
            z0, delta0, runway_pos, self.h_thr if h_thr is None else h_thr, self.tan_gs,
            tan_lgsa, d_garp, self.tan_fsdv, self.fsd_v_min, self.fsd_v_max,
            s_act_funnel, s_act_track, self.s_min, self.cos_gate,
            self.intercept_angle_deg, self.k_vf, self.w_track,
        )

        d_opt, s_opt = self._solve(A, b, A_track, b_track)

        xyz = z0 + d_opt[:, :3]
        chi = chi_nn + torch.rad2deg(d_opt[:, 3])  # keeps the heading convention of the prediction
        out = torch.cat([xyz, v_nn.unsqueeze(1), gamma_nn.unsqueeze(1), chi.unsqueeze(1)], dim=1)

        if self.debug:
            for i in range(out.shape[0]):
                saf = s_act_funnel[i] if isinstance(s_act_funnel, Tensor) else s_act_funnel
                sat = s_act_track[i] if isinstance(s_act_track, Tensor) else s_act_track
                print(f"[ns-funnel] s={s_[i]:8.1f}  e_y={e_y[i]:+7.1f}  e_z={e_z[i]:+6.1f}  "
                      f"funnel={bool(active[i])} track={bool(track_active[i])}  |  "
                      f"(x,y,h,chi)=({x_nn[i]:.1f},{y_nn[i]:.1f},{h_nn[i]:.1f},{chi_nn[i]:.1f}) -> "
                      f"({out[i,0]:.1f},{out[i,1]:.1f},{out[i,2]:.1f},{out[i,5]:.1f})  "
                      f"slack={s_opt[i].max():.4f}  s_act_funnel={saf:8.1f}  s_act_track={sat:8.1f}")
        return out


if __name__ == "__main__":
    class _LearnableStateModel(nn.Module):
        def __init__(self, states_norm: Tensor):
            super().__init__()
            self.states_norm = nn.Parameter(states_norm.clone())

        def forward(self, x: Tensor) -> Tensor:
            return self.states_norm

    torch.manual_seed(0)
    B = 3
    fix_x, fix_y, chi_r_deg = 0.0, 0.0, 100.0

    def make_pos(s, e_y):
        chi_r = chi_r_deg * DEG2RAD
        return (fix_x - s * math.sin(chi_r) + e_y * math.cos(chi_r),
                fix_y - s * math.cos(chi_r) - e_y * math.sin(chi_r))

    h_thr = 4.0
    h_gp = lambda s: h_thr + s * math.tan(3.0 * DEG2RAD)

    # (x, y, h, v, gamma, chi): on path / lateral outside funnel / far out at IF, 60 deg off heading
    states = []
    for s, ey, dh, dchi in [(5000.0, 5.0, 5.0, 3.0), (5000.0, 400.0, 0.0, 0.0), (17_000.0, 1500.0, 80.0, 60.0)]:
        x, y = make_pos(s, ey)
        states.append([x, y, h_gp(s) + dh, 70.0, -3.0, chi_r_deg + dchi])
    states_phys = torch.tensor(states, dtype=torch.float32)

    y_mean = np.array([0.0, 0.0, 500.0, 80.0, -3.0, 180.0], dtype=np.float32)
    y_std = np.array([10_000.0, 10_000.0, 500.0, 20.0, 2.0, 100.0], dtype=np.float32)
    states_norm = (states_phys - torch.from_numpy(y_mean)) / torch.from_numpy(y_std)

    runway_pos = torch.tensor([[fix_x, fix_y, chi_r_deg]]).repeat(B, 1)
    faf_pos = torch.tensor([make_pos(10_000.0, 0.0)]).repeat(B, 1)
    if_pos = torch.tensor([make_pos(18_000.0, 0.0)]).repeat(B, 1)
    runway_length_m = torch.tensor([2200.0, 11_500.0 * 0.3048, float("nan")])
    x_dummy = torch.zeros(B, 15)

    results = {}
    m1 = _LearnableStateModel(states_norm)
    l1 = ConstraintNextStateLayer(m1, y_mean, y_std, h_thr=h_thr, debug=True)
    results["next_state"] = check_gradient_flow("next_state", l1, m1, y_mean, y_std, x_dummy, runway_pos)

    m2 = _LearnableStateModel(states_norm)
    l2 = ConstraintNextStateLayer(m2, y_mean, y_std, h_thr=h_thr, debug=True)
    results["next_state_fix"] = check_gradient_flow(
        "next_state_fix", l2, m2, y_mean, y_std, x_dummy, runway_pos, faf_pos, if_pos, runway_length_m)

    print(f"\n{'=' * 70}\nsummary\n{'=' * 70}")
    for name, passed in results.items():
        print(f"  {name:16s}: {'PASSED' if passed else 'FAILED'}")