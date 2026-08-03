import math

import numpy as np
import torch
from torch import nn, Tensor

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

DEG2RAD = math.pi / 180.0


class AircraftFormulasTorch:
    """1:1-mirror of AircraftKinematicsStateCasadi (batched, torch)."""
    g = 9.81
    R = 287.05
    T = 288.15
    H = (R * T) / g
    rho0 = 1.225

    @classmethod
    def air_density(cls, h: Tensor) -> Tensor:
        return cls.rho0 * torch.exp(-h / cls.H)

    @classmethod
    def required_lift_coefficient(cls, h, v, load_factor, mass, area) -> Tensor:
        rho = cls.air_density(h)
        return 2.0 * load_factor * mass * cls.g / (rho * v ** 2 * area)

    @classmethod
    def drag_coefficient(cls, cl, C_D0, k) -> Tensor:
        return C_D0 + k * cl ** 2

    @classmethod
    def drag_force(cls, h, v, cl, C_D0, k, area) -> Tensor:
        rho = cls.air_density(h)
        CD = cls.drag_coefficient(cl, C_D0, k)
        return 0.5 * rho * v ** 2 * CD * area

    @classmethod
    def stall_speed(cls, h, mass, cl_max, area) -> Tensor:
        rho = cls.air_density(h)
        return torch.sqrt(2.0 * mass * cls.g / (rho * cl_max * area))


class _QPLayerBase(nn.Module):
    def __init__(self, base_model, y_mean, y_std, w_action=(4.0, 1.0, 1.0),
                 rho_slack: float = 1e4, solver_args: dict | None = None, debug: bool = True):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))
        self.w_action = w_action
        self.rho_slack = rho_slack
        self.solver_args = solver_args or {}
        self.debug = debug

    def _decode_action(self, x_norm: Tensor, t_max: Tensor):
        raw_norm = self.base_model(x_norm)
        raw_phys = raw_norm * self.y_std + self.y_mean
        T_nn, n_nn, roll_nn = raw_phys.unbind(dim=-1)
        phi_nn = torch.deg2rad(roll_nn)
        u_nn = torch.stack([
            torch.clamp(T_nn / t_max, 0.0, 1.0),
            n_nn * torch.cos(phi_nn),
            n_nn * torch.sin(phi_nn),
        ], dim=1)
        return T_nn, n_nn, roll_nn, u_nn

    @staticmethod
    def _box_constraints(u, n_min, n_max, tanphi):
        return [
            u[0] >= 0.0,
            u[0] <= 1.0,
            cp.norm(u[1:3], 2) <= n_max,
            u[1] >= n_min,
            u[2] <= tanphi * u[1],
            -u[2] <= tanphi * u[1],
        ]

    def _build_qp(self, n_rows: int) -> CvxpyLayer:
        u = cp.Variable(3)                 # [T̃, uv, uh]
        s = cp.Variable(n_rows, nonneg=True)

        u_nn = cp.Parameter(3)
        n_min = cp.Parameter(nonneg=True)
        n_max = cp.Parameter(nonneg=True)
        tanphi = cp.Parameter(nonneg=True)
        A = cp.Parameter((n_rows, 3))
        b = cp.Parameter(n_rows)

        w = np.sqrt(np.asarray(self.w_action, dtype=float))
        constraints = self._box_constraints(u, n_min, n_max, tanphi)
        constraints.append(A @ u + b + s >= 0.0)
        objective = cp.Minimize(cp.sum_squares(cp.multiply(w, u - u_nn)) + self.rho_slack * cp.sum_squares(s))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp(), "QP is not DPP conform"

        return CvxpyLayer(problem, parameters=[u_nn, n_min, n_max, tanphi, A, b], variables=[u, s])

    def _solve(self, layer, u_nn, n_min, n_max, tanphi, A, b, T_nn, n_nn, roll_nn, t_max, roll_max):
        try:
            u_opt, s_opt = layer(
                u_nn, torch.clamp(n_min, min=0.0), n_max, tanphi, A, b,
                solver_args=self.solver_args,
            )
        except Exception as e:
            print(f"[qp] WARN: Solver failed ({e}) -> pass unprocessed inputs as output")
            t_fallback = torch.clamp(T_nn, min=torch.zeros_like(t_max), max=t_max)
            n_fallback = torch.clamp(n_nn, min=n_min, max=n_max)
            roll_fallback = torch.clamp(roll_nn, min=-roll_max, max=roll_max)
            return torch.stack([t_fallback, n_fallback, roll_fallback], dim=-1), None

        thrust = t_max * u_opt[:, 0]
        n = torch.linalg.norm(u_opt[:, 1:3], dim=1)
        roll = torch.rad2deg(torch.atan2(u_opt[:, 2], u_opt[:, 1]))
        return torch.stack([thrust, n, roll], dim=-1), s_opt


class AircraftEnvelopeQPLayer(_QPLayerBase):
    def __init__(self, base_model, y_mean, y_std, stall_margin: float = 1.2, alpha_stall: float = 1.0,
                 w_action=(4.0, 1.0, 1.0), rho_slack: float = 1e4, solver_args: dict | None = None,
                 debug: bool = True):
        super().__init__(base_model, y_mean, y_std, w_action, rho_slack, solver_args, debug)
        self.stall_margin = stall_margin
        self.alpha_stall = alpha_stall
        self.layer = self._build_qp(n_rows=1)

    @staticmethod
    def compute_row(h, v, gamma, mass, C_D0, area, cl_max, k, t_max, n_nn, stall_margin, alpha_stall):
        F = AircraftFormulasTorch
        gam = torch.deg2rad(gamma)
        sin_g, cos_g = torch.sin(gam), torch.cos(gam)

        requested_cl = F.required_lift_coefficient(h, v, n_nn, mass, area)
        actual_cl = torch.minimum(requested_cl, cl_max)
        drag = F.drag_force(h, v, actual_cl, C_D0, k, area)
        v_stall = F.stall_speed(h, mass, cl_max, area)

        zeros = torch.zeros_like(h)
        h1 = v - stall_margin * v_stall
        a_row = torch.stack([t_max / mass, zeros, zeros], dim=1)
        b_row = (
            -drag / mass
            - F.g * sin_g
            - stall_margin * (v_stall / (2.0 * F.H)) * v * sin_g
            + alpha_stall * h1
        )
        return a_row, b_row, drag, v_stall, sin_g, cos_g

    @torch.no_grad()
    def forward(self, x_norm: Tensor, aircraft_state: Tensor, aircraft_dyn: Tensor, aircraft_limits: Tensor) -> Tensor:
        h, v, gamma = aircraft_state.unbind(dim=-1)
        mass, C_D0, area, cl_max, k = aircraft_dyn.unbind(dim=-1)
        t_max, n_min, n_max, roll_max = aircraft_limits.unbind(dim=-1)
        phi_max = torch.deg2rad(roll_max)

        T_nn, n_nn, roll_nn, u_nn = self._decode_action(x_norm, t_max)

        a_row, b_row, drag, v_stall, sin_g, cos_g = self.compute_row(
            h, v, gamma, mass, C_D0, area, cl_max, k, t_max, n_nn, self.stall_margin, self.alpha_stall,
        )
        A, b = a_row.unsqueeze(1), b_row.unsqueeze(1)

        out, s_opt = self._solve(self.layer, u_nn, n_min, n_max, torch.tan(phi_max), A, b,
                                  T_nn, n_nn, roll_nn, t_max, roll_max)
        if self.debug:
            for i in range(out.shape[0]):
                slack = f"{s_opt[i].max():.4f}" if s_opt is not None else "fallback"
                print(f"[env-qp] v={v[i]:6.1f}  {self.stall_margin:.2f}·v_stall={self.stall_margin * v_stall[i]:6.1f}  |  "
                      f"T {T_nn[i]/1e3:6.1f} -> {out[i,0]/1e3:6.1f} kN  |  n {n_nn[i]:5.2f} -> {out[i,1]:5.2f}  |  "
                      f"roll {roll_nn[i]:+6.1f} -> {out[i,2]:+6.1f}°  |  slack {slack}")
        return out


class AircraftFunnelQPLayer(_QPLayerBase):
    def __init__(self, base_model, y_mean, y_std, h_thr: float = 0.0, glideslope_deg: float = 3.0,
                 c_y0: float = 15.0, kappa_y_deg: float = 2.5, c_z0: float = 15.0, kappa_z_deg: float = 1.0,
                 k_funnel: tuple[float, float] = (0.4, 0.4), s_activate: float = 15_000.0, s_min: float = 150.0,
                 heading_gate_deg: float = 90.0, w_action=(4.0, 1.0, 1.0), rho_slack: float = 1e4,
                 solver_args: dict | None = None, debug: bool = True):
        super().__init__(base_model, y_mean, y_std, w_action, rho_slack, solver_args, debug)
        self.h_thr = h_thr
        self.gs = glideslope_deg * DEG2RAD
        self.c_y0 = c_y0
        self.kappa_y = math.tan(kappa_y_deg * DEG2RAD)
        self.c_z0 = c_z0
        self.kappa_z = math.tan(kappa_z_deg * DEG2RAD)
        self.k1_f, self.k2_f = k_funnel
        self.s_activate = s_activate
        self.s_min = s_min
        self.cos_gate = math.cos(heading_gate_deg * DEG2RAD)
        self.layer = self._build_qp(n_rows=2)

    @staticmethod
    def compute_rows(h, v, gamma, mass, C_D0, area, cl_max, k, t_max, n_nn, aircraft_pos, runway_pos,
                      h_thr, gs, c_y0, kappa_y, c_z0, kappa_z, k1_f, k2_f, s_activate, s_min, cos_gate):
        F = AircraftFormulasTorch
        gam = torch.deg2rad(gamma)
        sin_g, cos_g = torch.sin(gam), torch.cos(gam)

        requested_cl = F.required_lift_coefficient(h, v, n_nn, mass, area)
        actual_cl = torch.minimum(requested_cl, cl_max)
        drag = F.drag_force(h, v, actual_cl, C_D0, k, area)

        x_ac, y_ac, chi_deg = aircraft_pos.unbind(dim=-1)
        fix_x, fix_y, chi_r_deg = runway_pos.unbind(dim=-1)
        chi = torch.deg2rad(chi_deg)
        chi_r = torch.deg2rad(chi_r_deg)

        x_rel = x_ac - fix_x
        y_rel = y_ac - fix_y
        sin_r, cos_r = torch.sin(chi_r), torch.cos(chi_r)
        s_ = -(x_rel * sin_r + y_rel * cos_r)
        e_y = x_rel * cos_r - y_rel * sin_r
        e_z = h - h_thr - s_ * math.tan(gs)

        delta = torch.atan2(torch.sin(chi - chi_r), torch.cos(chi - chi_r))
        cdel, sdel = torch.cos(delta), torch.sin(delta)
        tg = math.tan(gs)

        s_dot = -v * cos_g * cdel
        ey_dot = v * cos_g * sdel
        ez_dot = v * sin_g + tg * v * cos_g * cdel

        aTs = -(cos_g * cdel) * t_max / mass
        auvs = F.g * sin_g * cdel
        auhs = F.g * sdel
        bs = cos_g * cdel * drag / mass

        aTz = sin_g * t_max / mass - tg * aTs
        auvz = F.g * cos_g - tg * auvs
        auhz = -tg * auhs
        bz = -sin_g * drag / mass - F.g - tg * bs

        cy = c_y0 + kappa_y * s_
        cz = c_z0 + kappa_z * s_
        hy = cy ** 2 - e_y ** 2
        hz = cz ** 2 - e_z ** 2
        hy_dot = 2.0 * cy * kappa_y * s_dot - 2.0 * e_y * ey_dot
        hz_dot = 2.0 * cz * kappa_z * s_dot - 2.0 * e_z * ez_dot

        aTe = (cos_g * sdel) * t_max / mass
        auve = -F.g * sin_g * sdel
        auhe = F.g * cdel
        be = -cos_g * sdel * drag / mass

        r1a = (2.0 * kappa_y * cy).unsqueeze(1) * torch.stack([aTs, auvs, auhs], dim=1) \
              - (2.0 * e_y).unsqueeze(1) * torch.stack([aTe, auve, auhe], dim=1)
        r1b = (2.0 * kappa_y ** 2 * s_dot ** 2 - 2.0 * ey_dot ** 2
               + 2.0 * kappa_y * cy * bs - 2.0 * e_y * be
               + (k1_f + k2_f) * hy_dot + k1_f * k2_f * hy)

        r2a = (2.0 * kappa_z * cz).unsqueeze(1) * torch.stack([aTs, auvs, auhs], dim=1) \
              - (2.0 * e_z).unsqueeze(1) * torch.stack([aTz, auvz, auhz], dim=1)
        r2b = (2.0 * kappa_z ** 2 * s_dot ** 2 - 2.0 * ez_dot ** 2
               + 2.0 * kappa_z * cz * bs - 2.0 * e_z * bz
               + (k1_f + k2_f) * hz_dot + k1_f * k2_f * hz)

        A = torch.stack([r1a, r2a], dim=1)
        b = torch.stack([r1b, r2b], dim=1)

        heading_ok = (cdel > cos_gate).float()
        active = ((s_ > s_min) & (s_ < s_activate)).float() * heading_ok
        gate = torch.stack([active, active], dim=1)
        A = A * gate.unsqueeze(-1)
        b = torch.where(gate > 0, b, torch.ones_like(b))

        return A, b, s_, e_y, e_z

    @torch.no_grad()
    def forward(self, x_norm: Tensor, aircraft_state: Tensor, aircraft_dyn: Tensor, aircraft_limits: Tensor,
                aircraft_pos: Tensor, runway_pos: Tensor) -> Tensor:
        h, v, gamma = aircraft_state.unbind(dim=-1)
        mass, C_D0, area, cl_max, k = aircraft_dyn.unbind(dim=-1)
        t_max, n_min, n_max, roll_max = aircraft_limits.unbind(dim=-1)
        phi_max = torch.deg2rad(roll_max)

        T_nn, n_nn, roll_nn, u_nn = self._decode_action(x_norm, t_max)

        A, b, s_, e_y, e_z = self.compute_rows(
            h, v, gamma, mass, C_D0, area, cl_max, k, t_max, n_nn, aircraft_pos, runway_pos,
            self.h_thr, self.gs, self.c_y0, self.kappa_y, self.c_z0, self.kappa_z,
            self.k1_f, self.k2_f, self.s_activate, self.s_min, self.cos_gate,
        )

        out, s_opt = self._solve(self.layer, u_nn, n_min, n_max, torch.tan(phi_max), A, b,
                                  T_nn, n_nn, roll_nn, t_max, roll_max)
        if self.debug:
            for i in range(out.shape[0]):
                slack = f"{s_opt[i].max():.4f}" if s_opt is not None else "fallback"
                print(f"[funnel-qp] T {T_nn[i]/1e3:6.1f} -> {out[i,0]/1e3:6.1f} kN  |  n {n_nn[i]:5.2f} -> {out[i,1]:5.2f}  |  "
                      f"roll {roll_nn[i]:+6.1f} -> {out[i,2]:+6.1f}°  |  slack {slack}  |  "
                      f"s={s_[i]:8.1f}  e_y={e_y[i]:+6.1f}  e_z={e_z[i]:+6.1f}")
        return out


class AircraftCombinedQPLayer(_QPLayerBase):
    def __init__(self, base_model, y_mean, y_std,
                 stall_margin: float = 1.2, alpha_stall: float = 1.0,
                 h_thr: float = 0.0, glideslope_deg: float = 3.0, c_y0: float = 15.0, kappa_y_deg: float = 2.5,
                 c_z0: float = 15.0, kappa_z_deg: float = 1.0, k_funnel: tuple[float, float] = (0.4, 0.4),
                 s_activate: float = 15_000.0, s_min: float = 150.0, heading_gate_deg: float = 90.0,
                 w_action=(4.0, 1.0, 1.0), rho_slack: float = 1e4, solver_args: dict | None = None,
                 debug: bool = True):
        super().__init__(base_model, y_mean, y_std, w_action, rho_slack, solver_args, debug)
        self.stall_margin = stall_margin
        self.alpha_stall = alpha_stall
        self.h_thr = h_thr
        self.gs = glideslope_deg * DEG2RAD
        self.c_y0 = c_y0
        self.kappa_y = math.tan(kappa_y_deg * DEG2RAD)
        self.c_z0 = c_z0
        self.kappa_z = math.tan(kappa_z_deg * DEG2RAD)
        self.k1_f, self.k2_f = k_funnel
        self.s_activate = s_activate
        self.s_min = s_min
        self.cos_gate = math.cos(heading_gate_deg * DEG2RAD)
        self.layer = self._build_qp(n_rows=3)

    @torch.no_grad()
    def forward(self, x_norm: Tensor, aircraft_state: Tensor, aircraft_dyn: Tensor, aircraft_limits: Tensor,
                aircraft_pos: Tensor, runway_pos: Tensor) -> Tensor:
        h, v, gamma = aircraft_state.unbind(dim=-1)
        mass, C_D0, area, cl_max, k = aircraft_dyn.unbind(dim=-1)
        t_max, n_min, n_max, roll_max = aircraft_limits.unbind(dim=-1)
        phi_max = torch.deg2rad(roll_max)

        T_nn, n_nn, roll_nn, u_nn = self._decode_action(x_norm, t_max)

        env_a, env_b, drag, v_stall, sin_g, cos_g = AircraftEnvelopeQPLayer.compute_row(
            h, v, gamma, mass, C_D0, area, cl_max, k, t_max, n_nn, self.stall_margin, self.alpha_stall,
        )
        fun_A, fun_b, s_, e_y, e_z = AircraftFunnelQPLayer.compute_rows(
            h, v, gamma, mass, C_D0, area, cl_max, k, t_max, n_nn, aircraft_pos, runway_pos,
            self.h_thr, self.gs, self.c_y0, self.kappa_y, self.c_z0, self.kappa_z,
            self.k1_f, self.k2_f, self.s_activate, self.s_min, self.cos_gate,
        )

        A = torch.cat([env_a.unsqueeze(1), fun_A], dim=1)
        b = torch.cat([env_b.unsqueeze(1), fun_b], dim=1)

        out, s_opt = self._solve(self.layer, u_nn, n_min, n_max, torch.tan(phi_max), A, b,
                                  T_nn, n_nn, roll_nn, t_max, roll_max)
        if self.debug:
            for i in range(out.shape[0]):
                slack = f"{s_opt[i].max():.4f}" if s_opt is not None else "fallback"
                print(f"[combo-qp] v={v[i]:6.1f}  {self.stall_margin:.2f}·v_stall={self.stall_margin*v_stall[i]:6.1f}  |  "
                      f"T {T_nn[i]/1e3:6.1f} -> {out[i,0]/1e3:6.1f} kN  |  n {n_nn[i]:5.2f} -> {out[i,1]:5.2f}  |  "
                      f"roll {roll_nn[i]:+6.1f} -> {out[i,2]:+6.1f}°  |  slack {slack}  |  "
                      f"s={s_[i]:8.1f}  e_y={e_y[i]:+6.1f}  e_z={e_z[i]:+6.1f}")
        return out


if __name__ == "__main__":
    class _FixedActionModel(nn.Module):
        def __init__(self, actions_norm: Tensor):
            super().__init__()
            self.actions_norm = actions_norm

        def forward(self, x: Tensor) -> Tensor:
            return self.actions_norm

    torch.manual_seed(0)
    B = 3
    y_mean = np.array([80_000.0, 1.0, 0.0], dtype=np.float32)
    y_std = np.array([60_000.0, 0.25, 12.0], dtype=np.float32)

    fix_x, fix_y, chi_r_deg = 0.0, 0.0, 100.0

    def make_pos(s, e_y, dchi_deg):
        chi_r = chi_r_deg * DEG2RAD
        x_rel = -s * math.sin(chi_r) + e_y * math.cos(chi_r)
        y_rel = -s * math.cos(chi_r) - e_y * math.sin(chi_r)
        return fix_x + x_rel, fix_y + y_rel, chi_r_deg + dchi_deg

    h_thr = 4.0
    h_gp = h_thr + 5000.0 * math.tan(3.0 * DEG2RAD)   # ~266 m
    aircraft_state = torch.tensor([
        [h_gp + 5.0,  74.0, -3.0],
        [h_gp + 5.0,  74.0, -3.0],
        [h_gp + 60.0, 64.0, -3.0],
    ])
    aircraft_dyn = torch.tensor([65000.0, 0.028, 124.6, 2.4, 0.045]).repeat(B, 1)
    aircraft_limits = torch.tensor([240e3, 0.3, 1.5, 30.0]).repeat(B, 1)

    poss = [make_pos(5000.0, 5.0, 3.0), make_pos(5000.0, 120.0, 0.0), make_pos(5000.0, 5.0, 3.0)]
    aircraft_pos = torch.tensor(poss, dtype=torch.float32)
    runway_pos = torch.tensor([[fix_x, fix_y, chi_r_deg]], dtype=torch.float32).repeat(B, 1)

    actions_phys = torch.tensor([
        [55_000.0, 1.00, 2.0],
        [55_000.0, 1.00, 0.0],
        [30_000.0, 1.00, 0.0],
    ])
    actions_norm = (actions_phys - torch.from_numpy(y_mean)) / torch.from_numpy(y_std)

    combined = AircraftCombinedQPLayer(
        base_model=_FixedActionModel(actions_norm),
        y_mean=y_mean, y_std=y_std,
        h_thr=4.0,
    )
    out_combined = combined(torch.zeros(B, 15), aircraft_state, aircraft_dyn, aircraft_limits,
                             aircraft_pos=aircraft_pos, runway_pos=runway_pos)
    print("\ncombined output [T(N), n, roll(deg)]:\n", out_combined.numpy().round(2))

    env_only = AircraftEnvelopeQPLayer(
        base_model=_FixedActionModel(actions_norm),
        y_mean=y_mean, y_std=y_std,
    )
    out_env = env_only(torch.zeros(B, 15), aircraft_state, aircraft_dyn, aircraft_limits)
    print("\nenvelope-only output [T(N), n, roll(deg)]:\n", out_env.numpy().round(2))

    funnel_only = AircraftFunnelQPLayer(
        base_model=_FixedActionModel(actions_norm),
        y_mean=y_mean, y_std=y_std,
        h_thr=4.0,
    )
    out_fun = funnel_only(torch.zeros(B, 15), aircraft_state, aircraft_dyn, aircraft_limits,
                           aircraft_pos=aircraft_pos, runway_pos=runway_pos)
    print("\nfunnel-only output [T(N), n, roll(deg)]:\n", out_fun.numpy().round(2))