import math

import numpy as np
import torch
from torch import nn, Tensor

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

DEG2RAD = math.pi / 180.0


class AircraftFormulasTorch:
    """
    1:1-mirror of AircraftKinematicsStateCasadi (batched, torch).
    """
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

class AircraftDynamicsQPLayer(nn.Module):
    """
    QP per Sample (Aircraft-Dynamics + optional ILS-Kegel heading Runway).

    Positions are in the airport center UTM-Frame: x (Easting) / y (Northing) / h in meter.
    Heading (chi) und runway_bearing in degree. The Ils kegel works in raw relative
     (x_rel = x_ac - fix_x), the airport offset cuts away.

        min_{u,s}  ‖ W (u − u_nn) ‖²  +  ρ_slack · ‖s‖²
        s.t.       0 ≤ T̃ ≤ 1
                   ‖(uv, uh)‖₂ ≤ n_max
                   uv ≥ n_min
                   |uh| ≤ tan(φ_max) · uv
                   a_st · T̃ + b_st + s₀ ≥ 0          (Stall-CBF)
                   a_y · u  + b_y  + s₁ ≥ 0          (Funnel lateral, HOCBF)
                   a_z · u  + b_z  + s₂ ≥ 0          (Funnel vertical, HOCBF)

    with uv = n·cos(φ), uh = n·sin(φ)  (Ruecktransform: n = ‖(uv,uh)‖, φ = atan2(uh, uv)).

    ILS-Kegel:
        s     = -(x_rel·sin χ_r + y_rel·cos χ_r)       along-track to the threshold
        e_y   =   x_rel·cos χ_r - y_rel·sin χ_r        laterale difference
        e_z   =   h - h_thr - s·tan(γ_gs)              Gleitpfad-difference
        c_y(s), c_z(s) shrink heading Threshold; h_y = c_y²-e_y², h_z = c_z²-e_z² ≥ 0.
    both rel. degree 2 -> HOCBF. Gating: active only for s_min < s < s_activate.
    """

    def __init__(
        self,
        base_model: nn.Module,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        stall_margin: float = 1.2,
        alpha_stall: float = 1.0,
        use_funnel: bool = True,
        h_thr: float = 0.0,
        glideslope_deg: float = 3.0,
        c_y0: float = 15.0,
        kappa_y_deg: float = 2.5,
        c_z0: float = 15.0,
        kappa_z_deg: float = 1.0,
        k_funnel: tuple[float, float] = (0.4, 0.4),
        s_activate: float = 15_000.0,
        s_min: float = 150.0,
        w_action: tuple[float, float, float] = (4.0, 1.0, 1.0),
        rho_slack: float = 1e4,
        solver_args: dict | None = None,
        debug: bool = True,
        heading_gate_deg: float = 90.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))
        self.stall_margin = stall_margin
        self.alpha_stall = alpha_stall
        self.use_funnel = use_funnel
        self.h_thr = h_thr
        self.gs = glideslope_deg * DEG2RAD
        self.c_y0 = c_y0
        self.kappa_y = math.tan(kappa_y_deg * DEG2RAD)
        self.c_z0 = c_z0
        self.kappa_z = math.tan(kappa_z_deg * DEG2RAD)
        self.k1_f, self.k2_f = k_funnel
        self.s_activate = s_activate
        self.s_min = s_min
        self.solver_args = solver_args or {}
        self.debug = debug
        self.cos_gate = math.cos(heading_gate_deg * DEG2RAD)

        n_rows = 3 if use_funnel else 1

        u = cp.Variable(3)                 # [T̃, uv, uh]
        s = cp.Variable(n_rows, nonneg=True)

        u_nn = cp.Parameter(3)
        n_min = cp.Parameter(nonneg=True)
        n_max = cp.Parameter(nonneg=True)
        tanphi = cp.Parameter(nonneg=True)
        A = cp.Parameter((n_rows, 3))
        b = cp.Parameter(n_rows)

        w = np.sqrt(np.asarray(w_action, dtype=float))
        constraints = [
            u[0] >= 0.0,
            u[0] <= 1.0,
            cp.norm(u[1:3], 2) <= n_max,
            u[1] >= n_min,
            u[2] <= tanphi * u[1],
            -u[2] <= tanphi * u[1],
            A @ u + b + s >= 0.0,
        ]
        objective = cp.Minimize(cp.sum_squares(cp.multiply(w, u - u_nn)) + rho_slack * cp.sum_squares(s))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp(), "QP is not DPP conform"

        self.layer = CvxpyLayer(
            problem,
            parameters=[u_nn, n_min, n_max, tanphi, A, b],
            variables=[u, s],
        )

    # todo: add constraint, to be as slow as possible
    @torch.no_grad()  # Inference-Layer; remove before training !!!
    def forward(
        self,
        x_norm: Tensor,                       # (B, input_dim) normalised Model-Input
        aircraft_state: Tensor,               # (B, 3) raw data: h [m], v [m/s], gamma [deg]
        aircraft_dyn: Tensor,                 # (B, 5) raw data: mass, C_D0, area, cl_max, k
        aircraft_limits: Tensor,              # (B, 4) raw data: max_thrust [N], min_lf, max_lf, max_roll [deg]
        aircraft_pos: Tensor | None = None,   # (B, 3) raw data: x, y [m], chi [deg]           -- only when funnel is active
        runway_pos: Tensor | None = None,     # (B, 3) raw data: fix_x, fix_y [m], chi_r [deg] -- only when funnel is active
    ) -> Tensor:                              # (B, 3) physics: thrust [N], n [-], roll [deg]
        F = AircraftFormulasTorch

        raw_norm = self.base_model(x_norm)
        raw_phys = raw_norm * self.y_std + self.y_mean
        T_nn, n_nn, roll_nn = raw_phys.unbind(dim=-1)

        h, v, gamma = aircraft_state.unbind(dim=-1)
        mass, C_D0, area, cl_max, k = aircraft_dyn.unbind(dim=-1)
        t_max, n_min, n_max, roll_max = aircraft_limits.unbind(dim=-1)

        phi_nn = torch.deg2rad(roll_nn)
        gam = torch.deg2rad(gamma)
        phi_max = torch.deg2rad(roll_max)

        u_nn = torch.stack(
            [
                torch.clamp(T_nn / t_max, 0.0, 1.0),
                n_nn * torch.cos(phi_nn),
                n_nn * torch.sin(phi_nn),
            ],
            dim=1,
        )

        requested_cl = F.required_lift_coefficient(h, v, n_nn, mass, area)
        actual_cl = torch.minimum(requested_cl, cl_max)
        drag = F.drag_force(h, v, actual_cl, C_D0, k, area)

        v_stall = F.stall_speed(h, mass, cl_max, area)
        sin_g, cos_g = torch.sin(gam), torch.cos(gam)
        zeros = torch.zeros_like(h)

        h1 = v - self.stall_margin * v_stall
        r0a = torch.stack([t_max / mass, zeros, zeros], dim=1)
        r0b = (
                -drag / mass
                - F.g * sin_g
                - self.stall_margin * (v_stall / (2.0 * F.H)) * v * sin_g
                + self.alpha_stall * h1
        )

        if self.use_funnel:
            assert aircraft_pos is not None and runway_pos is not None, "use_funnel=True needs aircraft_pos and runway_pos"
            x_ac, y_ac, chi_deg = aircraft_pos.unbind(dim=-1)
            fix_x, fix_y, chi_r_deg = runway_pos.unbind(dim=-1)
            chi = torch.deg2rad(chi_deg)
            chi_r = torch.deg2rad(chi_r_deg)

            x_rel = x_ac - fix_x
            y_rel = y_ac - fix_y

            sin_r, cos_r = torch.sin(chi_r), torch.cos(chi_r)
            s_ = -(x_rel * sin_r + y_rel * cos_r)
            e_y = x_rel * cos_r - y_rel * sin_r
            e_z = h - self.h_thr - s_ * math.tan(self.gs)

            delta = torch.atan2(torch.sin(chi - chi_r), torch.cos(chi - chi_r))
            cdel, sdel = torch.cos(delta), torch.sin(delta)
            tg = math.tan(self.gs)

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

            cy = self.c_y0 + self.kappa_y * s_
            cz = self.c_z0 + self.kappa_z * s_
            hy = cy ** 2 - e_y ** 2
            hz = cz ** 2 - e_z ** 2
            hy_dot = 2.0 * cy * self.kappa_y * s_dot - 2.0 * e_y * ey_dot
            hz_dot = 2.0 * cz * self.kappa_z * s_dot - 2.0 * e_z * ez_dot

            aTe = (cos_g * sdel) * t_max / mass
            auve = -F.g * sin_g * sdel
            auhe = F.g * cdel
            be = -cos_g * sdel * drag / mass

            r1a = (2.0 * self.kappa_y * cy).unsqueeze(1) * torch.stack([aTs, auvs, auhs], dim=1) \
                  - (2.0 * e_y).unsqueeze(1) * torch.stack([aTe, auve, auhe], dim=1)
            r1b = (2.0 * self.kappa_y ** 2 * s_dot ** 2 - 2.0 * ey_dot ** 2
                   + 2.0 * self.kappa_y * cy * bs - 2.0 * e_y * be
                   + (self.k1_f + self.k2_f) * hy_dot + self.k1_f * self.k2_f * hy)

            r2a = (2.0 * self.kappa_z * cz).unsqueeze(1) * torch.stack([aTs, auvs, auhs], dim=1) \
                  - (2.0 * e_z).unsqueeze(1) * torch.stack([aTz, auvz, auhz], dim=1)
            r2b = (2.0 * self.kappa_z ** 2 * s_dot ** 2 - 2.0 * ez_dot ** 2
                   + 2.0 * self.kappa_z * cz * bs - 2.0 * e_z * bz
                   + (self.k1_f + self.k2_f) * hz_dot + self.k1_f * self.k2_f * hz)

            A = torch.stack([r0a, r1a, r2a], dim=1)
            b = torch.stack([r0b, r1b, r2b], dim=1)

            heading_ok = (cdel > self.cos_gate).float()
            active = ((s_ > self.s_min) & (s_ < self.s_activate)).float() * heading_ok
            gate = torch.stack([torch.ones_like(active), active, active], dim=1)
            A = A * gate.unsqueeze(-1)
            b = torch.where(gate > 0, b, torch.ones_like(b))

        else:
            A = r0a.unsqueeze(1)
            b = r0b.unsqueeze(1)

        try:
            u_opt, s_opt = self.layer(
                u_nn,
                torch.clamp(n_min, min=0.0),
                n_max,
                torch.tan(phi_max),
                A,
                b,
                solver_args=self.solver_args,
            )
        except Exception as e:
            print(f"[dyn-qp] WARN: Solver failed ({e}) -> pass unprocessed inputs as output")

            t_fallback = torch.clamp(T_nn, min=torch.zeros_like(t_max), max=t_max)
            n_fallback = torch.clamp(n_nn, min=n_min, max=n_max)
            roll_fallback = torch.clamp(roll_nn, min=-roll_max, max=roll_max)
            return torch.stack([t_fallback, n_fallback, roll_fallback], dim=-1)

        thrust = t_max * u_opt[:, 0]
        n = torch.linalg.norm(u_opt[:, 1:3], dim=1)
        roll = torch.rad2deg(torch.atan2(u_opt[:, 2], u_opt[:, 1]))

        if self.debug:
            for i in range(thrust.shape[0]):
                extra = ""
                if self.use_funnel:
                    extra = f"  |  s={s_[i]:8.1f}  e_y={e_y[i]:+6.1f}  e_z={e_z[i]:+6.1f}"
                print(
                    f"[dyn-qp] v={v[i]:6.1f}  {self.stall_margin:.2f}·v_stall={self.stall_margin * v_stall[i]:6.1f}  |  "
                    f"T {T_nn[i] / 1e3:6.1f} -> {thrust[i] / 1e3:6.1f} kN  |  "
                    f"n {n_nn[i]:5.2f} -> {n[i]:5.2f}  |  "
                    f"roll {roll_nn[i]:+6.1f} -> {roll[i]:+6.1f}°  |  slack {s_opt[i].max():.4f}{extra}"
                )

        return torch.stack([thrust, n, roll], dim=-1)


if __name__ == "__main__":
    # test function to see if the gradients flow
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
    #   h             v      gamma
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

    layer = AircraftDynamicsQPLayer(
        base_model=_FixedActionModel(actions_norm),
        y_mean=y_mean, y_std=y_std,
        use_funnel=True,
        h_thr=4.0,
    )
    out = layer(torch.zeros(B, 15), aircraft_state, aircraft_dyn, aircraft_limits,
                aircraft_pos=aircraft_pos, runway_pos=runway_pos)
    print("\noutput [T(N), n, roll(deg)]:\n", out.numpy().round(2))