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
    def __init__(
        self,
        base_model: nn.Module,
        y_mean: float | Tensor,
        y_std: float | Tensor,
        solver: str,
        w_action: tuple[float, float, float] = (1.0, 1.0, 1.0),
        rho_slack: float = 1e4,
        solver_args: dict | None = None,
        debug: bool = True,
        rho_track: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_model: nn.Module = base_model
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))
        self.w_action: tuple[float, float, float] = w_action
        self.rho_slack: float = rho_slack # penalty factor for constraint
        self.rho_track: float = rho_track # weight of the optional soft centerline-tracking cost (see AircraftFunnelQPLayer)
        self.solver_args: dict = solver_args or {}
        self.debug: bool = debug
        self.fail_rate_warn: float = 0.1  # log warning if per-batch fail rate exceeds this
        self.last_solved_mask: Tensor | None = None  # (B,) bool, will be places after every forward() pass
        self.solver: str = solver

    def _decode_action(
        self, x_norm: Tensor, t_max: Tensor, raw_norm: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # raw_norm given -> prediction already computed outside (e.g. by a GNN that needs batch_idx),
        # the base_model is NOT called again
        if raw_norm is None:
            raw_norm = self.base_model(x_norm) # forward pass
        self.last_raw_norm = raw_norm
        raw_phys = raw_norm * self.y_std + self.y_mean
        T_nn, n_nn, roll_nn = raw_phys.unbind(dim=-1)
        phi_nn = torch.deg2rad(roll_nn)
        u_nn = torch.stack([
            torch.clamp(T_nn / t_max, 0.0, 1.0), # normalised thrust in [0, 1]
            n_nn * torch.cos(phi_nn), # vertical load factor component
            n_nn * torch.sin(phi_nn), # horizontal load factor component
        ], dim=1)
        return T_nn, n_nn, roll_nn, u_nn

    @staticmethod
    def _box_constraints(
        u: cp.Variable,
        n_min: cp.Parameter,
        n_max: cp.Parameter,
        tanphi: cp.Parameter,
    ) -> list[cp.Constraint]:
        return [
            u[0] >= 0.0,                    # thrust >= 0
            u[0] <= 1.0,                    # thrust <= 1
            cp.norm(u[1:3], 2) <= n_max, # ‖(uv,uh)‖₂ = n <= n_max -> Reparametrisation trick, (uv,uh) is exactly n cause -> uv²+uh² = n²cos²φ + n²sin²φ = n²
            u[1] >= n_min,                  # vertical load factor component n·cos(φ) >= n_min
            u[2] <= tanphi * u[1],          # horizontal load -> |uh| <= tan(φ_max)·uv cause uh/uv = tan(φ), see definition of (uv, uh)
            -u[2] <= tanphi * u[1],         # horizontal load
        ]

    def _build_qp(self, n_rows: int, n_track_rows: int = 0) -> CvxpyLayer:
        u = cp.Variable(3)                   # [T̃, uv, uh]
        s = cp.Variable(n_rows, nonneg=True) # slack per cbf row

        u_nn = cp.Parameter(3)
        n_min = cp.Parameter(nonneg=True)
        n_max = cp.Parameter(nonneg=True)
        tanphi = cp.Parameter(nonneg=True)
        A = cp.Parameter((n_rows, 3))
        b = cp.Parameter(n_rows)

        w = np.sqrt(np.asarray(self.w_action, dtype=float))
        constraints = self._box_constraints(u, n_min, n_max, tanphi)
        constraints.append(A @ u + b + s >= 0.0)

        # goal / objective function to minimize
        objective_expr = cp.sum_squares(cp.multiply(w, u - u_nn)) + self.rho_slack * cp.sum_squares(s)
        parameters = [u_nn, n_min, n_max, tanphi, A, b]

        if n_track_rows > 0:
            A_track = cp.Parameter((n_track_rows, 3))
            b_track = cp.Parameter(n_track_rows)
            objective_expr = objective_expr + self.rho_track * cp.sum_squares(A_track @ u + b_track)
            parameters += [A_track, b_track]

        objective = cp.Minimize(objective_expr)
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp(), "QP is not DPP conform"

        return CvxpyLayer(problem, parameters=parameters, variables=[u, s], solver=self.solver) #  ,solver=cp.MOREAU

    def _solve(
            self,
            layer: CvxpyLayer,
            u_nn: Tensor,
            n_min: Tensor,
            n_max: Tensor,
            tanphi: Tensor,
            A: Tensor,
            b: Tensor,
            T_nn: Tensor,
            n_nn: Tensor,
            roll_nn: Tensor,
            t_max: Tensor,
            roll_max: Tensor,
            A_track: Tensor | None = None,
            b_track: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        B = u_nn.shape[0]
        out_device = u_nn.device
        n_min_c = torch.clamp(n_min, min=0.0)

        layer_args_full: list[Tensor] = [u_nn, n_min_c, n_max, tanphi, A, b]
        if A_track is not None and b_track is not None:
            layer_args_full += [A_track, b_track]

        layer_args_cpu = [t.cpu() for t in layer_args_full] if out_device.type != "cpu" else layer_args_full

        finite_per_sample = torch.stack(
            [torch.isfinite(t).flatten(1).all(dim=1) if t.dim() > 1 else torch.isfinite(t) for t in layer_args_cpu],
            dim=0,
        ).all(dim=0)  # (B,) bool

        u_out = torch.zeros(B, 3, dtype=u_nn.dtype, device=out_device)
        n_rows = A.shape[1]
        s_out = torch.zeros(B, n_rows, dtype=u_nn.dtype, device=out_device)
        solved_mask = torch.zeros(B, dtype=torch.bool, device=out_device)

        finite_idx = finite_per_sample.nonzero(as_tuple=True)[0]
        if finite_idx.numel() > 0:
            sub_args = [t[finite_idx] for t in layer_args_cpu]
            try:
                u_sub, s_sub = layer(*sub_args, solver_args=self.solver_args)
                u_sub = u_sub.to(out_device, dtype=u_nn.dtype)
                s_sub = s_sub.to(out_device, dtype=u_nn.dtype)
                u_out[finite_idx] = self._unpack(u_sub, t_max[finite_idx])
                s_out[finite_idx] = s_sub
                solved_mask[finite_idx] = True
            except Exception as e:
                print(f"[qp] WARN: batched solve over {finite_idx.numel()} finite samples failed ({e}) " f"-> falling back to box-clamp for this subset")

        unsolved_idx = (~solved_mask).nonzero(as_tuple=True)[0]
        if unsolved_idx.numel() > 0:
            u_nn_u, n_min_u, n_max_u, tanphi_u = u_nn[unsolved_idx], n_min_c[unsolved_idx], n_max[unsolved_idx], tanphi[
                unsolved_idx]
            A_u, b_u, t_max_u = A[unsolved_idx], b[unsolved_idx], t_max[unsolved_idx]

            uv = torch.clamp(u_nn_u[:, 1], min=n_min_u, max=n_max_u)
            uh = torch.clamp(u_nn_u[:, 2], min=-tanphi_u * uv, max=tanphi_u * uv)
            u_fb = torch.stack([torch.clamp(u_nn_u[:, 0], 0.0, 1.0), uv, uh], dim=-1)
            Au = torch.einsum("bij,bj->bi", A_u, u_fb)
            s_fb = torch.clamp(-(Au + b_u), min=0.0)

            u_out[unsolved_idx] = self._unpack(u_fb, t_max_u)
            s_out[unsolved_idx] = s_fb

        self.last_solved_mask = solved_mask
        fail_rate = 1.0 - solved_mask.float().mean().item()
        if fail_rate > self.fail_rate_warn:
            print(f"[qp] WARN: high fail rate {fail_rate:.1%} in this batch ({unsolved_idx.numel()}/{B} samples fell back to box-clamp)")

        return u_out, s_out

    @staticmethod
    def _unpack(u_opt: Tensor, t_max: Tensor) -> Tensor:
        thrust = t_max * torch.clamp(u_opt[:, 0], 0.0, 1.0)
        n = torch.linalg.norm(u_opt[:, 1:3], dim=1)
        roll = torch.rad2deg(torch.atan2(u_opt[:, 2], u_opt[:, 1]))
        return torch.stack([thrust, n, roll], dim=-1)

    @staticmethod
    def _analytic_fallback(
        u_nn_i: Tensor,
        n_min_i: Tensor,
        n_max_i: Tensor,
        tanphi_i: Tensor,
        A_i: Tensor,
        b_i: Tensor,
    ) -> tuple[Tensor, Tensor]:
        t_hat = torch.clamp(u_nn_i[0], 0.0, 1.0)
        uv = torch.clamp(u_nn_i[1], min=n_min_i, max=n_max_i)
        uh = torch.clamp(u_nn_i[2], min=-tanphi_i * uv, max=tanphi_i * uv)
        u_fb = torch.stack([t_hat, uv, uh])
        s_fb = torch.clamp(-(A_i @ u_fb + b_i), min=0.0)
        return u_fb, s_fb


class AircraftEnvelopeQPLayer(_QPLayerBase):
    def __init__(
        self,
        base_model: nn.Module,
        y_mean: float | Tensor,
        y_std: float | Tensor,
        solver: str,
        stall_margin: float = 1.2,
        alpha_stall: float = 1.0,
        w_action: tuple[float, float, float] = (1.0, 1.0, 1.0),
        rho_slack: float = 1e4,
        solver_args: dict | None = None,
        debug: bool = True,
    ) -> None:
        super().__init__(
            base_model=base_model,
            y_mean=y_mean,
            y_std=y_std,
            w_action=w_action,
            rho_slack=rho_slack,
            solver_args=solver_args,
            debug=debug,
            solver=solver,
        )
        self.stall_margin: float = stall_margin
        self.alpha_stall: float = alpha_stall
        self.layer: CvxpyLayer = self._build_qp(n_rows=1)

    @staticmethod
    def compute_row(
        h: Tensor,
        v: Tensor,
        gamma: Tensor,
        mass: Tensor,
        C_D0: Tensor,
        area: Tensor,
        cl_max: Tensor,
        k: Tensor,
        t_max: Tensor,
        n_nn: Tensor,
        stall_margin: float,
        alpha_stall: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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

    def forward(
        self,
        x_norm: Tensor,
        aircraft_state: Tensor,
        aircraft_dyn: Tensor,
        aircraft_limits: Tensor
    ) -> Tensor:
        h, v, gamma = aircraft_state.unbind(dim=-1)
        mass, C_D0, area, cl_max, k = aircraft_dyn.unbind(dim=-1)
        t_max, n_min, n_max, roll_max = aircraft_limits.unbind(dim=-1)
        phi_max = torch.deg2rad(roll_max)

        T_nn, n_nn, roll_nn, u_nn = self._decode_action(x_norm, t_max)

        a_row, b_row, drag, v_stall, sin_g, cos_g = self.compute_row(
            h, v, gamma, mass, C_D0, area, cl_max, k, t_max, n_nn, self.stall_margin, self.alpha_stall,
        )
        A, b = a_row.unsqueeze(1), b_row.unsqueeze(1)

        out, s_opt = self._solve(self.layer, u_nn, n_min, n_max, torch.tan(phi_max), A, b, T_nn, n_nn, roll_nn, t_max, roll_max)
        if self.debug:
            for i in range(out.shape[0]):
                slack = f"{s_opt[i].max():.4f}" if s_opt is not None else "fallback"
                print(f"[env-qp] v={v[i]:6.1f}  {self.stall_margin:.2f}·v_stall={self.stall_margin * v_stall[i]:6.1f}  |  "
                      f"T {T_nn[i]/1e3:6.1f} -> {out[i,0]/1e3:6.1f} kN  |  n {n_nn[i]:5.2f} -> {out[i,1]:5.2f}  |  "
                      f"roll {roll_nn[i]:+6.1f} -> {out[i,2]:+6.1f}°  |  slack {slack}")
        return out


class AircraftFunnelQPLayer(_QPLayerBase):
    def __init__(
        self,
        base_model: nn.Module,
        y_mean: float | Tensor,
        y_std: float | Tensor,
        solver: str,
        h_thr: float = 0.0,
        glideslope_deg: float = 3.0,
        thr_half_width_lat_m: float = 106.68,   # 350 ft floor, Order 8260.58D Formula 3-1-1
        fsd_v_ratio: float = 0.25,              # RTCA DO-229: vertical FSD = 0.25 * GPA
        fsd_v_min_m: float = 45.0,              # DO-229 vertical FSD clamp, lower bound
        fsd_v_max_m: float = 150.0,             # DO-229 vertical FSD clamp, upper bound
        runway_length_m: float | None = None,   # else 9023 ft FPAP floor, para. 3-1-5.c(1)
        lgsa_deg: float | None = None,          # override; else derived from Formula 3-1-1
        k_funnel: tuple[float, float] = (0.4, 0.4),
        k_track: tuple[float, float] = (0.05, 0.4),
        intercept_angle_deg: float = 30.0,      # max. intercept angle onto the centerline (localizer-style)
        k_vf: float = 1e-3,                     # 1/m, how fast the commanded intercept angle shrinks near the centerline
        k_heading: float = 0.3,                 # 1/s, heading-tracking gain of the lateral tracking cost
        h_floor: float = 2000.0,
        s_activate_funnel: float = 10_000.0,    # distance [m] below which the HOCBF funnel rows are active
        s_activate_track: float = 20_000.0,     # distance [m] below which the soft tracking cost (30 deg intercept) is active
        s_min: float = 150.0,
        heading_gate_deg: float = 90.0,
        w_action: tuple[float, float, float] = (1.0, 1.0, 1.0),
        rho_slack: float = 1e4,
        rho_track: float = 1.0,
        solver_args: dict | None = None,
        debug: bool = True,
    ) -> None:
        super().__init__(
            base_model=base_model,
            y_mean=y_mean,
            y_std=y_std,
            w_action=w_action,
            rho_slack=rho_slack,
            solver_args=solver_args,
            debug=debug,
            rho_track=rho_track,
            solver=solver,
        )
        self.h_thr: float = h_thr                                   # height of the runway fix (LTP)
        self.gs: float = glideslope_deg * DEG2RAD                   # goal glide path angle (GPA) in radiant

        # --- RNAV (GPS) LPV funnel geometry, mirrors LpvFunnelGeometry ---
        # para. 3-1-5.c(1) + c(3)(a): FPAP = max(runway length, 9023 ft), GARP = FPAP + 1000 ft
        fpap_distance_ft = max((runway_length_m or 0.0) / 0.3048, 9023.0)
        self.d_garp: float = (fpap_distance_ft + 1000.0) * 0.3048   # GARP distance behind the LTP, in meters

        if lgsa_deg is not None:
            lgsa = lgsa_deg * DEG2RAD
        else:
            # Formula 3-1-1: course width at threshold -> effective LGSA (1.5 deg .. 2.0 deg)
            course_width_thr = max(thr_half_width_lat_m, math.tan(1.5 * DEG2RAD) * self.d_garp)
            lgsa = math.atan(course_width_thr / self.d_garp)
        self.tan_lgsa: float = math.tan(lgsa)                       # d(w_lat)/ds -> const., w_lat(s) is linear in s

        self.fsd_v_min: float = fsd_v_min_m                         # DO-229 vertical FSD clamp, lower bound
        self.fsd_v_max: float = fsd_v_max_m                         # DO-229 vertical FSD clamp, upper bound
        self.tan_fsdv: float = math.tan(fsd_v_ratio * self.gs)      # slope of the (unclamped) vertical FSD ramp

        self.k1_f, self.k2_f = k_funnel                             # k1 and k2 for the HOCBF "Verstärkungsfaktoren" for each of the both derivatives
        self.kp_track, self.kd_track = k_track                      # gains for the soft vertical tracking cost (not a barrier, see compute_rows)
        self.intercept_angle_deg: float = intercept_angle_deg       # lateral tracking: max. intercept angle onto the centerline
        self.k_vf: float = k_vf                                     # lateral tracking: vector-field gain (1/m)
        self.k_heading: float = k_heading                           # lateral tracking: heading gain (1/s)
        self.h_floor: float = h_floor                               # clamp on hy/hz fed into b, see param docstring above
        self.s_activate_funnel: float = s_activate_funnel           # when the funnel constraint (HOCBF rows) is activated, in meters
        self.s_activate_track: float = s_activate_track             # when the soft tracking cost (30 deg intercept + glide path PD) is activated, in meters
        self.s_min: float = s_min                                   # when the funnel constraint is deactivated again, due to fragile solving
        self.cos_gate: float = math.cos(heading_gate_deg * DEG2RAD) # heading gate, this means the funnel constraint is activated depending on the heading of the aircraft
        self.layer: CvxpyLayer = self._build_qp(n_rows=2, n_track_rows=2)

    @staticmethod
    def lpv_lateral_geometry(
        runway_length_m: Tensor,
        thr_half_width_lat_m: float,
        lgsa_deg: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        length_ft = torch.nan_to_num(runway_length_m / 0.3048, nan=0.0, posinf=0.0, neginf=0.0)
        fpap_ft = torch.clamp(length_ft, min=9023.0)
        d_garp = (fpap_ft + 1000.0) * 0.3048

        if lgsa_deg is not None:
            tan_lgsa = torch.full_like(d_garp, math.tan(lgsa_deg * DEG2RAD))
        else:
            course_width_thr = torch.clamp(math.tan(1.5 * DEG2RAD) * d_garp, min=thr_half_width_lat_m)
            tan_lgsa = course_width_thr / d_garp  # = tan(atan(w_thr / d_garp))
        return d_garp, tan_lgsa

    @staticmethod
    def fix_distance(fix_pos: Tensor, runway_pos: Tensor) -> Tensor:
        x, y = fix_pos.unbind(dim=-1)
        fix_x, fix_y, chi_r_deg = runway_pos.unbind(dim=-1)
        chi_r = torch.deg2rad(chi_r_deg)
        return -((x - fix_x) * torch.sin(chi_r) + (y - fix_y) * torch.cos(chi_r))

    @staticmethod
    def compute_rows(
        h: Tensor,
        v: Tensor,
        gamma: Tensor,
        mass: Tensor,
        C_D0: Tensor,
        area: Tensor,
        cl_max: Tensor,
        k: Tensor,
        t_max: Tensor,
        n_nn: Tensor,
        aircraft_pos: Tensor,
        runway_pos: Tensor,
        h_thr: float | Tensor,              # scalar or per-sample (B,), runway fix height incl. TCH
        gs: float,
        tan_lgsa: float | Tensor,           # scalar or per-sample (B,), e.g. from the runway length
        d_garp: float | Tensor,             # scalar or per-sample (B,), e.g. from the runway length
        tan_fsdv: float,
        fsd_v_min: float,
        fsd_v_max: float,
        k1_f: float,
        k2_f: float,
        h_floor: float,
        s_activate_funnel: float | Tensor,  # scalar or per-sample (B,), e.g. the FAF distance
        s_activate_track: float | Tensor,   # scalar or per-sample (B,), e.g. the IF distance
        s_min: float,
        cos_gate: float,
        kp_track: float,
        kd_track: float,
        intercept_angle_deg: float = 30.0,  # max. intercept angle onto the centerline (localizer-style)
        k_vf: float = 1e-3,                 # 1/m, how fast the commanded intercept angle shrinks near the centerline
        k_heading: float = 0.3,             # 1/s, heading-tracking gain
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        # distance to the runway fix along the landing axis
        s_ = -(x_rel * sin_r + y_rel * cos_r)
        # lateral error of the aircraft
        e_y = x_rel * cos_r - y_rel * sin_r
        # vertical error of the aircraft, relative to the glide path
        e_z = h - h_thr - s_ * math.tan(gs)

        # heading error of the aircraft to the landing runway axis -> runway heading wrapped [-180, 180] over arctan2
        delta = torch.atan2(torch.sin(chi - chi_r), torch.cos(chi - chi_r))
        cdel, sdel = torch.cos(delta), torch.sin(delta)
        tg = math.tan(gs)

        # kinematic part -> e.g. see the velocity vector of the AircraftKinematicsState class
        s_dot = -v * cos_g * cdel  # velocity
        ey_dot = v * cos_g * sdel
        ez_dot = v * sin_g + tg * v * cos_g * cdel

        # dynamic part with affine coefficients e.g. see v̇, γ̇, χ̇ dof equations
        # note: acceleration along the approach axis -> how strong does thrust act on the acceleration along the approach axis
        aTs = -(cos_g * cdel) * t_max / mass # aTs how strong affects thrust on the acceleration along the approach axis
        auvs = F.g * sin_g * cdel            # auvs, auhs how strong affects thrust ont the vertical and lateral axis
        auhs = F.g * sdel
        bs = cos_g * cdel * drag / mass

        # note: acceleration vertical to the glide path -> height error
        aTz = sin_g * t_max / mass - tg * aTs
        auvz = F.g * cos_g - tg * auvs
        auhz = -tg * auhs
        bz = -sin_g * drag / mass - F.g - tg * bs

        # note: lateral error along the approach axis
        aTe = (cos_g * sdel) * t_max / mass
        auve = -F.g * sin_g * sdel
        auhe = F.g * cdel
        be = -cos_g * sdel * drag / mass

        cy = tan_lgsa * (s_ + d_garp) # [-cy,+cy] -> allowed lateral corridor -> e.g. funnel

        raw_wz = tan_fsdv * s_
        ramp_active = (raw_wz >= fsd_v_min) & (raw_wz <= fsd_v_max)
        cz = torch.clamp(raw_wz, min=fsd_v_min, max=fsd_v_max)
        dcz_ds = torch.where(ramp_active, torch.full_like(s_, tan_fsdv), torch.zeros_like(s_))

        hy = cy ** 2 - e_y ** 2 # h_y > 0, am i inside of the lateral funnel, cause with cy grows s_
        hz = cz ** 2 - e_z ** 2
        hy_c = torch.clamp(hy, min=-h_floor)
        hz_c = torch.clamp(hz, min=-h_floor)
        # first order derivatives of the barrier functions see 4 lines above for reference
        hy_dot = 2.0 * cy * tan_lgsa * s_dot - 2.0 * e_y * ey_dot
        hz_dot = 2.0 * cz * dcz_ds * s_dot - 2.0 * e_z * ez_dot

        # HOCBF conditions, e.g. second order derivatives of the barrier functions
        r1a = (2.0 * tan_lgsa * cy).unsqueeze(1) * torch.stack([aTs, auvs, auhs], dim=1) \
              - (2.0 * e_y).unsqueeze(1) * torch.stack([aTe, auve, auhe], dim=1)
        r1b = (2.0 * tan_lgsa ** 2 * s_dot ** 2 - 2.0 * ey_dot ** 2
               + 2.0 * tan_lgsa * cy * bs - 2.0 * e_y * be
               + (k1_f + k2_f) * hy_dot + k1_f * k2_f * hy_c) # lateral barrier

        r2a = (2.0 * dcz_ds * cz).unsqueeze(1) * torch.stack([aTs, auvs, auhs], dim=1) \
              - (2.0 * e_z).unsqueeze(1) * torch.stack([aTz, auvz, auhz], dim=1)
        r2b = (2.0 * dcz_ds ** 2 * s_dot ** 2 - 2.0 * ez_dot ** 2
               + 2.0 * dcz_ds * cz * bs - 2.0 * e_z * bz
               + (k1_f + k2_f) * hz_dot + k1_f * k2_f * hz_c) # vertical barrier

        # (B, 2, 3) shape for A and B
        A = torch.stack([r1a, r2a], dim=1)
        b = torch.stack([r1b, r2b], dim=1)

        heading_ok = (cdel > cos_gate).float()
        active = ((s_ > s_min) & (s_ < s_activate_funnel)).float() * heading_ok
        gate = torch.stack([active, active], dim=1)
        A = A * gate.unsqueeze(-1)
        b = torch.where(gate > 0, b, torch.ones_like(b))

        chi_int = intercept_angle_deg * DEG2RAD
        delta_des = -chi_int * (2.0 / math.pi) * torch.atan(k_vf * e_y)
        delta_err = torch.atan2(torch.sin(delta - delta_des), torch.cos(delta - delta_des))
        zeros_v = torch.zeros_like(v)
        a_track_y = torch.stack([zeros_v, zeros_v, torch.full_like(v, F.g)], dim=1)
        b_track_y = v * cos_g * k_heading * delta_err

        # vertical: PD on (e_z, ez_dot) back onto the glide path
        a_track_z = torch.stack([aTz, auvz, auhz], dim=1)
        b_track_z = bz + kd_track * ez_dot + kp_track * e_z

        track_active = (s_ < s_activate_track).float() * heading_ok
        gate_t = torch.stack([track_active, track_active], dim=1)
        A_track = torch.stack([a_track_y, a_track_z], dim=1) * gate_t.unsqueeze(-1)
        b_track = torch.stack([b_track_y, b_track_z], dim=1) * gate_t

        return A, b, A_track, b_track, s_, e_y, e_z

    def forward(
        self,
        x_norm: Tensor,
        aircraft_state: Tensor,
        aircraft_dyn: Tensor,
        aircraft_limits: Tensor,
        aircraft_pos: Tensor,
        runway_pos: Tensor
    ) -> Tensor:
        h, v, gamma = aircraft_state.unbind(dim=-1)
        mass, C_D0, area, cl_max, k = aircraft_dyn.unbind(dim=-1)
        t_max, n_min, n_max, roll_max = aircraft_limits.unbind(dim=-1)
        phi_max = torch.deg2rad(roll_max)

        T_nn, n_nn, roll_nn, u_nn = self._decode_action(x_norm, t_max)

        A, b, A_track, b_track, s_, e_y, e_z = self.compute_rows(
            h, v, gamma, mass, C_D0, area, cl_max, k, t_max, n_nn, aircraft_pos, runway_pos,
            self.h_thr, self.gs, self.tan_lgsa, self.d_garp, self.tan_fsdv, self.fsd_v_min, self.fsd_v_max,
            self.k1_f, self.k2_f, self.h_floor, self.s_activate_funnel, self.s_activate_track,
            self.s_min, self.cos_gate, self.kp_track, self.kd_track,
            self.intercept_angle_deg, self.k_vf, self.k_heading,
        )

        out, s_opt = self._solve(self.layer, u_nn, n_min, n_max, torch.tan(phi_max), A, b, T_nn, n_nn, roll_nn, t_max, roll_max, A_track, b_track)
        if self.debug:
            for i in range(out.shape[0]):
                slack = f"{s_opt[i].max():.4f}" if s_opt is not None else "fallback"
                print(f"[funnel-qp] T {T_nn[i]/1e3:6.1f} -> {out[i,0]/1e3:6.1f} kN  |  n {n_nn[i]:5.2f} -> {out[i,1]:5.2f}  |  "
                      f"roll {roll_nn[i]:+6.1f} -> {out[i,2]:+6.1f}°  |  slack {slack}  |  "
                      f"s={s_[i]:8.1f}  e_y={e_y[i]:+6.1f}  e_z={e_z[i]:+6.1f}")
        return out


class AircraftCombinedQPLayer(_QPLayerBase):
    def __init__(
        self,
        base_model: nn.Module,
        y_mean: float | Tensor,
        y_std: float | Tensor,
        solver: str,
        stall_margin: float = 1.2,
        alpha_stall: float = 1.0,
        h_thr: float = 0.0,
        glideslope_deg: float = 3.0,
        thr_half_width_lat_m: float = 106.68,
        fsd_v_ratio: float = 0.25,
        fsd_v_min_m: float = 45.0,
        fsd_v_max_m: float = 150.0,
        runway_length_m: float | None = None, # fallback runway length, only used if no runway_length_m is passed to forward()
        lgsa_deg: float | None = None,
        k_funnel: tuple[float, float] = (0.4, 0.4),
        k_track: tuple[float, float] = (0.05, 0.4),
        intercept_angle_deg: float = 30.0,
        k_vf: float = 1e-3,
        k_heading: float = 0.3,
        h_floor: float = 2000.0,
        s_activate_funnel: float = 10_000.0, # fallback funnel activation distance, only used if no faf_pos is passed to forward()
        s_activate_track: float = 20_000.0,  # fallback tracking activation distance, only used if no if_pos is passed to forward()
        s_min: float = 150.0,
        heading_gate_deg: float = 90.0,
        w_action: tuple[float, float, float] = (1.0, 1.0, 1.0),
        rho_slack: float = 1e4,
        rho_track: float = 1.0,
        solver_args: dict | None = None,
        debug: bool = True,
    ) -> None:
        super().__init__(
            base_model=base_model,
            y_mean=y_mean,
            y_std=y_std,
            w_action=w_action,
            rho_slack=rho_slack,
            solver_args=solver_args,
            debug=debug,
            rho_track=rho_track,
            solver=solver
        )
        self.stall_margin = stall_margin
        self.alpha_stall = alpha_stall
        self.h_thr = h_thr  # 0.0 cause the data is centered around 0.0 of the airport
        self.gs = glideslope_deg * DEG2RAD
        self.thr_half_width_lat_m = thr_half_width_lat_m  # needed for the per-sample geometry in forward()
        self.lgsa_deg = lgsa_deg                          # needed for the per-sample geometry in forward()

        # --- RNAV (GPS) LPV funnel geometry, mirrors LpvFunnelGeometry ---
        # fallback values, used if no per-sample runway_length_m is passed to forward()
        fpap_distance_ft = max((runway_length_m or 0.0) / 0.3048, 9023.0)
        self.d_garp = (fpap_distance_ft + 1000.0) * 0.3048

        if lgsa_deg is not None:
            lgsa = lgsa_deg * DEG2RAD
        else:
            course_width_thr = max(thr_half_width_lat_m, math.tan(1.5 * DEG2RAD) * self.d_garp)
            lgsa = math.atan(course_width_thr / self.d_garp)
        self.tan_lgsa = math.tan(lgsa)

        self.fsd_v_min = fsd_v_min_m
        self.fsd_v_max = fsd_v_max_m
        self.tan_fsdv = math.tan(fsd_v_ratio * self.gs)

        self.k1_f, self.k2_f = k_funnel
        self.kp_track, self.kd_track = k_track
        self.intercept_angle_deg = intercept_angle_deg
        self.k_vf = k_vf
        self.k_heading = k_heading
        self.h_floor = h_floor
        self.s_activate_funnel = s_activate_funnel
        self.s_activate_track = s_activate_track
        self.s_min = s_min
        self.cos_gate = math.cos(heading_gate_deg * DEG2RAD)
        self.layer = self._build_qp(n_rows=3, n_track_rows=2)
        # rows of the last forward (envelope + 2 funnel rows), (B, 3, 3) / (B, 3) -> reused by the
        # multi-aircraft separation QP so it keeps the stall/funnel constraints of every aircraft
        self.last_A: Tensor | None = None
        self.last_b: Tensor | None = None

    def forward(
            self,
            x_norm: Tensor,
            aircraft_state: Tensor,
            aircraft_dyn: Tensor,
            aircraft_limits: Tensor,
            aircraft_pos: Tensor,
            runway_pos: Tensor,
            faf_pos: Tensor | None = None,  # (B, 2) faf x/y [m], same frame as runway_pos; None -> s_activate_funnel
            if_pos: Tensor | None = None,   # (B, 2) IF  x/y [m], same frame as runway_pos; None -> s_activate_track
            runway_length_m: Tensor | None = None,  # (B,) runway length [m], NaN -> 9023 ft floor; None -> constructor value
            raw_norm: Tensor | None = None,         # (B, 3) precomputed normalized prediction; None -> base_model(x_norm)
            h_thr: Tensor | None = None,            # (B,) runway fix height incl. TCH [m] per sample, same frame as aircraft_state h; None -> constructor value
    ) -> Tensor:
        h, v, gamma = aircraft_state.unbind(dim=-1)
        mass, C_D0, area, cl_max, k = aircraft_dyn.unbind(dim=-1)
        t_max, n_min, n_max, roll_max = aircraft_limits.unbind(dim=-1)
        phi_max = torch.deg2rad(roll_max)

        T_nn, n_nn, roll_nn, u_nn = self._decode_action(x_norm, t_max, raw_norm=raw_norm)

        # funnel activation distance: FAF -> runway fix distance [m] per sample if given, else the fixed default
        if faf_pos is not None:
            s_act_funnel: float | Tensor = AircraftFunnelQPLayer.fix_distance(faf_pos, runway_pos)
        else:
            s_act_funnel = self.s_activate_funnel

        # tracking (30 deg intercept) activation distance: IF -> runway fix distance [m] per sample if given
        if if_pos is not None:
            s_act_track: float | Tensor = AircraftFunnelQPLayer.fix_distance(if_pos, runway_pos)
        else:
            s_act_track = self.s_activate_track

        # lateral LPV geometry (GARP distance + LGSA): per sample from the runway length if given
        if runway_length_m is not None:
            d_garp, tan_lgsa = AircraftFunnelQPLayer.lpv_lateral_geometry(
                runway_length_m, self.thr_half_width_lat_m, self.lgsa_deg,
            )
        else:
            d_garp, tan_lgsa = self.d_garp, self.tan_lgsa

        env_a, env_b, drag, v_stall, sin_g, cos_g = AircraftEnvelopeQPLayer.compute_row(
            h, v, gamma, mass, C_D0, area, cl_max, k, t_max, n_nn, self.stall_margin, self.alpha_stall,
        )
        fun_A, fun_b, fun_A_track, fun_b_track, s_, e_y, e_z = AircraftFunnelQPLayer.compute_rows(
            h, v, gamma, mass, C_D0, area, cl_max, k, t_max, n_nn, aircraft_pos, runway_pos,
            self.h_thr if h_thr is None else h_thr, self.gs, tan_lgsa, d_garp, self.tan_fsdv, self.fsd_v_min, self.fsd_v_max,
            self.k1_f, self.k2_f, self.h_floor, s_act_funnel, s_act_track,
            self.s_min, self.cos_gate, self.kp_track, self.kd_track,
            self.intercept_angle_deg, self.k_vf, self.k_heading,
        )

        A = torch.cat([env_a.unsqueeze(1), fun_A], dim=1)
        b = torch.cat([env_b.unsqueeze(1), fun_b], dim=1)
        self.last_A, self.last_b = A, b

        out, s_opt = self._solve(self.layer, u_nn, n_min, n_max, torch.tan(phi_max), A, b, T_nn, n_nn, roll_nn, t_max, roll_max, fun_A_track, fun_b_track)
        if self.debug:
            for i in range(out.shape[0]):
                slack = f"{s_opt[i].max():.4f}" if s_opt is not None else "fallback"
                s_act_f_i = s_act_funnel[i] if isinstance(s_act_funnel, Tensor) else s_act_funnel
                s_act_t_i = s_act_track[i] if isinstance(s_act_track, Tensor) else s_act_track
                d_garp_i = d_garp[i] if isinstance(d_garp, Tensor) else d_garp
                print(f"[combo-qp] v={v[i]:6.1f}  {self.stall_margin:.2f}·v_stall={self.stall_margin*v_stall[i]:6.1f}  |  "
                      f"T {T_nn[i]/1e3:6.1f} -> {out[i,0]/1e3:6.1f} kN  |  n {n_nn[i]:5.2f} -> {out[i,1]:5.2f}  |  "
                      f"roll {roll_nn[i]:+6.1f} -> {out[i,2]:+6.1f}°  |  slack {slack}  |  "
                      f"s={s_[i]:8.1f}  e_y={e_y[i]:+6.1f}  e_z={e_z[i]:+6.1f}  "
                      f"s_act_funnel={s_act_f_i:8.1f}  s_act_track={s_act_t_i:8.1f}  d_garp={d_garp_i:7.1f}")
        return out


def check_gradient_flow(name: str, layer: nn.Module, model: nn.Module, y_mean, y_std, *forward_args) -> bool:
    print(f"\n{'=' * 70}\n[{name}] Gradient-Check\n{'=' * 70}")

    for p in model.parameters():
        p.grad = None

    out = layer(*forward_args)

    ok = True
    if not out.requires_grad:
        print(f"[{name}] FAIL: Output.requires_grad == False -> kein Graph aufgebaut")
        return False
    print(f"[{name}] OK: Output.requires_grad = True, grad_fn = {out.grad_fn}")

    y_mean_t = torch.as_tensor(y_mean, dtype=out.dtype)
    y_std_t = torch.as_tensor(y_std, dtype=out.dtype)
    out_norm = (out - y_mean_t) / y_std_t

    loss = out_norm.pow(2).mean()
    loss.backward()

    for pname, p in model.named_parameters():
        if p.grad is None:
            print(f"[{name}] FAIL: kein Gradient für Parameter '{pname}'")
            ok = False
            continue
        finite = torch.isfinite(p.grad).all().item()
        nonzero = (p.grad.abs().sum() > 0).item()
        print(f"[{name}] Parameter '{pname}': grad_norm={p.grad.norm():.6f}  "
              f"finite={finite}  nonzero={nonzero}")
        if not finite:
            print(f"[{name}] FAIL: NaN/Inf im Gradienten von '{pname}'")
            ok = False
        if not nonzero:
            print(f"[{name}] WARN: Gradient von '{pname}' ist exakt 0 (evtl. inaktive Constraints)")

    print(f"[{name}] {'PASSED' if ok else 'FAILED'}")
    return ok

def masked_mean_loss(per_sample_loss: Tensor, layer: _QPLayerBase) -> Tensor | None:
    mask = layer.last_solved_mask
    if mask is None or mask.all():
        return per_sample_loss.mean()
    if not mask.any():
        return None
    return per_sample_loss[mask].mean()

if __name__ == "__main__":
    class _LearnableActionModel(nn.Module):
        def __init__(self, actions_norm: Tensor):
            super().__init__()
            self.actions_norm = nn.Parameter(actions_norm.clone())

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

    # FAF 10 km and IF 18 km in front of the runway fix, on the centerline
    faf_x, faf_y, _ = make_pos(10_000.0, 0.0, 0.0)
    faf_pos = torch.tensor([[faf_x, faf_y]], dtype=torch.float32).repeat(B, 1)
    if_x, if_y, _ = make_pos(18_000.0, 0.0, 0.0)
    if_pos = torch.tensor([[if_x, if_y]], dtype=torch.float32).repeat(B, 1)

    # runway lengths [m]: short (-> 9023 ft floor), long (~11500 ft), unknown (NaN -> 9023 ft floor)
    runway_length_m = torch.tensor([2200.0, 11_500.0 * 0.3048, float("nan")], dtype=torch.float32)

    actions_phys = torch.tensor([
        [55_000.0, 1.00, 2.0],
        [55_000.0, 1.00, 0.0],
        [30_000.0, 1.00, 0.0],
    ])
    actions_norm = (actions_phys - torch.from_numpy(y_mean)) / torch.from_numpy(y_std)

    x_dummy = torch.zeros(B, 15)

    results = {}

    # Envelope
    env_model = _LearnableActionModel(actions_norm)
    env_layer = AircraftEnvelopeQPLayer(base_model=env_model, y_mean=y_mean, y_std=y_std, debug=False, solver=None)
    results["envelope"] = check_gradient_flow(
        "envelope", env_layer, env_model, y_mean, y_std,
        x_dummy, aircraft_state, aircraft_dyn, aircraft_limits,
    )

    # Funnel
    fun_model = _LearnableActionModel(actions_norm)
    fun_layer = AircraftFunnelQPLayer(
        base_model=fun_model, y_mean=y_mean, y_std=y_std, h_thr=4.0, debug=False, solver=None,
        s_activate_funnel=15_000.0, s_activate_track=25_000.0,
    )
    results["funnel"] = check_gradient_flow(
        "funnel", fun_layer, fun_model, y_mean, y_std,
        x_dummy, aircraft_state, aircraft_dyn, aircraft_limits, aircraft_pos, runway_pos,
    )

    # Combined, without FAF/IF -> s_activate_funnel / s_activate_track
    combo_model = _LearnableActionModel(actions_norm)
    combo_layer = AircraftCombinedQPLayer(
        base_model=combo_model, y_mean=y_mean, y_std=y_std, h_thr=4.0, debug=False, solver=None,
        s_activate_funnel=15_000.0, s_activate_track=25_000.0,
    )
    results["combined"] = check_gradient_flow(
        "combined", combo_layer, combo_model, y_mean, y_std,
        x_dummy, aircraft_state, aircraft_dyn, aircraft_limits, aircraft_pos, runway_pos,
    )

    # Combined, with FAF + IF -> FAF / IF distances
    combo_fix_model = _LearnableActionModel(actions_norm)
    combo_fix_layer = AircraftCombinedQPLayer(
        base_model=combo_fix_model, y_mean=y_mean, y_std=y_std, h_thr=4.0, debug=False, solver=None,
        s_activate_funnel=15_000.0, s_activate_track=25_000.0,
    )
    results["combined_fix"] = check_gradient_flow(
        "combined_fix", combo_fix_layer, combo_fix_model, y_mean, y_std,
        x_dummy, aircraft_state, aircraft_dyn, aircraft_limits, aircraft_pos, runway_pos, faf_pos, if_pos,
        runway_length_m,
    )
    print(f"[combined_fix] faf distance = {AircraftFunnelQPLayer.fix_distance(faf_pos, runway_pos).tolist()}  (expected ~10000 m)")
    print(f"[combined_fix] if  distance = {AircraftFunnelQPLayer.fix_distance(if_pos, runway_pos).tolist()}  (expected ~18000 m)")

    d_garp_t, tan_lgsa_t = AircraftFunnelQPLayer.lpv_lateral_geometry(runway_length_m, 106.68)
    print(f"[combined_fix] d_garp = {d_garp_t.tolist()}  (expected ~[3055, 3810, 3055] m)")
    print(f"[combined_fix] lgsa   = {torch.rad2deg(torch.atan(tan_lgsa_t)).tolist()}  (expected ~[2.00, 1.60, 2.00] deg)")
    print(f"[combined_fix] matches scalar default: "
          f"{abs(d_garp_t[0].item() - combo_fix_layer.d_garp) < 1e-3 and abs(tan_lgsa_t[0].item() - combo_fix_layer.tan_lgsa) < 1e-6}")

    print(f"\n{'=' * 70}\nsummary\n{'=' * 70}")
    for name, passed in results.items():
        print(f"  {name:12s}: {'PASSED' if passed else 'FAILED'}")