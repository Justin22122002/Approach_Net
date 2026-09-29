import math
from itertools import combinations

import torch
from torch import nn, Tensor

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from ml.utils.models.constraint_layer_cvxpy import AircraftFormulasTorch
from utils_project.separation_geometry import SeparationGeometry, NM_TO_M, FT_TO_M

DEG2RAD = math.pi / 180.0


class AircraftJointSeparationQPLayer(nn.Module):
    """
    Joint HOCBF safety QP for ALL aircraft in a scene simultaneously,
    ONE solve per timestep. Decision vector u = [T~_1,uv_1,uh_1, ...
    T~_N,uv_N,uh_N] (3N variables), with ONE HOCBF row PER PAIR (i,j).

    Two-stage:
      1) Conflict detection + resolution strategy (reference shaping, before the QP)
         - Detection: linear extrapolation, time until entering the
           separation ellipse |dp + dv t|_S = 1. A conflict is detected if entry
           occurs within lookahead_s (or if the pair is already inside).
         - in-trail (heading difference < intrail_angle_deg): speed ->
           leading aircraft +speed_bias thrust, trailing aircraft -speed_bias.
         - Crossing: complementary vertical resolution (similar to TCAS RA coordination) ->
           higher aircraft climbs (+vertical_bias load factor), lower aircraft
           descends; if |dh| < dh_tie_m, the aircraft index breaks the tie (smaller index
           climbs). Only while the relative vertical rate is below
           2 * vertical_rate_mps (TCAS: 1500 ft/min per aircraft).
         The bias only shifts the reference u_nn -> QP unchanged; no
         additional weighting is required (w_action remains (1,1,1)).
      2) HOCBF-QP: guarantees separation (soft constraint with slack) if the
         strategy alone is insufficient.

    Why not rely on the QP alone: the separation barrier h = dp^T S dp - 1
    only captures the RADIAL relative acceleration in h_ddot. The minimum-norm correction
    produces mirrored maneuvers for both aircraft; the lateral component is
    proportional to cos(crossing angle) and vanishes exactly at 90 degrees
    -> the QP only slows the approach and intervenes late and aggressively.
    """

    def __init__(
        self,
        base_model: nn.Module,
        y_mean: float | Tensor,
        y_std: float | Tensor,
        n_aircraft: int,
        d_horizontal_m: float = 2.5 * NM_TO_M,
        d_vertical_m: float = 1000.0 * FT_TO_M,
        contain_cylinder: bool = True,
        k_sep: tuple[float, float] = (0.4, 0.4),
        sep_activate_m: float = 10.0 * NM_TO_M,
        w_action: tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_resolution: bool = True,
        lookahead_s: float = 120.0,                            # Conflict-detection lookahead
        intrail_angle_deg: float = 45.0,                       # below this: in-trail -> speed resolution
        speed_bias: float = 0.2,                               # thrust bias (fraction of T_max): leading +, trailing -
        vertical_bias: float = 0.1,                            # load-factor bias (0.1 g): climb/descend
        vertical_rate_mps: float = 1500.0 * FT_TO_M / 60.0,    # TCAS RA target rate per aircraft
        dh_tie_m: float = 15.0,                                # below this: break ties using aircraft index
        rho_slack: float = 1e5,
        solver_args: dict | None = None,
        debug: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))
        self.N = n_aircraft
        self.pairs = list(combinations(range(n_aircraft), 2))
        self.n_pairs = len(self.pairs)

        geom: SeparationGeometry = SeparationGeometry(
            d_horizontal_m=d_horizontal_m, d_vertical_m=d_vertical_m,
            n=4.0, contain_cylinder=contain_cylinder,
        )
        self.d_h_eff, self.d_v_eff = geom.d_h_eff, geom.d_v_eff
        self.scale = torch.tensor([1.0 / self.d_h_eff ** 2, 1.0 / self.d_h_eff ** 2, 1.0 / self.d_v_eff ** 2])
        self.k1_s, self.k2_s = k_sep
        self.sep_activate_m = sep_activate_m
        self.w_action = w_action

        self.use_resolution = use_resolution
        self.lookahead_s = lookahead_s
        self.intrail_rad = intrail_angle_deg * DEG2RAD
        self.speed_bias = speed_bias
        self.vertical_bias = vertical_bias
        self.vertical_rate_mps = vertical_rate_mps
        self.dh_tie_m = dh_tie_m

        self.rho_slack = rho_slack
        self.solver_args = solver_args or {}
        self.debug = debug
        self.last_solved: bool = True
        self.last_tags: list[str] = [""] * n_aircraft

        self.layer = self._build_qp()

    def _build_qp(self) -> CvxpyLayer:
        N, P = self.N, self.n_pairs
        u = cp.Variable(3 * N)
        s = cp.Variable(max(P, 1), nonneg=True)

        u_nn = cp.Parameter(3 * N)
        n_min = cp.Parameter(N, nonneg=True)
        n_max = cp.Parameter(N, nonneg=True)
        tanphi = cp.Parameter(N, nonneg=True)
        A = cp.Parameter((max(P, 1), 3 * N))
        b = cp.Parameter(max(P, 1))

        w = torch.sqrt(torch.tensor(self.w_action)).numpy()
        w_full = list(w) * N

        constraints = []
        for i in range(N):
            ui = u[3 * i:3 * i + 3]
            constraints += [
                ui[0] >= 0.0,
                ui[0] <= 1.0,
                cp.norm(ui[1:3], 2) <= n_max[i],
                ui[1] >= n_min[i],
                ui[2] <= tanphi[i] * ui[1],
                -ui[2] <= tanphi[i] * ui[1],
            ]
        constraints.append(A @ u + b + s >= 0.0)

        objective = cp.Minimize(
            cp.sum_squares(cp.multiply(w_full, u - u_nn)) + self.rho_slack * cp.sum_squares(s)
        )
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp(), "Joint-QP ist nicht DPP-konform"
        return CvxpyLayer(problem, parameters=[u_nn, n_min, n_max, tanphi, A, b], variables=[u, s])

    def _decode_actions(self, x_norm: Tensor, t_max: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
    def _accel_coeffs(h, v, gamma, chi, mass, C_D0, area, cl_max, k, t_max, n_nn):
        F = AircraftFormulasTorch
        gam, chir = torch.deg2rad(gamma), torch.deg2rad(chi)
        sing, cosg = torch.sin(gam), torch.cos(gam)
        sinc, cosc = torch.sin(chir), torch.cos(chir)

        requested_cl = F.required_lift_coefficient(h, v, n_nn, mass, area)
        actual_cl = torch.minimum(requested_cl, cl_max)
        drag = F.drag_force(h, v, actual_cl, C_D0, k, area)

        a_x_T = (t_max / mass) * cosg * sinc
        a_x_uv = -F.g * sing * sinc
        a_x_uh = F.g * cosc
        b_x = -(drag / mass) * cosg * sinc

        a_y_T = (t_max / mass) * cosg * cosc
        a_y_uv = -F.g * sing * cosc
        a_y_uh = -F.g * sinc
        b_y = -(drag / mass) * cosg * cosc

        a_h_T = (t_max / mass) * sing
        a_h_uv = F.g * cosg
        a_h_uh = torch.zeros_like(a_h_uv)
        b_h = -(drag / mass) * sing - F.g

        x_dot = v * cosg * sinc
        y_dot = v * cosg * cosc
        h_dot = v * sing

        a = torch.stack([
            torch.stack([a_x_T, a_x_uv, a_x_uh], dim=-1),
            torch.stack([a_y_T, a_y_uv, a_y_uh], dim=-1),
            torch.stack([a_h_T, a_h_uv, a_h_uh], dim=-1),
        ], dim=-2)
        b_vec = torch.stack([b_x, b_y, b_h], dim=-1)
        vel = torch.stack([x_dot, y_dot, h_dot], dim=-1)
        return a, b_vec, vel

    def compute_pairwise_row(self, i: int, j: int, pos, state, dyn, limits, n_nn):
        x, y, chi = pos[:, 0], pos[:, 1], pos[:, 2]
        h, v, gamma = state[:, 0], state[:, 1], state[:, 2]
        mass, C_D0, area, cl_max, k = dyn.unbind(dim=-1)
        t_max = limits[:, 0]

        a_all, b_all, vel_all = self._accel_coeffs(
            h, v, gamma, chi, mass, C_D0, area, cl_max, k, t_max, n_nn,
        )

        dpos = torch.stack([x[i] - x[j], y[i] - y[j], h[i] - h[j]])
        dvel = vel_all[i] - vel_all[j]
        scale = self.scale.to(dpos)

        h_val = (dpos ** 2 * scale).sum() - 1.0
        h_dot = 2.0 * (dpos * dvel * scale).sum()
        coef = 2.0 * dpos * scale
        dot_sq = 2.0 * (dvel ** 2 * scale).sum()

        row = torch.zeros(3 * self.N, dtype=dpos.dtype, device=dpos.device)
        row[3 * i:3 * i + 3] = coef @ a_all[i]
        row[3 * j:3 * j + 3] = -(coef @ a_all[j])
        const = (coef * (b_all[i] - b_all[j])).sum() + dot_sq

        b_row = const + (self.k1_s + self.k2_s) * h_dot + self.k1_s * self.k2_s * h_val
        dist = torch.linalg.norm(dpos)
        return row, b_row, dist, h_val

    # Stage 1: Conflict detection + resolution strategy
    def predict_conflict(self, i: int, j: int, pos: Tensor, state: Tensor, vel_all: Tensor) -> tuple[bool, float]:
        """Linear extrapolation: time until entering the separation ellipse.
         Conflict is detected if already inside or if entry occurs within lookahead_s."""
        dpos = torch.stack([pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1], state[i, 0] - state[j, 0]])
        dvel = vel_all[i] - vel_all[j]
        scale = self.scale.to(dpos)
        a = float((dvel ** 2 * scale).sum())
        b = float(2.0 * (dpos * dvel * scale).sum())
        c = float((dpos ** 2 * scale).sum() - 1.0)
        if c <= 0.0:
            return True, 0.0
        disc = b * b - 4.0 * a * c
        if a < 1e-12 or disc < 0.0:
            return False, float("inf")
        t_in = (-b - math.sqrt(disc)) / (2.0 * a)
        return (0.0 < t_in < self.lookahead_s), t_in

    def resolution_bias(self, pos: Tensor, state: Tensor, vel_all: Tensor) -> tuple[Tensor, list[str]]:
        """Apply a bias to the reference (T~, uv, uh) for each aircraft + tags for debugging/HUD."""
        N = self.N
        bias = torch.zeros(N, 3, dtype=pos.dtype, device=pos.device)
        tags: list[list[str]] = [[] for _ in range(N)]
        chi = torch.deg2rad(pos[:, 2])

        for i, j in self.pairs:
            conflict, t_in = self.predict_conflict(i, j, pos, state, vel_all)
            if not conflict:
                continue

            dchi = float(torch.atan2(torch.sin(chi[i] - chi[j]), torch.cos(chi[i] - chi[j])))
            if abs(dchi) < self.intrail_rad:
                # in-trail -> speed: leading aircraft faster, trailing aircraft slower
                heading = torch.stack([torch.sin(chi[i]) + torch.sin(chi[j]),
                                       torch.cos(chi[i]) + torch.cos(chi[j])])
                dxy = torch.stack([pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]])
                lead, fol = (i, j) if float(dxy @ heading) > 0.0 else (j, i)
                bias[lead, 0] += self.speed_bias
                bias[fol, 0] -= self.speed_bias
                tags[lead].append(f"SPD+({t_in:.0f}s)")
                tags[fol].append(f"SPD-({t_in:.0f}s)")
            else:
                # crossing -> complementary vertical resolution: higher aircraft climbs, lower aircraft descends
                dh = float(state[i, 0] - state[j, 0])
                dhdot = float(vel_all[i, 2] - vel_all[j, 2])
                if abs(dh) > self.dh_tie_m:
                    up = i if dh > 0.0 else j
                else:
                    up = i                                    # Break tie using aircraft index (smaller index climbs)
                down = j if up == i else i
                sign = 1.0 if up == i else -1.0               # Sign of (h_i - h_j) after the resolution
                if sign * dhdot < 2.0 * self.vertical_rate_mps:
                    bias[up, 1] += self.vertical_bias
                    bias[down, 1] -= self.vertical_bias
                tags[up].append(f"UP({t_in:.0f}s)")
                tags[down].append(f"DOWN({t_in:.0f}s)")

        return bias, [",".join(t) for t in tags]

    def forward(
        self,
        x_norm: Tensor,
        aircraft_state: Tensor,
        aircraft_dyn: Tensor,
        aircraft_limits: Tensor,
        aircraft_pos: Tensor,
    ) -> Tensor:
        N, P = self.N, self.n_pairs
        t_max = aircraft_limits[:, 0]
        n_min = aircraft_limits[:, 1]
        n_max = aircraft_limits[:, 2]
        tanphi = torch.tan(torch.deg2rad(aircraft_limits[:, 3]))

        T_nn, n_nn, roll_nn, u_nn_2d = self._decode_actions(x_norm, t_max)

        # step 1: conflict resolution + strategy -> Reference ---
        if self.use_resolution and P > 0:
            h, v, gamma = aircraft_state.unbind(dim=-1)
            mass, C_D0, area, cl_max, k = aircraft_dyn.unbind(dim=-1)
            _, _, vel_all = self._accel_coeffs(
                h, v, gamma, aircraft_pos[:, 2], mass, C_D0, area, cl_max, k, t_max, n_nn,
            )
            bias, tags = self.resolution_bias(aircraft_pos, aircraft_state, vel_all.detach())
        else:
            bias, tags = torch.zeros_like(u_nn_2d), [""] * N
        self.last_tags = tags

        u_ref = u_nn_2d + bias
        u_ref = torch.stack([
            torch.clamp(u_ref[:, 0], 0.0, 1.0),
            torch.maximum(torch.minimum(u_ref[:, 1], n_max), n_min),
            u_ref[:, 2],
        ], dim=1)
        u_nn = u_ref.reshape(-1)

        # step 2: HOCBF-pair-rows (safe guarantee)
        rows, bs, dists, hs = [], [], [], []
        for i, j in self.pairs:
            row, b_row, dist, h_val = self.compute_pairwise_row(
                i, j, aircraft_pos, aircraft_state, aircraft_dyn, aircraft_limits, n_nn,
            )
            active = (dist < self.sep_activate_m).to(row.dtype)
            rows.append(row * active)
            bs.append(torch.where(active > 0, b_row, torch.ones_like(b_row)))
            dists.append(dist)
            hs.append(h_val)

        if P == 0:
            A = torch.zeros(1, 3 * N)
            b = torch.ones(1)
        else:
            A = torch.stack(rows)
            b = torch.stack(bs)

        try:
            u_opt, s_opt = self.layer(u_nn, n_min, n_max, tanphi, A, b, solver_args=self.solver_args)
            self.last_solved = True
        except Exception as e:
            if self.debug:
                print(f"[joint-qp] WARN: solve failed ({e}) -> box-clamp fallback")
            u_opt = torch.zeros(3 * N)
            for idx in range(N):
                t_hat = torch.clamp(u_nn[3 * idx], 0.0, 1.0)
                uv = torch.clamp(u_nn[3 * idx + 1], min=n_min[idx], max=n_max[idx])
                uh = torch.clamp(u_nn[3 * idx + 2], min=-tanphi[idx] * uv, max=tanphi[idx] * uv)
                u_opt[3 * idx:3 * idx + 3] = torch.stack([t_hat, uv, uh])
            s_opt = None
            self.last_solved = False

        u_opt = u_opt.reshape(N, 3)
        thrust = t_max * torch.clamp(u_opt[:, 0], 0.0, 1.0)
        n_out = torch.linalg.norm(u_opt[:, 1:3], dim=1)
        roll_out = torch.rad2deg(torch.atan2(u_opt[:, 2], u_opt[:, 1]))
        out = torch.stack([thrust, n_out, roll_out], dim=-1)

        if self.debug:
            for idx in range(N):
                print(f"[joint-qp] AC{idx}: T {T_nn[idx]/1e3:6.1f}->{out[idx,0]/1e3:6.1f}kN  "
                      f"n {n_nn[idx]:5.2f}->{out[idx,1]:5.2f}  roll {roll_nn[idx]:+6.1f}->{out[idx,2]:+6.1f}deg  "
                      f"res={tags[idx] or '-'}")
            if P > 0:
                slack_max = float(s_opt.max()) if s_opt is not None else float("nan")
                print(f"[joint-qp] min_h={min(hs).item():+.3f}  min_dist={min(dists).item():.0f}m  "
                      f"slack_max={slack_max:.3g}  solved={self.last_solved}")
        return out