import math
from itertools import combinations

import cvxpy as cp
import numpy as np
import torch
from torch import nn, Tensor
from cvxpylayers.torch import CvxpyLayer

from ml.utils.models.constraint_layer_cvxpy import AircraftFormulasTorch, _QPLayerBase
from engine.utils_project.separation_geometry import SeparationGeometry, NM_TO_M, FT_TO_M

DEG2RAD = math.pi / 180.0


def scene_groups(batch_idx: Tensor) -> dict[int, Tensor]:
    order = torch.argsort(batch_idx, stable=True)
    _, counts = torch.unique_consecutive(batch_idx[order], return_counts=True)
    groups: dict[int, list[Tensor]] = {}
    for nodes in torch.split(order, counts.tolist()):
        groups.setdefault(int(nodes.numel()), []).append(nodes)
    return {n: torch.stack(nodes_list) for n, nodes_list in groups.items()}


class AircraftSceneSeparationQPLayer(nn.Module):
    """
    Pairwise separation safety filter (HOCBF) for a batch of scenes with a VARIABLE number of aircraft.
    Same barrier as AircraftJointSeparationQPLayer, but:
      - no base_model: it filters a given reference action u_ref (e.g. the output of AircraftCombinedQPLayer)
      - batched over scenes via batch_idx (same convention as MultiAircraftLandingGNN / the multi dataset),
        scenes are grouped by aircraft count, one cvxpy problem per count (built lazily, cached)
      - optionally keeps the per-aircraft rows (envelope + funnel, A_air/b_air) as constraints, so avoiding a
        conflict can't silently break the stall/funnel constraints of stage 1
      - scenes without an active pair (all pairs farther than sep_activate_m) are NOT solved -> output = u_ref
      - a failing solve only affects its own scene: batched solve -> on error per-scene solves -> on error
        the scene keeps u_ref (= the already constrained stage-1 output)

    Barrier (superellipsoid, same exponent n as SeparationGeometry, default n = 4):
        q_h = (Δx² + Δy²) / D_h'²,   q_v = Δh² / D_v'²,   m = n / 2
        h   = q_h^m + q_v^m − 1                      (h > 0 -> separated)
        ḣ   = m q_h^(m−1) q̇_h + m q_v^(m−1) q̇_v
        ḧ   = m(m−1) q_h^(m−2) q̇_h² + m q_h^(m−1) q̈_h + (same for v)
        q̇_h = 2 (Δx Δẋ + Δy Δẏ) / D_h'²,          q̈_h = 2 (Δẋ² + Δẏ² + Δx Δẍ + Δy Δÿ) / D_h'²
        q̇_v = 2 Δh Δḣ / D_v'²,                    q̈_v = 2 (Δḣ² + Δh Δḧ) / D_v'²
      with D' = 2^(1/n) · D (contain_cylinder=True), so the FAA cylinder is fully contained.
      ḧ is affine in the relative acceleration Δp̈ = p̈_i − p̈_j, which is affine in u -> HOCBF row is linear in u:
        ḧ + (k1 + k2) ḣ + k1 k2 h >= 0

    QP per scene with n aircraft, u = [T~_1, uv_1, uh_1, ..., T~_n, uv_n, uh_n]:
        min  ||w (u - u_ref)||^2 + rho_slack ||s||^2
        s.t. box constraints per aircraft (thrust, load factor, roll)
             A_air u + b_air + s >= 0          (optional, per-aircraft rows, block diagonal)
             A_pair u + b_pair + s >= 0        (one HOCBF row per pair, gated by sep_activate_m)
    """

    def __init__(
        self,
        solver: str | None = None,
        d_horizontal_m: float = 2.5 * NM_TO_M,
        d_vertical_m: float = 1000.0 * FT_TO_M,
        sep_exponent: float = 4.0,
        contain_cylinder: bool = True,
        k_sep: tuple[float, float] = (0.1, 0.1),
        sep_activate_m: float = 10.0 * NM_TO_M,
        w_action: tuple[float, float, float] = (1.0, 1.0, 1.0),
        rho_slack: float = 1e5,
        normalize_rows: bool = True,        # scale every row to ||A_row|| = 1 (funnel rows ~1e3, separation rows ~1e-3)
        solver_args: dict | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__()
        geom: SeparationGeometry = SeparationGeometry(
            d_horizontal_m=d_horizontal_m, d_vertical_m=d_vertical_m,
            n=sep_exponent, contain_cylinder=contain_cylinder,
        )
        self.d_h_eff, self.d_v_eff = geom.d_h_eff, geom.d_v_eff
        self.m_sep: float = geom.n / 2.0
        self.register_buffer(
            "scale",
            torch.tensor([1.0 / self.d_h_eff ** 2, 1.0 / self.d_h_eff ** 2, 1.0 / self.d_v_eff ** 2], dtype=torch.float32),
            persistent=False,
        )
        self.k1_s, self.k2_s = k_sep
        self.sep_activate_m: float = sep_activate_m
        self.w_action: tuple[float, float, float] = w_action
        self.rho_slack: float = rho_slack
        self.normalize_rows: bool = normalize_rows

        self.solver: str | None = solver
        self.solver_args: dict = solver_args or {}
        self.debug: bool = debug

        # (n_aircraft, rows_per_aircraft) -> CvxpyLayer; plain dict -> NOT a submodule, not in the state_dict
        self._layers: dict[tuple[int, int], CvxpyLayer] = {}

        self.last_solved_mask: Tensor | None = None  # (N,) bool, False only for aircraft in a scene whose solve failed
        self.last_active_mask: Tensor | None = None  # (N,) bool, aircraft in a scene with >= 1 active pair (QP solved)
        self.last_stats: dict[str, int] = {}

    def _get_layer(self, n: int, r: int) -> CvxpyLayer:
        key = (n, r)
        if key in self._layers:
            return self._layers[key]

        n_pairs = n * (n - 1) // 2
        n_rows = n_pairs + n * r
        u = cp.Variable(3 * n)
        s = cp.Variable(n_rows, nonneg=True)

        u_ref = cp.Parameter(3 * n)
        n_min = cp.Parameter(n, nonneg=True)
        n_max = cp.Parameter(n, nonneg=True)
        tanphi = cp.Parameter(n, nonneg=True)
        A = cp.Parameter((n_rows, 3 * n))
        b = cp.Parameter(n_rows)

        w_full = np.tile(np.sqrt(np.asarray(self.w_action, dtype=float)), n)

        constraints = []
        for i in range(n):
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

        objective = cp.Minimize(cp.sum_squares(cp.multiply(w_full, u - u_ref)) + self.rho_slack * cp.sum_squares(s))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp(), "separation QP is not DPP conform"

        layer = CvxpyLayer(problem, parameters=[u_ref, n_min, n_max, tanphi, A, b], variables=[u, s], solver=self.solver)
        self._layers[key] = layer
        return layer

    @staticmethod
    def _accel_coeffs(h, v, gamma, chi, mass, C_D0, area, cl_max, k, t_max, n_ref):
        """same as AircraftJointSeparationQPLayer._accel_coeffs: p_ddot = a @ u + b, velocity vector"""
        F = AircraftFormulasTorch
        gam, chir = torch.deg2rad(gamma), torch.deg2rad(chi)
        sing, cosg = torch.sin(gam), torch.cos(gam)
        sinc, cosc = torch.sin(chir), torch.cos(chir)

        requested_cl = F.required_lift_coefficient(h, v, n_ref, mass, area)
        actual_cl = torch.minimum(requested_cl, cl_max)
        drag = F.drag_force(h, v, actual_cl, C_D0, k, area)

        a_x = torch.stack([(t_max / mass) * cosg * sinc, -F.g * sing * sinc, F.g * cosc], dim=-1)
        a_y = torch.stack([(t_max / mass) * cosg * cosc, -F.g * sing * cosc, -F.g * sinc], dim=-1)
        a_h = torch.stack([(t_max / mass) * sing, F.g * cosg, torch.zeros_like(cosg)], dim=-1)
        b_vec = torch.stack([
            -(drag / mass) * cosg * sinc,
            -(drag / mass) * cosg * cosc,
            -(drag / mass) * sing - F.g,
        ], dim=-1)
        vel = torch.stack([v * cosg * sinc, v * cosg * cosc, v * sing], dim=-1)
        a = torch.stack([a_x, a_y, a_h], dim=-2)  # (..., 3 [x,y,h], 3 [T~,uv,uh])
        return a, b_vec, vel

    def _barrier_terms(self, dpos: Tensor, dvel: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        m = self.m_sep
        scale = self.scale.to(dpos)

        q_h = (dpos[..., :2] ** 2).sum(-1) * scale[0]
        q_v = dpos[..., 2] ** 2 * scale[2]
        qd_h = 2.0 * (dpos[..., :2] * dvel[..., :2]).sum(-1) * scale[0]
        qd_v = 2.0 * dpos[..., 2] * dvel[..., 2] * scale[2]

        q_h_c, q_v_c = q_h.clamp(min=1e-12), q_v.clamp(min=1e-12)
        g1_h, g1_v = m * q_h_c ** (m - 1.0), m * q_v_c ** (m - 1.0)              # ∂h/∂q
        g2_h, g2_v = m * (m - 1.0) * q_h_c ** (m - 2.0), m * (m - 1.0) * q_v_c ** (m - 2.0)  # ∂²h/∂q²

        h_val = q_h ** m + q_v ** m - 1.0                      # > 0 -> separated
        h_dot = g1_h * qd_h + g1_v * qd_v

        w = torch.stack([g1_h, g1_h, g1_v], dim=-1)
        coef = 2.0 * w * dpos * scale                           # ∂ḧ/∂(p̈_i − p̈_j)
        dot_sq = g2_h * qd_h ** 2 + g2_v * qd_v ** 2 + 2.0 * (w * dvel ** 2 * scale).sum(-1)
        return h_val, h_dot, coef, dot_sq

    def _pair_rows(
        self, idx: Tensor, p_xyz: Tensor, a_all: Tensor, b_all: Tensor, vel_all: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """idx (S, n) -> A (S, P, 3n), b (S, P), active (S, P), h_val (S, P)"""
        S, n = idx.shape
        pairs = torch.tensor(list(combinations(range(n), 2)), dtype=torch.long, device=idx.device)
        I, J = pairs[:, 0], pairs[:, 1]

        p, vel, a, bv = p_xyz[idx], vel_all[idx], a_all[idx], b_all[idx]   # (S,n,3), (S,n,3), (S,n,3,3), (S,n,3)

        dpos = p[:, I] - p[:, J]                     # (S, P, 3)
        dvel = vel[:, I] - vel[:, J]
        h_val, h_dot, coef, dot_sq = self._barrier_terms(dpos, dvel)

        row_i = torch.einsum("spk,spkc->spc", coef, a[:, I])     # (S, P, 3)
        row_j = -torch.einsum("spk,spkc->spc", coef, a[:, J])
        onehot_i = nn.functional.one_hot(I, n).to(p.dtype)       # (P, n)
        onehot_j = nn.functional.one_hot(J, n).to(p.dtype)
        A = (torch.einsum("spc,pn->spnc", row_i, onehot_i)
             + torch.einsum("spc,pn->spnc", row_j, onehot_j)).reshape(S, -1, 3 * n)

        # HOCBF: ḧ + (k1 + k2) ḣ + k1 k2 h >= 0, with ḧ = coef · (a_i u_i - a_j u_j + b_i - b_j) + dot_sq
        b = (coef * (bv[:, I] - bv[:, J])).sum(-1) + dot_sq \
            + (self.k1_s + self.k2_s) * h_dot + self.k1_s * self.k2_s * h_val

        active = dpos.norm(dim=-1) < self.sep_activate_m
        A = A * active.unsqueeze(-1).to(A.dtype)
        b = torch.where(active, b, torch.ones_like(b))
        return A, b, active, h_val

    @staticmethod
    def _aircraft_rows(idx: Tensor, A_air: Tensor, b_air: Tensor) -> tuple[Tensor, Tensor]:
        """per-aircraft rows (N, r, 3) / (N, r) -> block diagonal (S, n*r, 3n) / (S, n*r)"""
        S, n = idx.shape
        r = A_air.shape[1]
        eye = torch.eye(n, dtype=A_air.dtype, device=A_air.device)
        A_blk = torch.einsum("skrc,kl->skrlc", A_air[idx], eye).reshape(S, n * r, 3 * n)
        return A_blk, b_air[idx].reshape(S, n * r)

    @staticmethod
    def _normalize(A: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        norm = A.norm(dim=-1)
        row_scale = torch.where(norm > 1e-6, 1.0 / norm.clamp(min=1e-6), torch.ones_like(norm)).detach()
        return A * row_scale.unsqueeze(-1), b * row_scale

    @staticmethod
    def to_qp_coords(u_phys: Tensor, t_max: Tensor) -> Tensor:
        """(thrust [N], n [-], roll [deg]) -> (T~, uv, uh), inverse of _QPLayerBase._unpack"""
        thrust, n, roll = u_phys.unbind(dim=-1)
        phi = torch.deg2rad(roll)
        return torch.stack([thrust / t_max, n * torch.cos(phi), n * torch.sin(phi)], dim=-1)

    def _solve_group(self, layer: CvxpyLayer, args: list[Tensor]) -> tuple[Tensor, Tensor]:
        """args: batched (S, ...) cpu tensors -> u (S, 3n), solved (S,) bool. Errors only hit their own scene."""
        S = args[0].shape[0]
        finite = torch.stack([torch.isfinite(t.reshape(S, -1)).all(dim=1) for t in args]).all(dim=0)
        u_sol = torch.zeros_like(args[0])
        solved = torch.zeros(S, dtype=torch.bool)

        fidx = finite.nonzero(as_tuple=True)[0]
        if fidx.numel() == 0:
            return u_sol, solved
        try:
            u, _ = layer(*[t[fidx] for t in args], solver_args=self.solver_args)
            u_sol[fidx] = u.to(u_sol.dtype)
            solved[fidx] = True
        except Exception as e:
            if self.debug:
                print(f"[sep-qp] WARN: batched solve over {fidx.numel()} scenes failed ({e}) -> per-scene solves")
            for i in fidx.tolist():
                try:
                    u, _ = layer(*[t[i:i + 1] for t in args], solver_args=self.solver_args)
                    u_sol[i] = u[0].to(u_sol.dtype)
                    solved[i] = True
                except Exception:
                    pass
        return u_sol, solved

    def forward(
        self,
        u_ref_phys: Tensor,          # (N, 3) reference action (thrust [N], n [-], roll [deg]), e.g. AircraftCombinedQPLayer output
        aircraft_state: Tensor,      # (N, 3) h, v, gamma
        aircraft_dyn: Tensor,        # (N, 5) mass, C_D0, area, cl_max, k
        aircraft_limits: Tensor,     # (N, 4) max_thrust, min_load_factor, max_load_factor, max_roll_angle
        aircraft_pos: Tensor,        # (N, 3) x, y, chi
        batch_idx: Tensor | None = None,  # (N,) scene id per aircraft; None -> all aircraft in one scene (inference)
        A_air: Tensor | None = None,      # (N, r, 3) optional per-aircraft rows kept as constraints
        b_air: Tensor | None = None,      # (N, r)
    ) -> Tensor:
        N = u_ref_phys.shape[0]
        device = u_ref_phys.device
        if batch_idx is None:
            batch_idx = torch.zeros(N, dtype=torch.long, device=device)

        t_max, n_min, n_max, roll_max = aircraft_limits.unbind(dim=-1)
        n_min = torch.clamp(n_min, min=0.0)
        tanphi = torch.tan(torch.deg2rad(roll_max))
        h, v, gamma = aircraft_state.unbind(dim=-1)
        mass, C_D0, area, cl_max, k = aircraft_dyn.unbind(dim=-1)

        u_ref = self.to_qp_coords(u_ref_phys, t_max)
        a_all, b_all, vel_all = self._accel_coeffs(
            h, v, gamma, aircraft_pos[:, 2], mass, C_D0, area, cl_max, k, t_max, u_ref_phys[:, 1],
        )
        p_xyz = torch.stack([aircraft_pos[:, 0], aircraft_pos[:, 1], h], dim=-1)
        groups = scene_groups(batch_idx)

        # separation QP, only for scenes with at least one active pair
        touched = torch.zeros(N, dtype=torch.bool, device=device)
        u_out = u_ref.clone()
        solved_mask = torch.ones(N, dtype=torch.bool, device=device)
        active_mask = torch.zeros(N, dtype=torch.bool, device=device)
        stats = {"scenes": 0, "scenes_multi": 0, "scenes_active": 0, "scenes_failed": 0}
        min_h = float("inf")

        for n, idx in groups.items():
            stats["scenes"] += idx.shape[0]
            if n < 2:
                continue
            stats["scenes_multi"] += idx.shape[0]

            A_p, b_p, active, h_val = self._pair_rows(idx, p_xyz, a_all, b_all, vel_all)
            need = active.any(dim=-1)
            if not need.any():
                continue
            idx_s, A_p, b_p = idx[need], A_p[need], b_p[need]
            min_h = min(min_h, float(h_val[need].min()))

            r = 0
            if A_air is not None and b_air is not None:
                r = A_air.shape[1]
                A_a, b_a = self._aircraft_rows(idx_s, A_air, b_air)
                A_p, b_p = torch.cat([A_a, A_p], dim=1), torch.cat([b_a, b_p], dim=1)
            if self.normalize_rows:
                A_p, b_p = self._normalize(A_p, b_p)

            S = idx_s.shape[0]
            args = [
                u_ref[idx_s].reshape(S, 3 * n), n_min[idx_s], n_max[idx_s], tanphi[idx_s], A_p, b_p,
            ]
            args_cpu = [t.cpu() for t in args] if device.type != "cpu" else args
            u_sol, solved = self._solve_group(self._get_layer(n, r), args_cpu)
            u_sol, solved = u_sol.to(device), solved.to(device)

            ok_nodes = idx_s[solved].reshape(-1)
            u_out[ok_nodes] = u_sol[solved].reshape(-1, 3)
            solved_mask[idx_s[~solved].reshape(-1)] = False
            active_mask[idx_s.reshape(-1)] = True
            touched[ok_nodes] = True
            stats["scenes_active"] += S
            stats["scenes_failed"] += int((~solved).sum())

        self.last_solved_mask = solved_mask
        self.last_active_mask = active_mask
        self.last_stats = stats

        # untouched aircraft keep u_ref_phys exactly (no unnecessary round trip through (T~, uv, uh))
        out = torch.where(touched.unsqueeze(-1), _QPLayerBase._unpack(u_out, t_max), u_ref_phys)

        if self.debug and stats["scenes_active"] > 0:
            print(f"[sep-qp] scenes={stats['scenes']} multi={stats['scenes_multi']} active={stats['scenes_active']} "
                  f"failed={stats['scenes_failed']} min_h={min_h:+.3f}")
        return out