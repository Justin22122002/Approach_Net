import math

import numpy as np
import torch
from torch import nn, Tensor

DEG2RAD = math.pi / 180.0
GLIDESLOPE_DEG = 3.0
G = 9.81
RHO0 = 1.225
H_SCALE = 287.05 * 288.15 / G  # (R * T) / g


class RunwayGuidanceLayer(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        activation_distance_m: float = 6000.0,
        capture_halfangle_deg: float = 90.0,  # Heading-Cone: w_align=0 from this angle to chi_r
        gain_lateral: float = 0.4,
        gain_heading: float = 0.6,
        gain_glideslope: float = 0.5,
        e_y_ref_m: float = 300.0,
        e_z_ref_m: float = 100.0,
        max_roll_adjust_deg: float = 15.0,
        max_load_adjust: float = 0.5,
        stall_margin: float = 1.15,      # v >= margin * v_stall should stay
        gain_stall_thrust: float = 3.0,  # scaling of Thrust-Boost relative to max_thrust
        debug: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))
        self.activation_distance_m = activation_distance_m
        self.capture_halfangle_deg = capture_halfangle_deg
        self.gain_lateral = gain_lateral
        self.gain_heading = gain_heading
        self.gain_glideslope = gain_glideslope
        self.e_y_ref_m = e_y_ref_m
        self.e_z_ref_m = e_z_ref_m
        self.max_roll_adjust_deg = max_roll_adjust_deg
        self.max_load_adjust = max_load_adjust
        self.stall_margin = stall_margin
        self.gain_stall_thrust = gain_stall_thrust
        self.debug = debug

    @torch.no_grad()
    def forward(
        self,
        x_norm: Tensor,              # (B, input_dim) normalised model input
        aircraft_pos: Tensor,        # (B, 4) raw: x, y, h [m], chi [deg]  (airport-centered UTM)
        runway_pos: Tensor,          # (B, 4) raw: fix_x, fix_y [m], fix_elevation_m, runway_bearing [deg]
        aircraft_state_dyn: Tensor,  # (B, 4) raw: v [m/s], mass [kg], area [m2], cl_max [-]
        max_thrust: Tensor,          # (B,) raw: max_thrust [N]
    ) -> Tensor:
        raw_norm = self.base_model(x_norm)                     # (B,3) normalised
        raw_phys = raw_norm * self.y_std + self.y_mean         # (B,3) physical: thrust[N], n[-], roll[deg]
        thrust, load_factor, roll = raw_phys.unbind(dim=-1)

        x_ac, y_ac, h, chi = aircraft_pos.unbind(dim=-1)
        fix_x, fix_y, fix_elev, chi_r = runway_pos.unbind(dim=-1)

        # 1.position of the aircraft is relative to the fix, direct in meter.
        #    x = Easting (East+), y = Northing (Nord+).
        x_rel = x_ac - fix_x   # East-offset airplane ggue. Fix
        y_rel = y_ac - fix_y   # North-offset

        # Runway-heading-vector (East, North) = (sin(chi_r), cos(chi_r))
        chi_r_rad = chi_r * DEG2RAD
        s = -(x_rel * torch.sin(chi_r_rad) + y_rel * torch.cos(chi_r_rad))   # Along-Track in front of s
        e_y = x_rel * torch.cos(chi_r_rad) - y_rel * torch.sin(chi_r_rad)    # lateral offset to the centerline
        e_z = h - fix_elev - s * math.tan(GLIDESLOPE_DEG * DEG2RAD)          # Glideslope-offset/error

        # 2. geometric course error: vector from the aircraft to the Fix
        dx_to_rwy = -x_rel
        dy_to_rwy = -y_rel
        target_chi_rad = torch.atan2(dx_to_rwy, dy_to_rwy)  # atan2(Ost, Nord) = Bearing von Nord

        chi_rad = chi * DEG2RAD
        delta_chi = target_chi_rad - chi_rad
        delta_chi = torch.atan2(torch.sin(delta_chi), torch.cos(delta_chi))

        # 3. activation heading  w = w_along * w_align
        #    w_along: only in front of s  (s>0) and in the Capture-distance.
        #             Behind s (s<=0) hard 0 -> no Glideslope.
        w_along = torch.clamp(1.0 - torch.clamp(s, min=0.0) / self.activation_distance_m, 0.0, 1.0)
        w_along = torch.where(s > 0.0, w_along, torch.zeros_like(w_along))

        #    w_align: Heading needs to be slightly the way of the Runway-heading. cos(chi - chi_r)=1
        #             with perfect Alignment, -1 the opposite way. Cone above
        #             capture_halfangle_deg -> other heading/Base/Downwind -> 0.
        align = torch.cos(chi_rad - chi_r_rad)
        cos_gate = math.cos(self.capture_halfangle_deg * DEG2RAD)
        w_align = torch.clamp((align - cos_gate) / (1.0 - cos_gate + 1e-6), 0.0, 1.0)

        w = w_along * w_align

        # 4. Regler (will be guided completely by w)
        roll_adjust = (
                self.gain_lateral * (e_y / self.e_y_ref_m)
                + self.gain_heading * torch.rad2deg(delta_chi)
        )
        roll_adjust = torch.clamp(roll_adjust, -self.max_roll_adjust_deg, self.max_roll_adjust_deg)

        load_adjust = -self.gain_glideslope * (e_z / self.e_z_ref_m)
        load_adjust = torch.clamp(load_adjust, -self.max_load_adjust, self.max_load_adjust)

        roll_out = roll + w * roll_adjust
        load_factor_out = load_factor + w * load_adjust

        # Stall-protection on Thrust (always active, dependent on w)
        v, mass, area, cl_max = aircraft_state_dyn.unbind(dim=-1)
        rho = RHO0 * torch.exp(-h / H_SCALE)
        v_stall = torch.sqrt(2.0 * mass * G / torch.clamp(rho * cl_max * area, min=1e-6))
        v_target = self.stall_margin * v_stall

        deficit = torch.clamp((v_target - v) / torch.clamp(v_target, min=1e-6), min=0.0)
        thrust_boost = self.gain_stall_thrust * deficit * max_thrust
        thrust_out = torch.clamp(thrust + thrust_boost, min=torch.zeros_like(max_thrust), max=max_thrust)

        if self.debug:
            print(
                f"[guidance] s={s.item():8.1f}m  e_y={e_y.item():7.1f}m  e_z={e_z.item():7.1f}m  "
                f"dchi={math.degrees(delta_chi.item()):6.1f}°  align={align.item():+.2f}  "
                f"wA={w_along.item():.2f} wH={w_align.item():.2f} w={w.item():.3f}  |  "
                f"roll {roll.item():+6.2f} -> {roll_out.item():+6.2f}° (adj {roll_adjust.item():+6.2f}°)  |  "
                f"n {load_factor.item():5.2f} -> {load_factor_out.item():5.2f} (adj {load_adjust.item():+5.2f})  |  "
                f"v={v.item():5.1f} v_stall={v_stall.item():5.1f} v_target={v_target.item():5.1f}  "
                f"T {thrust.item()/1e3:6.1f}kN -> {thrust_out.item()/1e3:6.1f}kN (boost {thrust_boost.item()/1e3:+6.1f}kN)"
            )

        return torch.stack([thrust_out, load_factor_out, roll_out], dim=-1)