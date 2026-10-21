from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from utils_project.rnav_lpv_funnel_geometry import LpvFunnelGeometry

FT_TO_M = 0.3048

H_LTP = 0.0
TCH = 15.0
S_FAF = 13_000.0
RUNWAY_LENGTH = 3_000.0
RUNWAY_WIDTH = 45.0
GLIDESLOPE_DEG = 3.0

C_FUNNEL, C_FILL, C_CENTER, C_RUNWAY, C_GROUND = "#c25a00", "#fbe3cf", "#1b4f8a", "#4a4a4a", "#2e7d32"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def project(funnel: LpvFunnelGeometry, x: float, y: float) -> tuple[float, float]:
    dx, dy = x - funnel.thr_x, y - funnel.thr_y
    return dx * funnel.along[0] + dy * funnel.along[1], dx * funnel.lateral[0] + dy * funnel.lateral[1]


def bisect(pred, lo: float, hi: float, iters: int = 60) -> float:
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if pred(mid) else (mid, hi)
    return 0.5 * (lo + hi)


def label_point(ax, xy, text, offset, ha="center", va="bottom", color="k") -> None:
    ax.plot(*xy, "o", color=color, ms=4, zorder=6)
    ax.annotate(text, xy, xytext=offset, textcoords="offset points", ha=ha, va=va,
                fontsize=8.5, fontweight="bold", color=color, zorder=7)


def draw_runway(ax, x0: float, x1: float, half_width: float, label_y: float) -> None:
    ax.add_patch(Rectangle((x0, -half_width), x1 - x0, 2 * half_width, fc="0.55", ec=C_RUNWAY, lw=0.8, zorder=4))
    ax.plot([x0, x1], [0, 0], color="white", lw=0.8, ls=(0, (6, 4)), zorder=5)   # runway centerline markings
    ax.text(0.5 * (x0 + x1), label_y, "Runway", ha="center", va="top", fontsize=8.5, fontweight="bold", color=C_RUNWAY)


def plot_plan_view(funnel: LpvFunnelGeometry, runway_length: float, out_path: Path) -> None:
    km = 1e-3
    d_garp = funnel.d_garp
    d_fpap = d_garp - 1000.0 * FT_TO_M

    def plan(s: float, sign_y: float) -> tuple[float, float]:
        bx, by, _ = funnel.boundary_point(s, sign_y, 1)
        return project(funnel, bx, by)

    s_all = np.linspace(0.0, funnel.s_max, 200)
    upper = np.array([plan(s, +1) for s in s_all])
    lower = np.array([plan(s, -1) for s in s_all])
    sector = np.linspace(-d_garp, 0.0, 20)
    sec_up = np.array([plan(s, +1) for s in sector])
    sec_lo = np.array([plan(s, -1) for s in sector])

    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    ax.add_patch(Polygon(np.vstack([upper, lower[::-1]]) * [km, 1], closed=True, fc=C_FILL, ec="none"))
    ax.plot(upper[:, 0] * km, upper[:, 1], color=C_FUNNEL, lw=1.6)
    ax.plot(lower[:, 0] * km, lower[:, 1], color=C_FUNNEL, lw=1.6)
    ax.plot(sec_up[:, 0] * km, sec_up[:, 1], "--", color=C_FUNNEL, lw=0.9)
    ax.plot(sec_lo[:, 0] * km, sec_lo[:, 1], "--", color=C_FUNNEL, lw=0.9)

    ax.plot([-d_garp * km, funnel.s_max * km + 0.5], [0, 0], color=C_CENTER, lw=1.0, zorder=3)
    draw_runway(ax, -runway_length * km, 0.0, 0.5 * RUNWAY_WIDTH, label_y=-0.5 * RUNWAY_WIDTH - 20)

    ax.text(
        0.62 * funnel.s_max * km, 0.25 * upper[-1, 1],
        "Lateral protection area\n(LPV course width)",
        ha="center", va="center", fontsize=8.5, color=C_FUNNEL
    )
    ax.annotate(
        "Lateral guidance sector\n(LGSA about the GARP)",
        xy=(-0.5 * d_garp * km, 0.5 * sec_up[10, 1]),
        xytext=(-0.62 * d_garp * km, 260), ha="center", fontsize=8, color=C_FUNNEL,
        arrowprops=dict(arrowstyle="->", color=C_FUNNEL, lw=0.7)
    )
    ax.annotate(
        "Final approach course", xy=(0.82 * funnel.s_max * km, 0),
        xytext=(0.82 * funnel.s_max * km, -170),
        ha="center", fontsize=8, color=C_CENTER, arrowprops=dict(arrowstyle="->", color=C_CENTER, lw=0.7)
    )

    label_point(ax, (-d_garp * km, 0), "GARP", offset=(-5, -4), ha="right", va="top")
    label_point(ax, (-d_fpap * km, 0), "FPAP", offset=(0, 8))
    label_point(ax, (0, 0), "Threshold (LTP)", offset=(4, 8), ha="left")
    label_point(ax, (funnel.s_max * km, 0), "PFAF", offset=(6, 0), ha="left", va="center")

    ax.set_xlim(-d_garp * km - 1.3, funnel.s_max * km + 1.8)
    ax.set_ylim(lower[-1, 1] - 60, upper[-1, 1] + 60)
    ax.set_xlabel("distance from threshold [km]")
    ax.set_ylabel("lateral offset [m]")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_vertical(funnel: LpvFunnelGeometry, h_ltp: float, runway_length: float, out_path: Path) -> None:
    km = 1e-3

    def lower_cut(s: float) -> float:
        return funnel.boundary_point(s, 1, -1)[2]

    def lower_raw(s: float) -> float:
        return funnel.centerline_height(s) - funnel.half_width_vertical(s)

    s_45 = bisect(lambda s: funnel.half_width_vertical(s) > funnel.fsd_v_min + 1e-9, 0.0, 1e6)
    s_150 = bisect(lambda s: funnel.half_width_vertical(s) >= funnel.fsd_v_max - 1e-9, 0.0, 1e6)
    s_cut = bisect(lambda s: lower_cut(s) <= lower_raw(s) + 1e-9, 0.0, funnel.s_max)
    s_ray = min(s_150, funnel.s_max)
    ray_slope = funnel.half_width_vertical(s_ray) / s_ray

    s_all = np.unique(np.concatenate([np.linspace(0.0, funnel.s_max, 400), [s_cut, min(s_45, funnel.s_max), s_ray]]))
    h_c = np.array([funnel.centerline_height(s) for s in s_all])
    up = np.array([funnel.boundary_point(s, 1, 1)[2] for s in s_all])
    lo = np.array([lower_cut(s) for s in s_all])

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    ax.fill_between(s_all * km, lo, up, color=C_FILL, lw=0)
    ax.plot(s_all * km, up, color=C_FUNNEL, lw=1.6)
    ax.plot(s_all * km, lo, color=C_FUNNEL, lw=1.6)
    for sign in (1, -1):
        ax.plot([0, s_ray * km], [funnel.thr_h, funnel.centerline_height(s_ray) + sign * ray_slope * s_ray],
                "--", color=C_FUNNEL, lw=0.7, alpha=0.7)
    ax.plot(s_all * km, h_c, color=C_CENTER, lw=1.5)

    ax.axhline(h_ltp, color="0.35", lw=0.8)
    ax.add_patch(Rectangle((-runway_length * km, h_ltp - 8), runway_length * km, 16, fc="0.55", ec=C_RUNWAY, lw=0.8, zorder=4))
    ax.text(-0.5 * runway_length * km, h_ltp - 16, "Runway", ha="center", va="top", fontsize=8.5, fontweight="bold", color=C_RUNWAY)

    top = up.max() * 1.12
    for sv in (s_45, s_150):
        if sv <= funnel.s_max * 1.2:
            ax.axvline(sv * km, color="0.55", lw=0.6, ls=":")
    ax.text(0.5 * s_45 * km, top, "45 m clamp", ha="center", fontsize=8)
    ax.text(0.5 * (s_45 + s_150) * km, top, "angular region", ha="center", fontsize=8)
    ax.text(s_150 * km + 0.15, top, "150 m clamp", ha="left", fontsize=8)

    # terms
    ax.annotate(
        "Glidepath", xy=(0.55 * funnel.s_max * km, funnel.centerline_height(0.55 * funnel.s_max)),
        xytext=(0.66 * funnel.s_max * km, funnel.centerline_height(0.4 * funnel.s_max) - 60), ha="center",
        fontsize=8.5, color=C_CENTER, arrowprops=dict(arrowstyle="->", color=C_CENTER, lw=0.7)
    )
    ax.annotate(
        "Vertical protection area\n(full-scale deflection)", xy=(0.78 * funnel.s_max * km, up[int(0.78 * len(up))] - 20),
        xytext=(0.36 * funnel.s_max * km, up.max() * 0.92), ha="center", fontsize=8, color=C_FUNNEL,
        arrowprops=dict(arrowstyle="->", color=C_FUNNEL, lw=0.7)
    )
    ax.annotate(
        "Ground floor", xy=(0.5 * s_cut * km, funnel.thr_h - funnel.ground_margin - 2), xytext=(1.6, h_ltp - 110),
        fontsize=8, color=C_GROUND, arrowprops=dict(arrowstyle="->", color=C_GROUND, lw=0.7)
    )

    label_point(ax, (0, funnel.thr_h), "FPCP\n(Threshold + TCH)", offset=(-6, 8), ha="right")
    label_point(ax, (funnel.s_max * km, h_c[-1]), "PFAF", offset=(6, 0), ha="left", va="center")

    ax.set_xlim(-runway_length * km - 0.6, funnel.s_max * km + 1.6)
    ax.set_ylim(h_ltp - 160, top * 1.08)
    ax.set_xlabel("distance from threshold [km]")
    ax.set_ylabel("height above threshold [m]")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    lpv = LpvFunnelGeometry(
        thr_xy=(0.0, 0.0),
        thr_h=H_LTP + TCH,
        faf_xy=(S_FAF, 0.0),
        glideslope_deg=GLIDESLOPE_DEG,
        runway_length_m=RUNWAY_LENGTH,
    )
    out_dir = Path(__file__).parent
    plot_plan_view(lpv, RUNWAY_LENGTH, out_dir / "fig_lpv_plan_view.pdf")
    plot_vertical(lpv, H_LTP, RUNWAY_LENGTH, out_dir / "fig_lpv_vertical.pdf")