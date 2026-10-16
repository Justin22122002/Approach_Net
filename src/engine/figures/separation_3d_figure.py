from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils_project.separation_geometry import SeparationGeometry

NM_TO_M = 1852.0
FT_TO_M = 0.3048

N_EXP = 4.0                     # superellipsoid exponent
CUT = (0.0, 0.5 * np.pi)        # azimuth range removed from the ellipsoid (cut-away towards the viewer)
VIEW_ELEV, VIEW_AZIM = 24, 45

C_ELLIP, C_ELLIP_FILL, C_CYL, C_AC, C_DIM = "#c25a00", "#f4b183", "#1b4f8a", "#222222", "#4a4a4a"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "serif",
    "font.size": 9,
})


def superellipsoid_surface(sep: SeparationGeometry, theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, ...]:
    """parametric surface of the (inflated) superellipsoid, horizontal in NM, vertical in ft"""
    e = 2.0 / sep.n
    a, c = sep.d_h_eff / NM_TO_M, sep.d_v_eff / FT_TO_M
    T, P = np.meshgrid(theta, phi)
    cp, sp = np.cos(P), np.sin(P)
    r = a * np.sign(cp) * np.abs(cp) ** e
    z = c * np.sign(sp) * np.abs(sp) ** e
    return r * np.cos(T), r * np.sin(T), z


def superellipse_profile(sep: SeparationGeometry, n_pts: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """meridian (r >= 0) of the superellipsoid: r [NM], z [ft]"""
    e = 2.0 / sep.n
    a, c = sep.d_h_eff / NM_TO_M, sep.d_v_eff / FT_TO_M
    phi = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_pts)
    return a * np.abs(np.cos(phi)) ** e, c * np.sign(np.sin(phi)) * np.abs(np.sin(phi)) ** e


def draw_cylinder(ax, sep: SeparationGeometry) -> None:
    """cylinder edges, same quarter cut away as the ellipsoid"""
    a, c = sep.d_h / NM_TO_M, sep.d_v / FT_TO_M
    t = np.linspace(CUT[1], CUT[0] + 2 * np.pi, 200)
    for z, ls in ((c, "-"), (-c, "-"), (0.0, ":")):
        ax.plot(a * np.cos(t), a * np.sin(t), np.full_like(t, z), color=C_CYL, lw=1.2, ls=ls, zorder=5)
    for ang in np.linspace(CUT[1], CUT[0] + 2 * np.pi, 4)[1:-1]:
        ax.plot([a * np.cos(ang)] * 2, [a * np.sin(ang)] * 2, [-c, c], color=C_CYL, lw=0.8, zorder=5)


def draw_cut_planes(ax, sep: SeparationGeometry) -> None:
    """superellipse + cylinder rectangle in the two cut planes -> shows the cross-section"""
    r, z = superellipse_profile(sep)
    a, c = sep.d_h / NM_TO_M, sep.d_v / FT_TO_M
    for ang in CUT:
        ca, sa = np.cos(ang), np.sin(ang)
        verts = [list(zip(r * ca, r * sa, z))]
        ax.add_collection3d(Poly3DCollection(verts, fc=C_ELLIP_FILL, ec="none", alpha=0.55, zorder=1))
        ax.plot(r * ca, r * sa, z, color=C_ELLIP, lw=1.6, zorder=6)
        rect_r = np.array([0, a, a, 0])
        rect_z = np.array([c, c, -c, -c])
        ax.plot(rect_r * ca, rect_r * sa, rect_z, color=C_CYL, lw=1.2, zorder=7)


def annotate3d(ax, xyz: tuple[float, float, float], text: str, offset: tuple[float, float],
               color: str, ha: str = "center", bold: bool = False) -> None:
    """2D annotation with arrow, anchored at a projected 3D point -> no overlap with 3D depth sorting"""
    x2, y2, _ = proj3d.proj_transform(*xyz, ax.get_proj())
    ax.annotate(text, (x2, y2), xytext=offset, textcoords="offset points", ha=ha, va="center",
                fontsize=8, fontweight="bold" if bold else "normal", color=color, zorder=20,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.7, shrinkA=2, shrinkB=3))


def plot_separation_3d(sep: SeparationGeometry, out_path: Path) -> None:
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)

    # superellipsoid with a quarter cut away
    theta = np.linspace(CUT[1], CUT[0] + 2 * np.pi, 120)
    phi = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 80)
    X, Y, Z = superellipsoid_surface(sep, theta, phi)
    ax.plot_surface(X, Y, Z, color=C_ELLIP_FILL, alpha=0.35, lw=0, antialiased=True, shade=True, zorder=0)
    ax.plot_wireframe(X, Y, Z, rstride=16, cstride=20, color=C_ELLIP, lw=0.35, alpha=0.6, zorder=2)

    draw_cut_planes(ax, sep)
    draw_cylinder(ax, sep)

    a, c = sep.d_h / NM_TO_M, sep.d_v / FT_TO_M
    a_eff, c_eff = sep.d_h_eff / NM_TO_M, sep.d_v_eff / FT_TO_M

    # aircraft + dimension lines in the first cut plane (theta = 0 -> x axis)
    ax.scatter([0], [0], [0], color=C_AC, s=22, depthshade=False, zorder=10)
    ax.plot([0, a], [0, 0], [0, 0], color=C_DIM, lw=0.9, zorder=8)
    ax.plot([a, a], [0, 0], [0, c], color=C_DIM, lw=0.9, zorder=8)
    if sep.contain_cylinder:
        ax.scatter([a], [0], [c], color=C_ELLIP, s=18, depthshade=False, zorder=10)

    # legend proxies
    ax.plot([], [], color=C_ELLIP, lw=1.6, label=f"superellipsoid ($n$ = {sep.n:g}), "
                                                   f"$kD_h$ = {a_eff:.2f} NM, $kD_v$ = {c_eff:.0f} ft")
    ax.plot([], [], color=C_CYL, lw=1.2, label="FAA cylinder (radar OR vertical minimum)")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)

    lim_h = 1.08 * a_eff
    ax.set_xlim(-lim_h, lim_h)
    ax.set_ylim(-lim_h, lim_h)
    ax.set_zlim(-1.08 * c_eff, 1.08 * c_eff)
    ax.set_box_aspect((1, 1, 0.55))
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    ax.set_xlabel(r"$\Delta x$ [NM]", labelpad=2)
    ax.set_ylabel(r"$\Delta y$ [NM]", labelpad=2)
    ax.set_zlabel(r"$\Delta h$ [ft]", labelpad=4)
    ax.tick_params(labelsize=7.5, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("0.85")
        axis._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 1)

    # 2D annotations anchored at projected 3D points (after view/limits are fixed)
    annotate3d(ax, (0, 0, 0), "own aircraft", offset=(18, 22), color=C_AC, bold=True)
    annotate3d(ax, (0.5 * a, 0, 0), f"$D_h$ = {a:.1f} NM", offset=(-38, -26), color=C_CYL)
    annotate3d(ax, (a, 0, 0.5 * c), f"$D_v$ = {c:.0f} ft", offset=(-42, -4), color=C_CYL, ha="right")
    if sep.contain_cylinder:
        annotate3d(ax, (a, 0, c), f"cylinder corner on surface\n$k = 2^{{1/n}}$ = {2 ** (1 / sep.n):.3f}",
                   offset=(-30, 80), color=C_ELLIP, ha="right")

    fig.subplots_adjust(left=0.0, right=0.95, bottom=0.04, top=1.0)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    sep = SeparationGeometry(
        d_horizontal_m=2.5 * NM_TO_M,
        d_vertical_m=1000.0 * FT_TO_M,
        n=N_EXP,
        contain_cylinder=True,
    )
    out_dir = Path(__file__).parent
    plot_separation_3d(sep, out_dir / "fig_separation_3d.pdf")