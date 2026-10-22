import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pygame
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from PIL import Image
from OpenGL.GL import (
    glReadPixels, glReadBuffer, glPixelStorei, glGetIntegerv, glDeleteVertexArrays, glDeleteBuffers,
    glViewport, glClear,
    GL_BACK, GL_RGB, GL_UNSIGNED_BYTE, GL_PACK_ALIGNMENT, GL_VIEWPORT, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
)
from pyproj import Transformer

from utils.airport import Airport
from utils.arinc_runway import ArincRunway
from utils.clients.airport_client import OpenStreetMapAirportClient
from utils.clients.osm_water_client import OpenStreetMapWaterClient

from engine.utils.engine import Engine
from engine.utils.color import Color
from engine.utils.camera import FlyCamera, OrbitCamera
from engine.utils.lineRenderer import LineRenderer
from engine.utils.pointRenderer import PointRenderer
from engine.utils.sky import Sky
from engine.utils.texture import Texture
from engine.utils.vec3 import Vec3
from engine.utils_project.building_project_mesh import BuildingProjectMesh
from engine.utils_project.geo_transformer import GeoTransformer
from engine.utils_project.runway_project_mesh import RunwayProjectMesh
from engine.utils_project.taxiway_project_mesh import TaxiwayProjectMesh
from engine.utils_project.terminal_project_mesh import TerminalProjectMesh
from engine.utils_project.terrain_project_mesh import TerrainProjectMesh
from engine.utils_project.road_project_mesh import RoadProjectMesh
from engine.utils_project.separation_geometry import SeparationGeometry
from engine.utils_project.water_texture_renderer import WaterTextureRenderer

from dotenv import load_dotenv

load_dotenv()

ICAO = "CYVR"
WINDOW_W = 900
WINDOW_H = 900
SCREENSHOT_SQUARE: bool = True
LOCAL_SCALE = 0.01
USE_SATELLITE_TEXTURE = False
TERRAIN_TEX_SIZE = 8192
PREFERRED_BEARING_DEG = -90.0

DATA_ROOT = Path("./data")
OUTPUT_FOLDER = "multiple_airports_ft_transformer_gnn_rollout_constraint_training"
RUN_ID = "2026-10-11_17-13-54"
CONSTRAINED = True
TAG = f"gnn{'_constrained' if CONSTRAINED else ''}_rollout_{RUN_ID}"

SUCCESS_S: float = 10.0
CLOSE_METRIC: str = "3d"
GRID_S: float = 10.0
MIN_CONCURRENT: int = 3
REQUIRE_ALL_SUCCESS: bool = True
CLEAN_MARGIN_S: float = 10 * 60.0
SCENE_SEPARATION_S: float = 30 * 60.0
TOP_K: int = 10
MAX_FLIGHTS: int = 15

PRE_S: float | None = 10 * 60.0

N_MARKERS: int = 5
MARKER_SIZE: float = 5000.0

GRADIENT_STOPS: list[tuple[float, tuple[float, float, float]]] = [
    (0.00, (0.24, 0.10, 0.42)),   # deep purple
    (0.35, (0.55, 0.18, 0.47)),   # plum
    (0.70, (0.80, 0.30, 0.30)),   # brick red
    (1.00, (0.93, 0.60, 0.25)),   # ochre
]
HIDDEN = Vec3(0.0, -1.0e5, 0.0)
FALLBACK_COLOR = Color(1.0, 1.0, 1.0)

SCREENSHOT_DIR = DATA_ROOT / ICAO / "screenshots"

CLOSE_DT_S: float = 1.0
CLOSE_MIN_AGL_M: float = 300.0
CLOSE_VIEW_MARGIN: float = 0.8
CLOSE_PITCH_DEG: float = 89.0
CLOSE_YAW_DEG: float = -90.0
CLOSE_POINT_SIZE: float = 140.0
CLOSE_MARKER_SIZE: float = 100.0
CLOSE_CYL_COLOR = Color(0.2, 0.9, 0.4)
CLOSE_ELLIP_COLOR = Color(1.0, 0.9, 0.2)
CLOSE_LINK_COLOR = Color(1.0, 1.0, 1.0)

# close-up layering: rendered as separate passes and composited in this order (bottom -> top),
# so the active pair is always fully in front, independent of depth / engine alpha support
CLOSE_INACTIVE_COLOR = Color(0.80, 0.80, 0.80)   # other trajectories of the scene
CLOSE_INACTIVE_ALPHA: float = 0.30
CLOSE_INACTIVE_WIDTH: float = 2.0
CLOSE_SEP_ALPHA: float = 0.80                     # separation cylinders / ellipsoids
CLOSE_ACTIVE_WIDTH: float = 4.0                   # the two trajectories of the closest pair
CLOSE_DIFF_THRESHOLD: int = 4                     # pixel difference that counts as "drawn by this layer"

SHOW_CLOSE_RING: bool = True
CLOSE_RING_COLOR = Color(1.0, 1.0, 1.0)
CLOSE_RING_WIDTH: float = 3.0
CLOSE_RING_SEGMENTS: int = 96
CLOSE_FOV_FALLBACK_DEG: float = 60.0
CLOSE_ROUND_IMAGE: bool = True
CLOSE_ROUND_BORDER_PX: int = 6

LEGEND_CLOSE_COLOR: str = "black"
LEGEND_CLOSE_WIDTH: float = 2.2

SEP_FULL = SeparationGeometry()
SEP_HALF = SeparationGeometry(
    d_horizontal_m=SEP_FULL.d_h / 2.0,
    d_vertical_m=SEP_FULL.d_v / 2.0,
    n=SEP_FULL.n,
    contain_cylinder=SEP_FULL.contain_cylinder,
)

LEGEND_FORMATS: tuple[str, ...] = ("png", "pdf")
LEGEND_SIZE_IN: tuple[float, float] = (5.0, 1.2)
LEGEND_DPI: int = 300
LEGEND_TRANSPARENT: bool = True
LEGEND_LABEL: str = "simulation time (UTC)"


@dataclass
class SnapFlight:
    icao24: str
    flight_id: int
    ok: bool                       # success at SUCCESS_S
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    h: np.ndarray

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    def position(self, t: float) -> tuple[float, float, float]:
        return float(np.interp(t, self.t, self.x)), float(np.interp(t, self.t, self.y)), float(np.interp(t, self.t, self.h))


@dataclass
class SnapScene:
    t_snap: float
    flights: list[SnapFlight]
    objects: list = field(default_factory=list)   # (game object, position) -> parked when switching
    lines: list = field(default_factory=list)     # (line renderer, points, colors)
    color_t0: float = 0.0
    color_t1: float = 1.0
    marker_times: list[float] = field(default_factory=list)
    closest: dict | None = None                   # cached result of closest_pair()


def load_success(eval_path: Path, s: float) -> dict[tuple[str, int], bool]:
    if not eval_path.exists():
        raise RuntimeError(f"{eval_path} not found -> run trajectory_evaluator_all.py first")
    ev = pd.read_csv(eval_path)
    ok = ev["converged"].astype(bool) & (ev["s_needed"].astype(float) <= s)
    return {(str(i), int(f)): bool(o) for i, f, o in zip(ev["icao24"], ev["flight_id"], ok)}


def load_flights(path: Path, eval_path: Path) -> list[SnapFlight]:
    df = pd.read_parquet(path)
    if "t_sim" not in df.columns:
        raise RuntimeError("no t_sim column -> rerun the rollout with the current multi_rollout_evaluator.py")
    success = load_success(eval_path, SUCCESS_S)

    flights: list[SnapFlight] = []
    for (icao24, flight_id), g in df.groupby(["icao24", "flight_id"], sort=False):
        g = g.sort_values("step")
        t = g["t_sim"].to_numpy(dtype=float).copy()
        dt = np.nanmedian(np.diff(t)) if np.isfinite(t).sum() > 1 else 1.0
        for i in range(1, len(t)):
            if not np.isfinite(t[i]):
                t[i] = t[i - 1] + dt
        if not np.isfinite(t).all() or len(t) < 2:
            continue
        flights.append(SnapFlight(
            icao24=str(icao24), flight_id=int(flight_id),
            ok=success.get((str(icao24), int(flight_id)), False),   # not evaluated -> not a success
            t=t, x=g["x"].to_numpy(float), y=g["y"].to_numpy(float), h=g["h"].to_numpy(float),
        ))
    n_ok = sum(f.ok for f in flights)
    print(f"[snapshot] {len(flights)} flights loaded, {n_ok} successful at s={SUCCESS_S:g}")
    return flights


def airborne_count(starts: np.ndarray, ends: np.ndarray, grid: np.ndarray) -> np.ndarray:
    s, e = np.sort(starts), np.sort(ends)
    return np.searchsorted(s, grid, side="right") - np.searchsorted(e, grid, side="left")


def rank_scenes(flights: list[SnapFlight]) -> list[SnapScene]:
    ok = [f for f in flights if f.ok]
    bad = [f for f in flights if not f.ok]
    if not ok:
        raise RuntimeError(f"no successful flights at s={SUCCESS_S:g} in this rollout")

    t_lo = min(f.t_start for f in flights)
    t_hi = max(f.t_end for f in flights)
    grid = np.arange(t_lo, t_hi + GRID_S, GRID_S)

    n_ok = airborne_count(np.array([f.t_start for f in ok]), np.array([f.t_end for f in ok]), grid)
    if REQUIRE_ALL_SUCCESS and bad:
        n_bad = airborne_count(np.array([f.t_start for f in bad]), np.array([f.t_end for f in bad]), grid)
        m = int(round(CLEAN_MARGIN_S / GRID_S))
        bad_near = np.convolve(n_bad, np.ones(2 * m + 1), mode="same") > 0
        n_ok = np.where(bad_near, 0, n_ok)

    scenes: list[SnapScene] = []
    for i in np.argsort(-n_ok, kind="stable"):
        c = int(n_ok[i])
        if c < MIN_CONCURRENT:
            break
        lo, hi = i, i
        while lo > 0 and n_ok[lo - 1] == c:
            lo -= 1
        while hi < len(n_ok) - 1 and n_ok[hi + 1] == c:
            hi += 1
        t_snap = float(grid[(lo + hi) // 2])
        if any(abs(t_snap - s.t_snap) < SCENE_SEPARATION_S for s in scenes):
            continue

        sel = [f for f in ok if f.t_start <= t_snap <= f.t_end]
        sel.sort(key=lambda f: f.t_end)
        if len(sel) > MAX_FLIGHTS:
            sel = sel[:MAX_FLIGHTS]
        scenes.append(SnapScene(t_snap=t_snap, flights=sel))
        if len(scenes) >= TOP_K:
            break

    if not scenes:
        raise RuntimeError(f"no scene with >= {MIN_CONCURRENT} simultaneous successful flights "
                           f"(try lower MIN_CONCURRENT or REQUIRE_ALL_SUCCESS=False)")
    for k, s in enumerate(scenes):
        print(f"  scene {k}: {fmt_time(s.t_snap)}  {len(s.flights)} aircraft simultaneously airborne")
    return scenes


def fmt_time(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def gradient(u: float) -> tuple[float, float, float]:
    u = float(np.clip(u, 0.0, 1.0))
    for (u0, c0), (u1, c1) in zip(GRADIENT_STOPS[:-1], GRADIENT_STOPS[1:]):
        if u <= u1:
            a = (u - u0) / max(u1 - u0, 1e-9)
            return tuple(c0[i] + a * (c1[i] - c0[i]) for i in range(3))
    return GRADIENT_STOPS[-1][1]


def marker_fractions() -> np.ndarray:
    return np.linspace(0.0, 1.0, N_MARKERS)


def set_lines(lr: LineRenderer, points: list[Vec3], colors: list[Color]) -> None:
    if len(points) < 2:
        points, colors = [HIDDEN, HIDDEN], [FALLBACK_COLOR, FALLBACK_COLOR]
    seg_pts: list[Vec3] = []
    seg_cols: list[Color] = []
    for i in range(len(points) - 1):
        seg_pts += [points[i], points[i + 1]]
        seg_cols += [colors[i], colors[i + 1]]
    lr.points, lr.colors = seg_pts, seg_cols
    if lr._vao is not None:
        glDeleteVertexArrays(1, [lr._vao])
        glDeleteBuffers(1, [lr._vbo])
        lr._vao = None
        lr._vbo = None


def set_points(pr: PointRenderer, points: list[Vec3], colors: list[Color]) -> None:
    pr.points, pr.colors = points, colors
    if pr._vao is not None:
        glDeleteVertexArrays(1, [pr._vao])
        glDeleteBuffers(1, [pr._vbo])
        pr._vao = None
        pr._vbo = None


def save_screenshot(path: Path) -> None:
    x, y, w, h = (int(v) for v in glGetIntegerv(GL_VIEWPORT))
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    glReadBuffer(GL_BACK)
    data = glReadPixels(x, y, w, h, GL_RGB, GL_UNSIGNED_BYTE)
    img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)[::-1]
    if SCREENSHOT_SQUARE and w != h:
        side = min(w, h)
        y0, x0 = (h - side) // 2, (w - side) // 2
        img = img[y0:y0 + side, x0:x0 + side]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    print(f"[snapshot] saved {path}")


def save_legend(base: Path, t0: float, t1: float, marker_times: list[float], t_close: float | None = None) -> None:
    cmap = LinearSegmentedColormap.from_list("snapshot_time", [(u, c) for u, c in GRADIENT_STOPS])
    to_num = lambda t: mdates.date2num(datetime.fromtimestamp(t, tz=timezone.utc))

    fig = Figure(figsize=LEGEND_SIZE_IN, dpi=LEGEND_DPI)
    cax = fig.add_axes((0.04, 0.32, 0.92, 0.22))
    cb = fig.colorbar(ScalarMappable(norm=Normalize(to_num(t0), to_num(t1)), cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.outline.set_linewidth(0.6)

    xs = [to_num(t) for t in marker_times]
    cols = [gradient(u) for u in marker_fractions()]

    max_ = fig.add_axes((0.04, 0.58, 0.92, 0.22))
    max_.set_xlim(to_num(t0), to_num(t1))
    max_.set_ylim(0.0, 1.0)
    max_.scatter(xs, [0.5] * len(xs), s=90, c=cols, edgecolors="#222222", linewidths=0.8, clip_on=False)
    max_.axis("off")

    if t_close is not None:
        cax.axvline(to_num(t_close), color=LEGEND_CLOSE_COLOR, linewidth=LEGEND_CLOSE_WIDTH)

    cax.set_xticks(xs)
    cax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    cax.tick_params(labelsize=8, width=0.6, length=3)
    cb.set_label(LEGEND_LABEL, fontsize=8)

    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in LEGEND_FORMATS:
        out = base.parent / f"{base.name}_legend.{fmt}"
        fig.savefig(out, transparent=LEGEND_TRANSPARENT, bbox_inches="tight", pad_inches=0.03)
        print(f"[snapshot] saved {out}")


basePath: Path = DATA_ROOT
airport = Airport.load_from_folder(basePath, ICAO)

crs_to: str = GeoTransformer.utm_epsg(airport.lat, airport.lon)
to_utm: Transformer = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)

runways: list[ArincRunway] = airport.arinc_424_runway

horizontal_scale = LOCAL_SCALE
vertical_scale = LOCAL_SCALE

geo = GeoTransformer(
    lat=airport.lat,
    lon=airport.lon,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    transformer=to_utm,
)

engine = Engine(WINDOW_W, WINDOW_H, f"Rollout snapshot - {ICAO} - {TAG}")
scene = engine.scene


def _runway_group_key(identifier: str) -> frozenset:
    m = re.match(r"^(\d{1,2})([LRC]?)$", identifier.upper())
    n, suf = int(m.group(1)), m.group(2)
    n_recip = n + 18 if n <= 18 else n - 18
    suf_recip = {"L": "R", "R": "L", "C": "C", "": ""}[suf]
    return frozenset({(n, suf), (n_recip, suf_recip)})


def _bearing_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


groups: dict[frozenset, list[ArincRunway]] = {}
for runway in runways:
    groups.setdefault(_runway_group_key(runway.runway_identifier), []).append(runway)

runways_one_per_direction: list[ArincRunway] = [
    min(group, key=lambda r: _bearing_diff(r.bearing, PREFERRED_BEARING_DEG))
    for group in groups.values()
]

for runway in runways_one_per_direction:
    runway.runway_altitude += 10
    RunwayProjectMesh(
        runway=runway,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        transformer=to_utm,
    ).spawn(scene, geo)

for taxiway in airport.taxiways:
    if len(taxiway.coordinates) < 2:
        continue
    TaxiwayProjectMesh(
        coordinates=taxiway.coordinates,
        airport_lat=airport.lat,
        airport_lon=airport.lon,
        airport_altitude_ft=runways[0].runway_altitude if runways else 0.0,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        transformer=to_utm,
        use_mesh=True,
        taxiway_width_m=23.0,
    ).spawn(scene, geo)

map_client = OpenStreetMapAirportClient()

water_tex_path = basePath / ICAO / ("terrain_water_sat.png" if USE_SATELLITE_TEXTURE else "terrain_water.png")

if water_tex_path.exists():
    print(f"[TERRAIN] using cached water texture: {water_tex_path}")
else:
    print("[TERRAIN] fetching OSM water features + rendering texture ...")
    water_features = OpenStreetMapWaterClient().get_water_features(
        lat_min=airport.terrain.lat_min,
        lon_min=airport.terrain.lon_min,
        lat_max=airport.terrain.lat_max,
        lon_max=airport.terrain.lon_max,
    )
    water_renderer = WaterTextureRenderer(
        lat_min=airport.terrain.lat_min,
        lon_min=airport.terrain.lon_min,
        lat_max=airport.terrain.lat_max,
        lon_max=airport.terrain.lon_max,
        width=TERRAIN_TEX_SIZE,
        height=TERRAIN_TEX_SIZE,
    )
    base_rgb: np.ndarray | None = None
    if USE_SATELLITE_TEXTURE:
        base_rgb = map_client.terrain_client.get_satellite_image(
            lat_min=airport.terrain.lat_min,
            lon_min=airport.terrain.lon_min,
            lat_max=airport.terrain.lat_max,
            lon_max=airport.terrain.lon_max,
            width=TERRAIN_TEX_SIZE,
            height=TERRAIN_TEX_SIZE,
        )
    water_rgb = water_renderer.composite(water_features, base_rgb=base_rgb, land_color=(102, 140, 77))
    Image.fromarray(water_rgb).save(water_tex_path)
    print(f"[TERRAIN] saved {water_tex_path}")

terrain_texture = Texture(str(water_tex_path))

terrain_project_mesh = TerrainProjectMesh(
    grid=airport.terrain.elevation_grid,
    lat_min=airport.terrain.lat_min,
    lon_min=airport.terrain.lon_min,
    lat_max=airport.terrain.lat_max,
    lon_max=airport.terrain.lon_max,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    transformer=to_utm,
    texture=terrain_texture,
)
terrain_project_mesh.spawn(scene=scene, geo=geo)

TerminalProjectMesh(
    terminals=airport.terminals,
    terrain=terrain_project_mesh,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    transformer=to_utm,
).spawn(scene, geo)

buildings, roads = map_client.get_buildings_and_roads(
    lat_min=airport.terrain.lat_min, lon_min=airport.terrain.lon_min,
    lat_max=airport.terrain.lat_max, lon_max=airport.terrain.lon_max,
)

RoadProjectMesh(
    roads=roads,
    terrain=terrain_project_mesh,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    transformer=to_utm,
).spawn(scene, geo)

BuildingProjectMesh(
    buildings=buildings,
    terrain=terrain_project_mesh,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    transformer=to_utm,
).spawn(scene, geo)

scene.ambient_light = Color(0.25, 0.25, 0.3)

engine.sky = Sky(
    sun_elevation=30.0,
    sun_azimuth=45.0,
    cloud_density=1.0,
    cloud_speed=0.012,
    cloud_altitude=400.0,
    zenith_color=(0.07, 0.22, 0.68),
    horizon_color=(0.52, 0.75, 0.94),
)
scene.directional_light = engine.sky.directional_light

folder = DATA_ROOT / ICAO / OUTPUT_FOLDER
all_flights = load_flights(folder / f"{TAG}.parquet", folder / f"{TAG}_evaluation.csv")
scenes = rank_scenes(all_flights)


def to_local(x: float, y: float, h: float) -> Vec3:
    xg, zg = geo.to_local_in_m(x, y)
    return Vec3(xg, geo.altitude_to_local(h), zg)


def drawn_indices(f: SnapFlight, s: SnapScene) -> np.ndarray:
    t_from = f.t_start if PRE_S is None else max(f.t_start, s.t_snap - PRE_S)
    return np.flatnonzero(f.t >= t_from)


def time_color(s: SnapScene, t: float) -> Color:
    r, g, b = gradient((t - s.color_t0) / max(s.color_t1 - s.color_t0, 1e-6))
    return Color(r, g, b)


def close_fov_deg() -> float:
    for name in ("fov", "fov_deg", "fovy", "fov_y"):
        v = getattr(closeup_camera, name, None)
        if isinstance(v, (int, float)) and 0.0 < v < 180.0:
            return float(v)
    return CLOSE_FOV_FALLBACK_DEG


def closeup_view(c: dict) -> dict:
    (ax_, ay_, ah_), (bx_, by_, bh_) = c["a"].position(c["t"]), c["b"].position(c["t"])
    pa, pb = to_local(ax_, ay_, ah_), to_local(bx_, by_, bh_)
    extent = float(np.hypot(pa.x - pb.x, pa.z - pb.z)) + 2.0 * SEP_HALF.d_h * LOCAL_SCALE
    distance = CLOSE_VIEW_MARGIN * extent
    half_width_local = distance * np.tan(np.radians(close_fov_deg()) / 2.0)
    return {
        "target": Vec3((pa.x + pb.x) / 2, (pa.y + pb.y) / 2, (pa.z + pb.z) / 2),
        "distance": distance,
        "center_xy": ((ax_ + bx_) / 2.0, (ay_ + by_) / 2.0),
        "h": (ah_ + bh_) / 2.0,
        "radius_m": float(half_width_local / LOCAL_SCALE),   # circle inscribed in the square close-up image
    }


def ring_points(center_xy: tuple[float, float], h: float, radius_m: float) -> list[Vec3]:
    a = np.linspace(0.0, 2.0 * np.pi, CLOSE_RING_SEGMENTS + 1)
    return [to_local(center_xy[0] + radius_m * np.cos(t), center_xy[1] + radius_m * np.sin(t), h) for t in a]


def show_scene(k: int) -> None:
    s = scenes[k]
    if s.objects or s.lines:
        for go, pos in s.objects:
            go.transform.position = pos
        for lr, pts, cols in s.lines:
            set_lines(lr, pts, cols)
        return

    segs = [(f, drawn_indices(f, s)) for f in s.flights]
    s.color_t0 = min(float(f.t[idx[0]]) for f, idx in segs)
    s.color_t1 = max(f.t_end for f, _ in segs)
    s.marker_times = [s.color_t0 + u * (s.color_t1 - s.color_t0) for u in marker_fractions()]

    marker_pts: list[Vec3] = []
    marker_cols: list[Color] = []

    for f, idx in segs:
        pts = [to_local(f.x[i], f.y[i], f.h[i]) for i in idx]
        cols = [time_color(s, f.t[i]) for i in idx]
        if len(pts) < 2:
            continue
        name = f"s{k}_{f.icao24}_{f.flight_id}"

        lr = LineRenderer([pts[0], pts[1]], color=cols[0])
        set_lines(lr, pts, cols)
        line_go = scene.spawn(f"traj_{name}")
        line_go.add_component(lr)
        s.lines.append((lr, pts, cols))

        t_first, t_last = float(f.t[idx[0]]), float(f.t[idx[-1]])
        for t_m, u in zip(s.marker_times, marker_fractions()):
            if t_first <= t_m <= t_last:
                marker_pts.append(to_local(*f.position(t_m)))
                marker_cols.append(Color(*gradient(u)))

    if marker_pts:
        marker_go = scene.spawn(f"markers_s{k}")
        marker_go.add_component(PointRenderer(points=marker_pts, colors=marker_cols, size=MARKER_SIZE))
        marker_go.transform.position = Vec3(0.0, 0.0, 0.0)
        s.objects.append((marker_go, Vec3(0.0, 0.0, 0.0)))

    if s.closest is None:
        s.closest = closest_pair(s)
    c = s.closest
    if SHOW_CLOSE_RING and c is not None:
        view = closeup_view(c)
        pts = ring_points(view["center_xy"], view["h"], view["radius_m"])
        cols = [CLOSE_RING_COLOR] * len(pts)
        lr = LineRenderer.strip([HIDDEN, HIDDEN], color=CLOSE_RING_COLOR, line_width=CLOSE_RING_WIDTH)
        set_lines(lr, pts, cols)
        scene.spawn(f"close_ring_s{k}").add_component(lr)
        s.lines.append((lr, pts, cols))


def hide_scene(k: int) -> None:
    s = scenes[k]
    for go, _ in s.objects:
        go.transform.position = HIDDEN
    for lr, _, _ in s.lines:
        set_lines(lr, [], [])


AIRPORT_ELEV_M = airport.get_mean_runway_elevation_ft() * 0.3048

def closest_pair(s: SnapScene) -> dict | None:
    t_from = -np.inf if PRE_S is None else s.t_snap - PRE_S
    best: dict | None = None
    for a, b in combinations(s.flights, 2):
        lo, hi = max(a.t_start, b.t_start, t_from), min(a.t_end, b.t_end)
        if hi <= lo:
            continue
        tt = np.arange(lo, hi + 1e-6, CLOSE_DT_S)
        ax, ay, ah = (np.interp(tt, a.t, v) for v in (a.x, a.y, a.h))
        bx, by, bh = (np.interp(tt, b.t, v) for v in (b.x, b.y, b.h))
        d_h = np.hypot(ax - bx, ay - by)
        d_v = np.abs(ah - bh)
        if CLOSE_METRIC == "3d":
            metric = np.sqrt(d_h ** 2 + d_v ** 2)
        elif CLOSE_METRIC == "horizontal":
            metric = d_h.copy()
        else:
            metric = np.sqrt((d_h / SEP_FULL.d_h) ** 2 + (d_v / SEP_FULL.d_v) ** 2)
        low = (np.minimum(ah, bh) - AIRPORT_ELEV_M) < CLOSE_MIN_AGL_M
        metric[low] = np.inf
        i = int(np.argmin(metric))
        if np.isfinite(metric[i]) and (best is None or metric[i] < best["metric"]):
            best = {"t": float(tt[i]), "a": a, "b": b, "metric": float(metric[i]),
                    "d_h_m": float(d_h[i]), "d_v_m": float(d_v[i])}
    if best is not None:
        print(f"[close] {best['a'].icao24} / {best['b'].icao24} at {fmt_time(best['t'])}: "
              f"d_h={best['d_h_m'] / 1852:.2f} NM  d_v={best['d_v_m'] / 0.3048:.0f} ft  "
              f"(normalized {best['metric']:.2f}, <1 = inside separation minimum)")
    return best

_n_cyl = len(SEP_HALF.cylinder_edges((0.0, 0.0, 0.0)))
_n_ell = len(SEP_HALF.ellipsoid_rings((0.0, 0.0, 0.0)))

def _hidden_line(name: str, color: Color, width: float | None = None) -> LineRenderer:
    lr = (LineRenderer.strip([HIDDEN, HIDDEN], color=color) if width is None
          else LineRenderer.strip([HIDDEN, HIDDEN], color=color, line_width=width))
    scene.spawn(name).add_component(lr)
    return lr

close_cyl = [[_hidden_line(f"close_cyl_{k}_{i}", CLOSE_CYL_COLOR) for i in range(_n_cyl)] for k in range(2)]
close_ell = [[_hidden_line(f"close_ell_{k}_{i}", CLOSE_ELLIP_COLOR) for i in range(_n_ell)] for k in range(2)]
close_trail = [_hidden_line(f"close_trail_{k}", FALLBACK_COLOR, CLOSE_ACTIVE_WIDTH) for k in range(2)]
close_inactive: list[LineRenderer] = []   # grows on demand: other trajectories of the scene
close_link = _hidden_line("close_link", CLOSE_LINK_COLOR)
close_points = PointRenderer(points=[HIDDEN, HIDDEN], color=CLOSE_LINK_COLOR, size=CLOSE_POINT_SIZE)
scene.spawn("close_points").add_component(close_points)
close_markers = PointRenderer(points=[HIDDEN], color=CLOSE_LINK_COLOR, size=max(CLOSE_MARKER_SIZE, 1.0))
scene.spawn("close_markers").add_component(close_markers)

closeup_camera = OrbitCamera(distance=1.0)

def show_close(c: dict | None, s: SnapScene, layers: tuple[str, ...] = ("inactive", "sep", "active")) -> None:
    for lr in [*close_cyl[0], *close_cyl[1], *close_ell[0], *close_ell[1], *close_trail, *close_inactive, close_link]:
        set_lines(lr, [], [])
    set_points(close_points, [HIDDEN, HIDDEN], [FALLBACK_COLOR] * 2)
    set_points(close_markers, [HIDDEN], [FALLBACK_COLOR])
    if c is None:
        return

    t = c["t"]
    pair = (c["a"], c["b"])
    centers = [c["a"].position(t), c["b"].position(t)]

    if "inactive" in layers:
        others = [f for f in s.flights if f is not c["a"] and f is not c["b"]]
        while len(close_inactive) < len(others):
            close_inactive.append(_hidden_line(f"close_inactive_{len(close_inactive)}",
                                               CLOSE_INACTIVE_COLOR, CLOSE_INACTIVE_WIDTH))
        for lr, f in zip(close_inactive, others):
            idx = drawn_indices(f, s)
            pts = [to_local(f.x[i], f.y[i], f.h[i]) for i in idx]
            set_lines(lr, pts, [CLOSE_INACTIVE_COLOR] * len(pts))

    if "sep" in layers:
        for k, center in enumerate(centers):
            for lr, edge in zip(close_cyl[k], SEP_HALF.cylinder_edges(center)):
                pts = [to_local(*p) for p in edge]
                set_lines(lr, pts, [CLOSE_CYL_COLOR] * len(pts))
            for lr, ring in zip(close_ell[k], SEP_HALF.ellipsoid_rings(center)):
                pts = [to_local(*p) for p in ring]
                set_lines(lr, pts, [CLOSE_ELLIP_COLOR] * len(pts))

    if "active" in layers:
        marker_pts: list[Vec3] = []
        marker_cols: list[Color] = []
        for k, f in enumerate(pair):
            idx = drawn_indices(f, s)
            pts = [to_local(f.x[i], f.y[i], f.h[i]) for i in idx]
            set_lines(close_trail[k], pts, [time_color(s, f.t[i]) for i in idx])

            if CLOSE_MARKER_SIZE > 0 and len(idx):
                t_first, t_last = float(f.t[idx[0]]), float(f.t[idx[-1]])
                for t_m, u in zip(s.marker_times, marker_fractions()):
                    if t_first <= t_m <= t_last:
                        marker_pts.append(to_local(*f.position(t_m)))
                        marker_cols.append(Color(*gradient(u)))

        local = [to_local(*p) for p in centers]
        set_lines(close_link, local, [CLOSE_LINK_COLOR] * 2)
        point_color = time_color(s, t)
        set_points(close_points, local, [point_color, point_color])
        if marker_pts:
            set_points(close_markers, marker_pts, marker_cols)


def render_offscreen(camera, size: int) -> np.ndarray:
    glViewport(0, 0, size, size)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    view = camera.get_view()
    proj = camera.get_projection(size, size)
    if engine.sky:
        engine.sky.render(view_mat=view, proj_mat=proj, time=engine.time,
                          width=size, height=size, cam_pos=camera.position)
    engine.scene._draw_all(view, proj)
    raw = glReadPixels(0, 0, size, size, GL_RGB, GL_UNSIGNED_BYTE)
    return np.flipud(np.frombuffer(raw, dtype=np.uint8).reshape(size, size, 3))


def round_magnifier(img: Image.Image) -> Image.Image:
    from PIL import ImageDraw

    size = img.size[0]
    ss = 4                                                   # supersampling for smooth edges
    mask = Image.new("L", (size * ss, size * ss), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, size * ss - 1, size * ss - 1], fill=255)
    mask = mask.resize((size, size), Image.LANCZOS)

    out = img.convert("RGBA")
    out.putalpha(mask)

    rim = Image.new("RGBA", (size * ss, size * ss), (0, 0, 0, 0))
    b = CLOSE_ROUND_BORDER_PX * ss
    ImageDraw.Draw(rim).ellipse([0, 0, size * ss - 1, size * ss - 1], outline=(255, 255, 255, 255), width=b)
    return Image.alpha_composite(out, rim.resize((size, size), Image.LANCZOS))


def save_closeup(k: int, base: Path) -> None:
    s = scenes[k]
    if s.closest is None:
        s.closest = closest_pair(s)
    c = s.closest
    if c is None:
        print("[close] no aircraft pair overlapping in time above CLOSE_MIN_AGL_M in this scene")
        return

    view = closeup_view(c)
    closeup_camera.target = view["target"]
    closeup_camera.distance = view["distance"]
    closeup_camera.pitch = CLOSE_PITCH_DEG
    closeup_camera.yaw = CLOSE_YAW_DEG

    hide_scene(k)
    size = min(engine.width, engine.height)
    show_close(None, s)
    base_img = render_offscreen(closeup_camera, size).astype(np.float32)
    comp = base_img.copy()
    for layer, alpha in (("inactive", CLOSE_INACTIVE_ALPHA), ("sep", CLOSE_SEP_ALPHA), ("active", 1.0)):
        show_close(c, s, layers=(layer,))
        img_layer = render_offscreen(closeup_camera, size).astype(np.float32)
        mask = np.abs(img_layer - base_img).max(axis=2) > CLOSE_DIFF_THRESHOLD
        comp[mask] = (1.0 - alpha) * comp[mask] + alpha * img_layer[mask]
    img = np.clip(comp, 0, 255).astype(np.uint8)
    show_close(None, s)
    show_scene(k)
    glViewport(0, 0, engine.width, engine.height)

    path = base.parent / f"{base.name}_closest.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = Image.fromarray(img)
    if CLOSE_ROUND_IMAGE:
        out = round_magnifier(out)
    out.save(path)
    info = {
        "time_utc": fmt_time(c["t"]),
        "aircraft": [f"{c['a'].icao24}/{c['a'].flight_id}", f"{c['b'].icao24}/{c['b'].flight_id}"],
        "horizontal_distance_m": c["d_h_m"],
        "horizontal_distance_nm": c["d_h_m"] / 1852.0,
        "vertical_distance_m": c["d_v_m"],
        "vertical_distance_ft": c["d_v_m"] / 0.3048,
        "normalized_separation": c["metric"],
        "separation_minimum_h_m": SEP_FULL.d_h,
        "separation_minimum_v_m": SEP_FULL.d_v,
    }
    path.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"[close] saved {path} (+ .json)")


airport_elev_m = runways[0].runway_altitude * 0.3048 if runways else 0.0
cx, cz = geo.to_local(airport.lat, airport.lon)
cy = geo.altitude_to_local(airport_elev_m)

camera = FlyCamera(speed=100.0)
camera.position = Vec3(cx, cy + 300, cz + 800)
engine.camera = camera

ui = {"scene": 0, "shot": False, "legend": False, "close": False, "hud": True, "prev": {}}
show_scene(0)


def pressed(keys, key: int) -> bool:
    now = bool(keys[key])
    was = ui["prev"].get(key, False)
    ui["prev"][key] = now
    return now and not was


def switch(delta: int) -> None:
    hide_scene(ui["scene"])
    ui["scene"] = (ui["scene"] + delta) % len(scenes)
    show_scene(ui["scene"])
    s = scenes[ui["scene"]]
    print(f"[snapshot] scene {ui['scene']}: {fmt_time(s.t_snap)}  {len(s.flights)} aircraft")


def on_update(dt: float) -> None:
    keys = pygame.key.get_pressed()
    if pressed(keys, pygame.K_p):
        ui["shot"] = True
    if pressed(keys, pygame.K_l):
        ui["legend"] = True
    if pressed(keys, pygame.K_c):
        ui["close"] = True
    if pressed(keys, pygame.K_h):
        ui["hud"] = not ui["hud"]
    if pressed(keys, pygame.K_n):
        switch(+1)
    if pressed(keys, pygame.K_b):
        switch(-1)


def on_gui() -> None:
    s = scenes[ui["scene"]]

    if ui["shot"] or ui["legend"] or ui["close"]:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = SCREENSHOT_DIR / f"{ICAO}_{TAG}_scene{ui['scene']}_{int(s.t_snap)}_{stamp}"
        if ui["shot"]:
            save_screenshot(base.parent / f"{base.name}.png")
        if ui["shot"] or ui["legend"]:
            t_close = s.closest["t"] if s.closest is not None else None
            save_legend(base, s.color_t0, s.color_t1, s.marker_times, t_close)
        if ui["shot"] or ui["close"]:
            save_closeup(ui["scene"], base)
        ui["shot"] = ui["legend"] = ui["close"] = False

    if not ui["hud"]:
        return

    hud = engine.hud
    hud.panel(10, 10, 440, 130, border_radius=8)
    hud.text(f"{ICAO}  scene {ui['scene'] + 1}/{len(scenes)}  ({len(s.flights)} aircraft, all success @ s={SUCCESS_S:g})",
             20, 16, size=14, color=(255, 210, 60), shadow=True)
    hud.text(f"{fmt_time(s.color_t0)} - {fmt_time(s.color_t1)[11:]}", 20, 38, size=12, color=(200, 200, 210))

    bar_x, bar_y, bar_w, bar_h, n = 20, 62, 410, 12, 96
    sw = bar_w / n
    for i in range(n):
        r, g, b = gradient(i / (n - 1))
        hud.bar(int(bar_x + i * sw), bar_y, int(sw) + 1, bar_h, value=1.0,
                fg_color=(int(r * 255), int(g * 255), int(b * 255), 255))
    for t_m, u in zip(s.marker_times, marker_fractions()):
        mx = int(bar_x + u * (bar_w - 40))
        hud.text(fmt_time(t_m)[11:16], mx, bar_y + 16, size=11, color=(220, 220, 230))
    if s.closest is not None:
        u = (s.closest["t"] - s.color_t0) / max(s.color_t1 - s.color_t0, 1e-6)
        hud.bar(int(bar_x + u * bar_w) - 1, bar_y - 3, 3, bar_h + 6, value=1.0, fg_color=(0, 0, 0, 255))
        hud.text(f"closest {fmt_time(s.closest['t'])[11:19]}", 300, 38, size=12, color=(200, 200, 210))
    hud.text("N/B: scene   P: screenshot+legend+closest   L: legend   C: closest   H: HUD", 20, 112, size=11,
             color=(160, 160, 180))


engine.on_update = on_update
engine.on_gui = on_gui
engine.run()