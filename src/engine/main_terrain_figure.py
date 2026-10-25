import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
from PIL import Image
from pyproj import Transformer
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_RGB,
    GL_UNSIGNED_BYTE,
    glClear,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glReadPixels,
    glViewport,
)

from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset

from utils.airport import Airport
from utils.arinc_runway import ArincRunway
from utils.clients.airport_client import OpenStreetMapAirportClient
from utils.clients.osm_water_client import OpenStreetMapWaterClient

from engine.utils.engine import Engine
from engine.utils.lineRenderer import LineRenderer
from engine.utils.mesh import Mesh
from engine.utils.meshRenderer import MeshRenderer
from engine.utils.simplematerial import SimpleMaterial
from engine.utils.color import Color
from engine.utils.camera import FlyCamera, OrbitCamera
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
from engine.utils_project.rnav_lpv_funnel_project_mesh import RnavLpvFunnelProjectMesh
from engine.utils_project.water_texture_renderer import WaterTextureRenderer

from dotenv import load_dotenv

load_dotenv()

ICAO = "CYVR"
DATA_ROOT = Path("../data")
TEST_DATASET = "20261005_1406"   # 20261005_1406 -> new test set, 20260709_1207 -> old test set

MODE = "paper"
START_INDEX = 105
ONLY_COMMON = True

WINDOW_W = 1280
WINDOW_H = 720

LATERAL_TOL_M = 15.0
VERTICAL_TOL_UP_M = 20.0
VERTICAL_TOL_DOWN_M = 20.0

LOCAL_SCALE = 0.01
TRAJECTORY_LINE_WIDTH = 3.0
WINDOW_COLOR = Color.blue()
WINDOW_FILL_OPACITY = 0.25
SAVE_LEGEND_SEPARATE = True

SCREENSHOT_DIR = DATA_ROOT / ICAO / "screenshots"

USE_SATELLITE_TEXTURE = False
TERRAIN_TEX_SIZE = 8192
PREFERRED_BEARING_DEG = -90.0

RUN_FT_ACTION = "2026-10-10_20-33-57"
RUN_FT_ACTION_CONSTRAINT_TRAINING = "2026-10-11_06-09-41"

RUN_RESNET_ACTION = "2026-10-10_20-02-35"
RUN_RESNET_ACTION_CONSTRAINT_TRAINING = "2026-10-11_05-34-36"

RUN_FT_NEXT_STATE = "2026-10-11_16-49-30"
RUN_RESNET_NEXT_STATE = "2026-10-11_18-23-44"

RUN_FT_GNN = "2026-10-10_23-39-39"
RUN_FT_GNN_CONSTRAINT_TRAINING = "2026-10-11_17-13-54"

RUN_RESNET_GNN = "2026-10-12_21-14-14"
RUN_RESNET_GNN_CONSTRAINT_TRAINING = "2026-10-13_12-57-59"

RUN_RNAV = TEST_DATASET

# label -> (output_dir_name, name, run_id, constrained)
EXPERIMENTS: dict[str, tuple[str, str, str, bool]] = {
    "RNAV_OPTIMIZER": ("rnav_optimized", "rnav_optimizer", RUN_RNAV, False),

    "RESNET101_ACTION_BASELINE": ("multiple_airports_resnet101_rollout", "resnet101", RUN_RESNET_ACTION, False),
    "RESNET101_ACTION_CONSTRAINT": ("multiple_airports_resnet101_rollout_constraint", "resnet101",
                                    RUN_RESNET_ACTION, True),
    "RESNET101_ACTION_CONSTRAINT_TRAINING": ("multiple_airports_resnet101_rollout_constraint_training", "resnet101",
                                             RUN_RESNET_ACTION_CONSTRAINT_TRAINING, True),

    "FT_TRANSFORMER_ACTION_BASELINE": ("multiple_airports_ft_transformer_rollout", "ft_transformer",
                                       RUN_FT_ACTION, False),
    "FT_TRANSFORMER_ACTION_CONSTRAINT": ("multiple_airports_ft_transformer_rollout_constraint", "ft_transformer",
                                         RUN_FT_ACTION, True),
    "FT_TRANSFORMER_ACTION_CONSTRAINT_TRAINING": ("multiple_airports_ft_transformer_rollout_constraint_training",
                                                  "ft_transformer", RUN_FT_ACTION_CONSTRAINT_TRAINING, True),

    "RESNET101_NEXT_STATE_BASELINE": ("multiple_airports_next_state_resnet101_rollout", "resnet101_next_state",
                                      RUN_RESNET_NEXT_STATE, False),
    "RESNET101_NEXT_STATE_CONSTRAINT": ("multiple_airports_next_state_resnet101_rollout_constraint",
                                        "resnet101_next_state", RUN_RESNET_NEXT_STATE, True),
    "FT_TRANSFORMER_NEXT_STATE_BASELINE": ("multiple_airports_ft_transformer_next_state_rollout",
                                           "ft_transformer_next_state", RUN_FT_NEXT_STATE, False),
    "FT_TRANSFORMER_NEXT_STATE_CONSTRAINT": ("multiple_airports_ft_transformer_next_state_rollout_constraint",
                                             "ft_transformer_next_state", RUN_FT_NEXT_STATE, True),

    "FT_GNN_BASELINE": ("multiple_airports_ft_transformer_gnn_rollout", "gnn", RUN_FT_GNN, False),
    "FT_GNN_CONSTRAINT": ("multiple_airports_ft_transformer_gnn_rollout_constraint", "gnn", RUN_FT_GNN, True),
    "FT_GNN_CONSTRAINT_TRAINING": ("multiple_airports_ft_transformer_gnn_rollout_constraint_training", "gnn",
                                   RUN_FT_GNN_CONSTRAINT_TRAINING, True),

    "RESNET_GNN_BASELINE": ("multiple_airports_resnet_101_gnn_rollout", "gnn", RUN_RESNET_GNN, False),
    "RESNET_GNN_CONSTRAINT": ("multiple_airports_resnet_101_gnn_rollout_constraint", "gnn", RUN_RESNET_GNN, True),
    "RESNET_GNN_CONSTRAINT_TRAINING": ("multiple_airports_resnet_101_gnn_rollout_constraint_training", "gnn",
                                       RUN_RESNET_GNN_CONSTRAINT_TRAINING, True),
}

MODES: dict[str, list[tuple[str, str]]] = {
    "paper": [
        ("RNAV_OPTIMIZER", "Ipopt-Optimizer"),
        ("FT_TRANSFORMER_ACTION_BASELINE", "FT-Transformer"),
        ("FT_TRANSFORMER_ACTION_CONSTRAINT", "ApproachNet (FT-Transformer)"),
    ],
    "ft_action": [
        ("RNAV_OPTIMIZER", "Ipopt-Optimizer"),
        ("FT_TRANSFORMER_ACTION_BASELINE", "FT-Transformer"),
        ("FT_TRANSFORMER_ACTION_CONSTRAINT", "FT-Transformer + QP"),
        ("FT_TRANSFORMER_ACTION_CONSTRAINT_TRAINING", "FT-Transformer + QP (trained)"),
    ],
    "resnet_action": [
        ("RNAV_OPTIMIZER", "Ipopt-Optimizer"),
        ("RESNET101_ACTION_BASELINE", "ResNet"),
        ("RESNET101_ACTION_CONSTRAINT", "ResNet + QP"),
        ("RESNET101_ACTION_CONSTRAINT_TRAINING", "ResNet + QP (trained)"),
    ],
    "next_state": [
        ("RNAV_OPTIMIZER", "Ipopt-Optimizer"),
        ("FT_TRANSFORMER_NEXT_STATE_BASELINE", "FT-Transformer next state"),
        ("FT_TRANSFORMER_NEXT_STATE_CONSTRAINT", "FT-Transformer next state + QP"),
        ("RESNET101_NEXT_STATE_BASELINE", "ResNet next state"),
        ("RESNET101_NEXT_STATE_CONSTRAINT", "ResNet next state + QP"),
    ],
    "ft_gnn": [
        ("RNAV_OPTIMIZER", "Ipopt-Optimizer"),
        ("FT_GNN_BASELINE", "FT-GNN"),
        ("FT_GNN_CONSTRAINT", "FT-GNN + QP"),
        ("FT_GNN_CONSTRAINT_TRAINING", "FT-GNN + QP (trained)"),
    ],
    "resnet_gnn": [
        ("RNAV_OPTIMIZER", "Ipopt-Optimizer"),
        ("RESNET_GNN_BASELINE", "ResNet-GNN"),
        ("RESNET_GNN_CONSTRAINT", "ResNet-GNN + QP"),
        ("RESNET_GNN_CONSTRAINT_TRAINING", "ResNet-GNN + QP (trained)"),
    ],
}

PALETTE: list[Color] = [
    Color(0.894, 0.102, 0.110),  # red
    Color(0.216, 0.494, 0.722),  # blue
    Color(0.302, 0.686, 0.290),  # green
    Color(0.596, 0.306, 0.639),  # purple
    Color(1.000, 0.498, 0.000),  # orange
    Color(0.651, 0.337, 0.157),  # brown
]

HIDDEN = Vec3(0.0, -1.0e5, 0.0)


@dataclass
class Case:
    label: str
    traj_path: Path
    status_path: Path
    eval_path: Path
    loaded: bool = False
    available: bool = False
    _df: pd.DataFrame | None = None
    _rows: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)
    _success: dict[tuple[str, int], bool] = field(default_factory=dict)

    @classmethod
    def from_experiment(cls, label: str) -> "Case":
        out_dir, name, run_id, constrained = EXPERIMENTS[label]
        tag = f"{name}{'_constrained' if constrained else ''}_rollout_{run_id}"
        folder = DATA_ROOT / ICAO / out_dir
        return cls(label, folder / f"{tag}.parquet", folder / f"{tag}_status.jsonl",
                   folder / f"{tag}_evaluation.csv")

    def load(self) -> None:
        if self.loaded:
            return
        self.loaded = True
        if not self.traj_path.exists():
            print(f"[WARN] {self.label}: trajectory file not found: {self.traj_path}")
            return
        self._df = pd.read_parquet(self.traj_path, columns=["icao24", "flight_id", "step", "x", "y", "h"])
        self._rows = {(str(k[0]), int(k[1])): v for k, v in self._df.groupby(["icao24", "flight_id"]).indices.items()}

        if self.eval_path.exists():
            ev = pd.read_csv(self.eval_path)
            self._success = {(str(r.icao24), int(r.flight_id)): bool(r.success) for r in ev.itertuples()}
            src = "evaluation.csv"
        elif self.status_path.exists():
            for line in open(self.status_path):
                s = json.loads(line)
                self._success[(str(s["icao24"]), int(s["flight_id"]))] = bool(s["success"])
            src = "status.jsonl"
        else:
            src = "none"
        self.available = True
        print(f"[DATA] {self.label}: {len(self._rows)} flights  (success from {src})  {self.traj_path.name}")

    def has(self, key: tuple[str, int]) -> bool:
        self.load()
        return key in self._rows

    def trajectory(self, key: tuple[str, int]) -> np.ndarray | None:
        """(n, 3) x, y, h sorted by step"""
        self.load()
        if key not in self._rows:
            return None
        g = self._df.iloc[self._rows[key]].sort_values("step")
        return g[["x", "y", "h"]].to_numpy(float)

    def success(self, key: tuple[str, int]) -> bool | None:
        self.load()
        return self._success.get(key)


case_cache: dict[str, Case] = {}


def mode_cases(mode: str) -> list[tuple[Case, str, Color]]:
    out = []
    for i, (label, display) in enumerate(MODES[mode]):
        if label not in case_cache:
            case_cache[label] = Case.from_experiment(label)
        case_cache[label].load()
        out.append((case_cache[label], display, PALETTE[i % len(PALETTE)]))
    return out


def _rgb255(color: Color) -> tuple[int, int, int]:
    return (
        int(round(max(0.0, min(1.0, color.r)) * 255)),
        int(round(max(0.0, min(1.0, color.g)) * 255)),
        int(round(max(0.0, min(1.0, color.b)) * 255)),
    )


def set_lines(lr: LineRenderer, points: list[Vec3], color: Color) -> None:
    """polyline -> segment list, force GPU buffer rebuild; < 2 points hides the line"""
    if len(points) < 2:
        points = [HIDDEN, HIDDEN]
    seg_pts: list[Vec3] = []
    for i in range(len(points) - 1):
        seg_pts += [points[i], points[i + 1]]
    lr.points, lr.colors = seg_pts, [color] * len(seg_pts)
    if lr._vao is not None:
        glDeleteVertexArrays(1, [lr._vao])
        glDeleteBuffers(1, [lr._vbo])
        lr._vao = None
        lr._vbo = None


def quad_mesh(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> Mesh:
    """double-sided quad, vertex layout pos(3) normal(3) uv(2) like the funnel fill"""
    u = np.array([p1.x - p0.x, p1.y - p0.y, p1.z - p0.z])
    v = np.array([p3.x - p0.x, p3.y - p0.y, p3.z - p0.z])
    n = np.cross(u, v)
    n = n / max(np.linalg.norm(n), 1e-12)

    verts: list[float] = []
    for tri in ((p0, p1, p2), (p0, p2, p3)):
        for p in tri:                 # front side
            verts += [p.x, p.y, p.z, *n, 0.0, 0.0]
        for p in reversed(tri):       # back side
            verts += [p.x, p.y, p.z, *(-n), 0.0, 0.0]
    return Mesh(np.asarray(verts, dtype=np.float32), np.arange(len(verts) // 8, dtype=np.uint32))


airport = Airport.load_from_folder(DATA_ROOT, ICAO)

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

engine = Engine(WINDOW_W, WINDOW_H, f"Comparison Figure - {ICAO}")
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

for runway in runways:
    for approach in runway.approaches:
        funnel_mesh = RnavLpvFunnelProjectMesh(
            approach=approach,
            transformer=to_utm,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            runway=runway,
        )
        funnel_mesh.spawn(scene, geo)

for taxiway in airport.taxiways:
    if len(taxiway.coordinates) < 2:
        continue

    builder = TaxiwayProjectMesh(
        coordinates=taxiway.coordinates,
        airport_lat=airport.lat,
        airport_lon=airport.lon,
        airport_altitude_ft=runways[0].runway_altitude if runways else 0.0,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        transformer=to_utm,
        use_mesh=True,
    )

    builder.spawn(scene, geo)

map_client = OpenStreetMapAirportClient()

water_tex_path = DATA_ROOT / ICAO / ("terrain_water_sat.png" if USE_SATELLITE_TEXTURE else "terrain_water.png")

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

flight_dataset = LazyFlightParquetDataset(str(DATA_ROOT / ICAO / "test" / TEST_DATASET))
flights: list[tuple[str, int]] = [(str(a), int(b)) for a, b in flight_dataset.flights]
print(f"[DATA] test set {TEST_DATASET}: {len(flights)} flights")

line_pool: list[LineRenderer] = []


def pool_line(i: int) -> LineRenderer:
    while len(line_pool) <= i:
        lr = LineRenderer.strip([HIDDEN, HIDDEN], color=PALETTE[0], line_width=TRAJECTORY_LINE_WIDTH)
        scene.spawn(f"Trajectory_{len(line_pool)}").add_component(lr)
        line_pool.append(lr)
    return line_pool[i]


window_lr = LineRenderer.strip([HIDDEN, HIDDEN], color=WINDOW_COLOR, line_width=TRAJECTORY_LINE_WIDTH)
scene.spawn("ToleranceWindow").add_component(window_lr)

window_fill = MeshRenderer(
    quad_mesh(HIDDEN, HIDDEN, HIDDEN, HIDDEN),
    material=SimpleMaterial(color=WINDOW_COLOR, opacity=WINDOW_FILL_OPACITY),
    cast_shadow=False,
)
scene.spawn("ToleranceWindowFill").add_component(window_fill)


def set_window_fill(corners: list[Vec3] | None) -> None:
    old = window_fill.mesh
    window_fill.mesh = quad_mesh(*corners) if corners else quad_mesh(HIDDEN, HIDDEN, HIDDEN, HIDDEN)
    old.destroy()


def to_local(x: float, y: float, h: float) -> Vec3:
    xg, zg = geo.to_local_in_m(x, y)
    return Vec3(xg, geo.altitude_to_local(h), zg)


main_camera = FlyCamera(speed=100.0)
closeup_camera = OrbitCamera(distance=1.0)
closeup_camera.pitch = 10.0
closeup_camera.yaw = -90.0
engine.camera = main_camera

state = {
    "mode": MODE,
    "idx": START_INDEX % len(flights),
    "common": ONLY_COMMON,
    "threshold": Vec3(0.0, 0.0, 0.0),
    "rows": [],          # HUD rows: (display, color, n_points | None, success | None)
    "fix_ok": False,
    "prev_keys": {},
}


def runway_fix(icao24: str, flight_id: int) -> tuple[float, float, float, float] | None:
    try:
        row = flight_dataset.get_runway_assignment(icao24, flight_id)
        fx, fy = to_utm.transform(float(row["fix_longitude"]), float(row["fix_latitude"]))
        return fx, fy, float(row["fix_elevation_m"]), float(row["runway_bearing"])
    except Exception as e:
        print(f"[WARN] no runway assignment for {icao24}/{flight_id}: {e}")
        return None


def reset_main_camera() -> None:
    t = state["threshold"]
    main_camera.position = Vec3(t.x, t.y + 300, t.z + 800)


def show_flight(idx: int, reset_camera: bool = False) -> None:
    state["idx"] = idx
    key = flights[idx]
    cases = mode_cases(state["mode"])

    # trajectories
    state["rows"] = []
    for i, (case, display, color) in enumerate(cases):
        xyz = case.trajectory(key) if case.available else None
        pts = [to_local(*p) for p in xyz] if xyz is not None else []
        set_lines(pool_line(i), pts, color)
        state["rows"].append((display, color, len(pts) if xyz is not None else None, case.success(key)))
    for i in range(len(cases), len(line_pool)):
        set_lines(line_pool[i], [], PALETTE[0])

    fix = runway_fix(*key)
    state["fix_ok"] = fix is not None
    if fix is None:
        set_lines(window_lr, [], WINDOW_COLOR)
        set_window_fill(None)
    else:
        fx, fy, fh, bearing = fix
        chi_r = np.radians(bearing)
        sin_r, cos_r = np.sin(chi_r), np.cos(chi_r)

        def frame_to_utm(s: float, e_y: float) -> tuple[float, float]:
            return fx + (-sin_r * s + cos_r * e_y), fy + (-cos_r * s - sin_r * e_y)

        (x0, y0), (x1, y1) = frame_to_utm(0.0, -LATERAL_TOL_M), frame_to_utm(0.0, +LATERAL_TOL_M)
        lo, hi = fh - VERTICAL_TOL_DOWN_M, fh + VERTICAL_TOL_UP_M
        corners = [(x0, y0, lo), (x1, y1, lo), (x1, y1, hi), (x0, y0, hi), (x0, y0, lo)]
        window_pts = [to_local(*c) for c in corners]
        set_lines(window_lr, window_pts, WINDOW_COLOR)
        set_window_fill(window_pts[:4])

        state["threshold"] = to_local(fx, fy, fh)
        closeup_camera.target = state["threshold"]

    if reset_camera:
        reset_main_camera()

    summary = "  ".join(f"{d}={'-' if n is None else n}" for d, _, n, _ in state["rows"])
    print(f"[VIEW] mode={state['mode']}  idx={idx}  {key[0]}/{key[1]}  {summary}")


def has_all(idx: int) -> bool:
    key = flights[idx]
    avail = [c for c, _, _ in mode_cases(state["mode"]) if c.available]
    return all(c.has(key) for c in avail)


def step(delta: int) -> None:
    n = len(flights)
    i = (state["idx"] + delta) % n
    if state["common"]:
        direction = 1 if delta >= 0 else -1
        for _ in range(n):
            if has_all(i):
                break
            i = (i + direction) % n
        else:
            print("[WARN] no flight present in all experiments of this mode")
    show_flight(i)


def next_mode() -> None:
    names = list(MODES)
    state["mode"] = names[(names.index(state["mode"]) + 1) % len(names)]
    print(f"[MODE] {state['mode']}")
    step(0)


def render_camera_offscreen(camera, width: int, height: int) -> np.ndarray:
    glViewport(0, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    view = camera.get_view()
    proj = camera.get_projection(width, height)

    if engine.sky:
        engine.sky.render(
            view_mat=view, proj_mat=proj, time=engine.time,
            width=width, height=height, cam_pos=camera.position,
        )

    engine.scene._draw_all(view, proj)

    raw = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    return np.flipud(arr)


def legend_image(legend_scale: float = 2.5) -> Image.Image | None:
    from PIL import ImageDraw, ImageFont

    font_size = int(round(20 * legend_scale))
    line_height = int(round(26 * legend_scale))
    padding = int(round(10 * legend_scale))
    swatch_w = int(round(24 * legend_scale))
    dot_diameter = int(round(12 * legend_scale))
    dot_offset = (line_height - dot_diameter) // 2

    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        font = ImageFont.load_default()

    entries: list[tuple[str, Color]] = [(d, c) for d, c, n, _ in state["rows"] if n]
    if state["fix_ok"]:
        entries.append(("Tolerance Window", WINDOW_COLOR))
    if not entries:
        return None

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_w = max(measure.textbbox((0, 0), label, font=font)[2] for label, _ in entries)
    box_w = swatch_w + text_w + padding * 2
    box_h = len(entries) * line_height + padding * 2

    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, box_w - 1, box_h - 1], fill=(35, 35, 35, 255))

    for i, (label, color) in enumerate(entries):
        y = padding + i * line_height
        draw.ellipse(
            [padding + dot_offset, y + dot_offset,
             padding + dot_offset + dot_diameter, y + dot_offset + dot_diameter],
            fill=_rgb255(color),
        )
        draw.text((padding + swatch_w, y), label, fill=(255, 255, 255), font=font)

    return img


def save_legend_for(image_path: Path) -> None:
    if not SAVE_LEGEND_SEPARATE:
        return
    img = legend_image()
    if img is None:
        return
    path = image_path.with_name(f"{image_path.stem}_legend.png")
    img.save(path)
    print(f"[SAVED] {path.resolve()}")


def out_path(kind: str) -> Path:
    icao24, flight_id = flights[state["idx"]]
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    base = f"{ICAO}_{state['mode']}_idx{state['idx']}_{icao24}_{flight_id}_{kind}"
    path, n = SCREENSHOT_DIR / f"{base}.png", 1
    while path.exists():
        n += 1
        path = SCREENSHOT_DIR / f"{base}_{n}.png"
    return path


def save_plain_screenshot() -> None:
    arr = render_camera_offscreen(engine.camera, engine.width, engine.height)
    path = out_path("screenshot")
    Image.fromarray(arr, mode="RGB").save(path)
    print(f"[SAVED] {path.resolve()}  ({engine.width}x{engine.height}px)")
    save_legend_for(path)


def save_composite_screenshot() -> None:
    from PIL import ImageDraw, ImageFilter

    square_size = min(engine.width, engine.height)

    main_img = Image.fromarray(render_camera_offscreen(main_camera, square_size, square_size), mode="RGB")
    closeup_img = Image.fromarray(render_camera_offscreen(closeup_camera, square_size, square_size), mode="RGB")

    inset_size = square_size // 2
    closeup_img = closeup_img.resize((inset_size, inset_size))

    INSET_BORDER_PX = 3
    SHADOW_BLUR = 14
    SHADOW_OPACITY = 140  # 0-255

    inset_x = square_size - inset_size
    inset_y = square_size - inset_size

    composite = main_img.copy().convert("RGBA")

    shadow_layer = Image.new("RGBA", composite.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow_layer).rectangle(
        [inset_x, inset_y, inset_x + inset_size, inset_y + inset_size],
        fill=(0, 0, 0, SHADOW_OPACITY),
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))
    composite = Image.alpha_composite(composite, shadow_layer)

    framed = Image.new("RGB", (inset_size + 2 * INSET_BORDER_PX, inset_size + 2 * INSET_BORDER_PX), (0, 0, 0))
    framed.paste(closeup_img, (INSET_BORDER_PX, INSET_BORDER_PX))
    composite.paste(framed, (inset_x - INSET_BORDER_PX, inset_y - INSET_BORDER_PX))
    path = out_path("comparison")
    composite.convert("RGB").save(path)
    print(f"[SAVED] {path.resolve()}  ({square_size}x{square_size}px)")
    save_legend_for(path)


def pressed(keys, key: int) -> bool:
    now = bool(keys[key])
    was = state["prev_keys"].get(key, False)
    state["prev_keys"][key] = now
    return now and not was


def on_update(dt: float) -> None:
    keys = pygame.key.get_pressed()
    if pressed(keys, pygame.K_n):
        step(+1)
    if pressed(keys, pygame.K_b):
        step(-1)
    if pressed(keys, pygame.K_PAGEUP):
        step(+10)
    if pressed(keys, pygame.K_PAGEDOWN):
        step(-10)
    if pressed(keys, pygame.K_m):
        next_mode()
    if pressed(keys, pygame.K_f):
        state["common"] = not state["common"]
        print(f"[FILTER] only common flights: {state['common']}")
    if pressed(keys, pygame.K_r):
        reset_main_camera()


def on_gui() -> None:
    hud = engine.hud
    icao24, flight_id = flights[state["idx"]]

    hud.text(f"{ICAO}  mode: {state['mode']}   flight {state['idx']}/{len(flights) - 1}:  {icao24} / {flight_id}"
             f"{'' if state['fix_ok'] else '   (no runway fix)'}", 10, 10, size=16, shadow=True)
    hud.text(f"N/B: +-1   PgUp/PgDn: +-10   M: mode   F: only common [{'on' if state['common'] else 'off'}]"
             f"   R: reset camera", 10, 32, size=13, color=(180, 180, 200))

    y = 56
    for display, color, n, success in state["rows"]:
        r, g, b = _rgb255(color)
        status = "missing" if n is None else ("n/a" if success is None else ("SUCCESS" if success else "FAIL"))
        status_color = {"SUCCESS": (120, 230, 120), "FAIL": (255, 110, 90)}.get(status, (170, 170, 180))
        hud.text("●", 10, y, size=14, color=(r, g, b))
        hud.text(f"{display}", 28, y, size=13, color=(230, 230, 235))
        hud.text(f"{status}" + ("" if n is None else f"  ({n} pts)"), 300, y, size=13, color=status_color)
        y += 18

    hud.button("< prev", x=10, y=engine.height - 94, w=90, h=34, callback=lambda: step(-1),
               color=(60, 60, 60, 220), hover_color=(90, 90, 90, 240))
    hud.button("next >", x=110, y=engine.height - 94, w=90, h=34, callback=lambda: step(+1),
               color=(60, 60, 60, 220), hover_color=(90, 90, 90, 240))
    hud.button("mode", x=210, y=engine.height - 94, w=90, h=34, callback=next_mode,
               color=(60, 60, 60, 220), hover_color=(90, 90, 90, 240))

    hud.button("Save Comparison Figure", x=10, y=engine.height - 50, w=260, h=36,
               callback=save_composite_screenshot, color=(30, 100, 60, 220), hover_color=(50, 150, 90, 240))
    hud.button("Save Screenshot", x=280, y=engine.height - 50, w=180, h=36,
               callback=save_plain_screenshot,
               color=(60, 60, 140, 220), hover_color=(90, 90, 180, 240))


state["idx"] = (START_INDEX - 1) % len(flights)
step(+1)
reset_main_camera()

engine.on_update = on_update
engine.on_gui = on_gui
engine.run()