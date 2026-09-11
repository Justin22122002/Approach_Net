from pathlib import Path
import re
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_RGB,
    GL_UNSIGNED_BYTE,
    glClear,
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
from engine.utils.color import Color
from engine.utils.camera import FlyCamera, OrbitCamera
from engine.utils.sky import Sky
from engine.utils.texture import Texture
from engine.utils.vec3 import Vec3
from engine.utils_project.approach_project_mesh import ApproachProjectMesh
from engine.utils_project.building_project_mesh import BuildingProjectMesh
from engine.utils_project.geo_transformer import GeoTransformer
from engine.utils_project.runway_project_mesh import RunwayProjectMesh
from engine.utils_project.taxiway_project_mesh import TaxiwayProjectMesh
from engine.utils_project.terminal_project_mesh import TerminalProjectMesh
from engine.utils_project.terrain_project_mesh import TerrainProjectMesh
from engine.utils_project.road_project_mesh import RoadProjectMesh

from dotenv import load_dotenv

from utils_project.water_texture_renderer import WaterTextureRenderer

load_dotenv()

ICAO = "CYVR"
TEST_BASE_PATH = f"../data/{ICAO}/test/20260709_1207"

WINDOW_W = 1280
WINDOW_H = 720

EXAMPLE_INDEX = 105

LATERAL_TOL_M = 15.0
VERTICAL_TOL_UP_M = 20.0
VERTICAL_TOL_DOWN_M = 20.0

LOCAL_SCALE = 0.01

TRAJECTORY_LINE_WIDTH = 3.0

WINDOW_COLOR = Color.blue()

SCREENSHOT_DIR = Path("")

USE_SATELLITE_TEXTURE = False
TERRAIN_TEX_SIZE = 8192

CASES = [
    # (label, color, traj_path, status_path)
    (
        "Ipopt-Optimizer",
        Color(0.894, 0.102, 0.110),  # 228,26,28
        Path(f"../data/{ICAO}/rnav_optimized/rnav_trajectories.parquet"),
        Path(f"../data/{ICAO}/rnav_optimized/rnav_status.jsonl"),
    ),
    (
        "FT-Transformer",
        Color(0.216, 0.494, 0.722),  # 55,126,184
        Path(f"../data/{ICAO}/multiple_airports_ft_transformer_rollout_4/ft_transformer_rollout.parquet"),
        Path(f"../data/{ICAO}/multiple_airports_ft_transformer_rollout_4/ft_transformer_rollout_status.jsonl"),
    ),
    (
        "ApproachNet (FT-Transformer)",
        Color(0.302, 0.686, 0.290),  # 77,175,74
        Path(f"../data/{ICAO}/multiple_airports_ft_transformer_rollout_constraint_4/combined_qp_rollout.parquet"),
        Path(f"../data/{ICAO}/multiple_airports_ft_transformer_rollout_constraint_4/combined_qp_rollout_status.jsonl"),
    ),
]


def _rgb255(color: Color) -> tuple[int, int, int]:
    return (
        int(round(max(0.0, min(1.0, color.r)) * 255)),
        int(round(max(0.0, min(1.0, color.g)) * 255)),
        int(round(max(0.0, min(1.0, color.b)) * 255)),
    )


def load_case_trajectory(traj_path: Path, icao24: str, flight_id: int) -> pd.DataFrame | None:
    if not traj_path.exists():
        print(f"[WARN] trajectory file not found: {traj_path}")
        return None
    df_traj = pd.read_parquet(traj_path)
    group = df_traj[
        (df_traj["icao24"].astype(str) == str(icao24))
        & (df_traj["flight_id"].astype(int) == int(flight_id))
    ].sort_values("step")
    if group.empty:
        print(f"[WARN] flight {icao24}/{flight_id} not found in {traj_path}")
        return None
    return group


basePath: Path = Path("../data")
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

engine = Engine(WINDOW_W, WINDOW_H, f"Comparison Figure - {ICAO}")
scene = engine.scene


def _runway_group_key(identifier: str) -> frozenset:
    n, suf = int(re.match(r"^(\d{1,2})([LRC]?)$", identifier.upper()).groups()[0]), \
             re.match(r"^(\d{1,2})([LRC]?)$", identifier.upper()).group(2)
    n_recip = n + 18 if n <= 18 else n - 18
    suf_recip = {"L": "R", "R": "L", "C": "C", "": ""}[suf]
    return frozenset({(n, suf), (n_recip, suf_recip)})


PREFERRED_BEARING_DEG = -90.0


def _bearing_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


groups: dict[frozenset, list[ArincRunway]] = {}
for runway in runways:
    key = _runway_group_key(runway.runway_identifier)
    groups.setdefault(key, []).append(runway)

runways_one_per_direction: list[ArincRunway] = [
    min(group, key=lambda r: _bearing_diff(r.bearing, PREFERRED_BEARING_DEG))
    for group in groups.values()
]

runway_row = None
for runway in runways_one_per_direction:
    runway_mesh_builder = RunwayProjectMesh(
        runway=runway,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        transformer=to_utm,
    )
    runway_mesh_builder.spawn(scene, geo)

for taxiway in airport.taxiways:
    if len(taxiway.coordinates) < 2:
        continue
    TaxiwayProjectMesh(
        coordinates=taxiway.coordinates,
        airport_lat=airport.lat,
        airport_lon=airport.lon,
        airport_altitude_ft=runways[0].runway_altitude - 1 if runways else 0.0,
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
    print(f"[TERRAIN] {water_features}")

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

flight_dataset = LazyFlightParquetDataset(TEST_BASE_PATH)
icao24, flight_id = flight_dataset.flights[EXAMPLE_INDEX]

fix_row = flight_dataset.get_runway_assignment(icao24, flight_id)
fix_x, fix_y = to_utm.transform(float(fix_row["fix_longitude"]), float(fix_row["fix_latitude"]))
fix_elev_m = float(fix_row["fix_elevation_m"])
runway_bearing_deg = float(fix_row["runway_bearing"])
chi_r = np.radians(runway_bearing_deg)
sin_r, cos_r = np.sin(chi_r), np.cos(chi_r)


def threshold_frame_to_utm(s: float, e_y: float) -> tuple[float, float]:
    x_rel = -sin_r * s + cos_r * e_y
    y_rel = -cos_r * s - sin_r * e_y
    return fix_x + x_rel, fix_y + y_rel


for label, color, traj_path, status_path in CASES:
    df_case = load_case_trajectory(traj_path, icao24, flight_id)
    if df_case is None:
        continue

    pts: list[Vec3] = []
    for _, row in df_case.iterrows():
        xg, zg = geo.to_local_in_m(float(row["x"]), float(row["y"]))
        yg = geo.altitude_to_local(float(row["h"]))
        pts.append(Vec3(xg, yg, zg))

    if len(pts) >= 2:
        traj_go = scene.spawn(f"Trajectory_{label}")
        traj_go.add_component(LineRenderer.strip(pts, color=color, line_width=TRAJECTORY_LINE_WIDTH))

box_x0, box_y0 = threshold_frame_to_utm(0.0, -LATERAL_TOL_M)
box_x1, box_y1 = threshold_frame_to_utm(0.0, +LATERAL_TOL_M)

corners_world = [
    (box_x0, box_y0, fix_elev_m - VERTICAL_TOL_DOWN_M),
    (box_x1, box_y1, fix_elev_m - VERTICAL_TOL_DOWN_M),
    (box_x1, box_y1, fix_elev_m + VERTICAL_TOL_UP_M),
    (box_x0, box_y0, fix_elev_m + VERTICAL_TOL_UP_M),
    (box_x0, box_y0, fix_elev_m - VERTICAL_TOL_DOWN_M),
]
window_pts: list[Vec3] = []
for wx, wy, wh in corners_world:
    xg, zg = geo.to_local_in_m(wx, wy)
    yg = geo.altitude_to_local(wh)
    window_pts.append(Vec3(xg, yg, zg))

window_go = scene.spawn("ToleranceWindow")
window_go.add_component(LineRenderer.strip(window_pts, color=WINDOW_COLOR, line_width=TRAJECTORY_LINE_WIDTH))

threshold_x, threshold_z = geo.to_local_in_m(fix_x, fix_y)
threshold_y = geo.altitude_to_local(fix_elev_m)
threshold_pos = Vec3(threshold_x, threshold_y, threshold_z)

main_camera = FlyCamera(speed=100.0)
main_camera.position = Vec3(threshold_x, threshold_y + 300, threshold_z + 800)

closeup_camera = OrbitCamera(distance=1.0)
closeup_camera.target = threshold_pos
closeup_camera.pitch = 10.0
closeup_camera.yaw = -90.0

engine.camera = main_camera

_screenshot_counter = {"n": 0}


def on_gui() -> None:
    hud = engine.hud
    hud.text(f"Example flight: {icao24} / {flight_id}  (index={EXAMPLE_INDEX})", 10, 10, size=16, shadow=True)
    hud.text("Main camera: fly freely  |  Press [Save] to export comparison figure", 10, 32, size=13,
              color=(180, 180, 200))

    hud.button(
        "Save Comparison Figure",
        x=10, y=engine.height - 50,
        w=260,
        h=36,
        callback=save_composite_screenshot,
        color=(30, 100, 60, 220),
        hover_color=(50, 150, 90, 240),
    )

    hud.button(
        "Save Screenshot",
        x=280, y=engine.height - 50,
        w=180,
        h=36,
        callback=lambda: save_plain_screenshot(with_legend=False),
        color=(60, 60, 140, 220),
        hover_color=(90, 90, 180, 240),
    )

    hud.button(
        "Save Screenshot + Legend",
        x=470, y=engine.height - 50,
        w=220,
        h=36,
        callback=lambda: save_plain_screenshot(with_legend=True),
        color=(140, 90, 30, 220),
        hover_color=(180, 130, 50, 240),
    )


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


def draw_legend(img, legend_scale: float = 2.5):
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img)

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

    legend_x, legend_y = 16, 16

    entries: list[tuple[str, Color]] = [(label, color) for label, color, _, _ in CASES]
    entries.append(("Tolerance Window", WINDOW_COLOR))

    text_w = max(draw.textbbox((0, 0), label, font=font)[2] for label, _ in entries)
    box_w = swatch_w + text_w + padding * 2
    box_h = len(entries) * line_height + padding * 2

    draw.rectangle(
        [legend_x - padding, legend_y - padding, legend_x - padding + box_w, legend_y - padding + box_h],
        fill=(35, 35, 35),
    )

    for i, (label, color) in enumerate(entries):
        y = legend_y + i * line_height
        rgb = _rgb255(color)
        draw.ellipse(
            [legend_x + dot_offset, y + dot_offset,
             legend_x + dot_offset + dot_diameter, y + dot_offset + dot_diameter],
            fill=rgb,
        )
        draw.text((legend_x + swatch_w, y), label, fill=(255, 255, 255), font=font)

    return img


def save_plain_screenshot(with_legend: bool = False) -> None:
    from PIL import Image

    arr = render_camera_offscreen(engine.camera, engine.width, engine.height)
    img = Image.fromarray(arr, mode="RGB")

    if with_legend:
        img = draw_legend(img)

    _screenshot_counter["n"] += 1
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCREENSHOT_DIR / f"screenshot_{_screenshot_counter['n']}.png"
    img.save(out_path)
    print(f"[SAVED] {out_path.resolve()}  ({engine.width}x{engine.height}px)  legend={with_legend}")


def save_composite_screenshot() -> None:
    from PIL import Image, ImageDraw, ImageFilter

    square_size = min(engine.width, engine.height)

    main_arr = render_camera_offscreen(main_camera, square_size, square_size)
    closeup_arr = render_camera_offscreen(closeup_camera, square_size, square_size)

    main_img = Image.fromarray(main_arr, mode="RGB")
    closeup_img = Image.fromarray(closeup_arr, mode="RGB")

    inset_size = square_size // 2
    closeup_img = closeup_img.resize((inset_size, inset_size))

    INSET_BORDER_PX = 3
    SHADOW_BLUR = 14
    SHADOW_OPACITY = 140  # 0-255

    inset_x = square_size - inset_size
    inset_y = square_size - inset_size

    composite = main_img.copy().convert("RGBA")

    shadow_layer = Image.new("RGBA", composite.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    shadow_draw.rectangle(
        [inset_x, inset_y, inset_x + inset_size, inset_y + inset_size],
        fill=(0, 0, 0, SHADOW_OPACITY),
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))
    composite = Image.alpha_composite(composite, shadow_layer)

    framed = Image.new(
        "RGB",
        (inset_size + 2 * INSET_BORDER_PX, inset_size + 2 * INSET_BORDER_PX),
        (0, 0, 0),
    )
    framed.paste(closeup_img, (INSET_BORDER_PX, INSET_BORDER_PX))

    composite.paste(
        framed,
        (inset_x - INSET_BORDER_PX, inset_y - INSET_BORDER_PX),
    )
    composite = composite.convert("RGB")

    composite = draw_legend(composite)

    out_path = Path("comparison_figure.png")
    composite.save(out_path)
    print(f"[SAVED] {out_path.resolve()}  ({square_size}x{square_size}px)")


engine.on_gui = on_gui
engine.run()