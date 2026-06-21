from collections import defaultdict, deque
from datetime import timedelta
from pathlib import Path

import pandas as pd
from pyproj import Transformer

from utils.aircraft_history_interpolator import AircraftHistoryInterpolator
from utils.airport import Airport
from utils.arinc_runway import ArincRunway
from utils.engine import Engine
from utils.color import Color
from utils.directionalLight import DirectionalLight
from utils.camera import FlyCamera
from utils.clients.terrain_client import CopernicusDEMProvider
from utils.vec3 import Vec3
from utils_project.approach_project_mesh import ApproachProjectMesh
from utils_project.geo_transformer import GeoTransformer
from utils_project.runway_project_mesh import RunwayProjectMesh
from utils_project.taxiway_project_mesh import TaxiwayProjectMesh
from utils_project.terrain_project_mesh import TerrainProjectMesh
from utils.pointRenderer import PointRenderer
from utils.lineRenderer import LineRenderer

from OpenGL.GL import glDeleteVertexArrays, glDeleteBuffers

icao_frankfurt = {
    "icao": "EDDF",
    "lat": 50.0379,
    "lon": 8.5622
}

icao_vancouver = {
    "icao": "CYVR",
    "lat": 49.1967,
    "lon": -123.1815
}

icao = icao_vancouver

crs_to: str = GeoTransformer.utm_epsg(icao["lat"], icao["lon"])
transformer: Transformer = Transformer.from_crs(
    crs_from="EPSG:4326",
    crs_to=crs_to,
    always_xy=True
)

basePath: Path = Path("../data")
frankfurt = Airport.load_from_folder(basePath, icao["icao"])

history_file_path: Path = basePath / Path("CYVR/arrivals_20260621_1946.parquet")
interpolator: AircraftHistoryInterpolator = AircraftHistoryInterpolator(history_file_path, airport_height=frankfurt.arinc_424_runway[0].runway_altitude)
# interpolator.clean_dataframe()
# interpolator.interpolate()

runways: list[ArincRunway] = frankfurt.arinc_424_runway

email = "justin.groh@study.thws.de"
password = "Copernicus2026!"

delta = 0.2
LAT_MIN = icao["lat"] - delta
LAT_MAX = icao["lat"] + delta
LON_MIN = icao["lon"] - delta
LON_MAX = icao["lon"] + delta

provider = CopernicusDEMProvider(email, password)
grid = provider.get_terrain(
    lat_min=LAT_MIN,
    lon_min=LON_MIN,
    lat_max=LAT_MAX,
    lon_max=LON_MAX,
    width=1024,
    height=1024
)

horizontal_scale = 0.01
vertical_scale   = 0.01

geo = GeoTransformer(
    lat=0.0,
    lon=0.0,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    transformer=transformer
)

terrain_project_mesh = TerrainProjectMesh(
    grid=grid,
    lat_min=LAT_MIN,
    lon_min=LON_MIN,
    lat_max=LAT_MAX,
    lon_max=LON_MAX,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    transformer=transformer
)

engine = Engine(1280, 720, f"Terrain - {icao['icao']}")
scene  = engine.scene

terrain_project_mesh.spawn(scene=scene, geo=geo)

rw = None
for runway in runways:
    runway_mesh_builder = RunwayProjectMesh(
        runway=runway,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        transformer=transformer
    )
    objects = runway_mesh_builder.spawn(scene, geo)
    rw = objects[0].transform.position

    for approach in runway.approaches:
        builder = ApproachProjectMesh(
            approach=approach,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            transformer=transformer
        )
        builder.spawn(scene, geo)

for taxiway in frankfurt.taxiways:
    if len(taxiway.coordinates) < 2:
        continue
    builder = TaxiwayProjectMesh(
        coordinates=taxiway.coordinates,
        airport_lat=icao["lat"],
        airport_lon=icao["lon"],
        airport_altitude_ft=runways[0].runway_altitude if runways else 0.0,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        transformer=transformer,
    )
    builder.spawn(scene, geo)

scene.ambient_light     = Color(0.25, 0.25, 0.3)
scene.directional_light = DirectionalLight(
    direction=Vec3(-1, -2, -1),
    color=Color(1.0, 0.95, 0.85),
    intensity=1.0
)

engine.camera          = FlyCamera(speed=100.0)
engine.camera.position = Vec3(
    rw.x,
    geo.altitude_to_local(terrain_project_mesh.altitude) + 50,
    rw.z
)

SIM_SPEED: float = 10.0   # 1 real second = 60 sim seconds  (1 min/s)

df = interpolator.dataframe
sim_time = df["time"].min()

grouped: dict[str, pd.DataFrame] = {
    icao24: grp.sort_values("time").set_index("time")
    for icao24, grp in df.groupby("icao24")
}

trails: dict[str, deque[Vec3]] = defaultdict(lambda: deque(maxlen=10))

aircraft_gos: dict[str, dict] = {}

def latlon_to_local(lat: float, lon: float, alt_m: float) -> Vec3:
    lx, lz = geo.to_local(lat, lon)
    ly = geo.altitude_to_local(alt_m)
    return Vec3(lx, ly, lz)


def get_row_at(group: pd.DataFrame, t: pd.Timestamp) -> pd.Series | None:
    t = t.as_unit(group.index.unit)
    idx = group.index.searchsorted(t, side="right") - 1
    if idx < 0:
        return None
    return group.iloc[idx]


for icao24 in grouped:
    point_go = scene.spawn(f"ac_{icao24}_point")
    point_renderer = PointRenderer(
        points=[Vec3.zero()],
        color=Color(0.2, 0.9, 1.0),   # Cyan
        size=150.0,
    )
    point_go.add_component(point_renderer)

    line_go = scene.spawn(f"ac_{icao24}_trail")
    line_renderer = LineRenderer.strip(
        points=[Vec3.zero(), Vec3.zero()],
        color=Color(0.1, 0.6, 0.9),
    )
    line_go.add_component(line_renderer)

    aircraft_gos[icao24] = {
        "point_go":       point_go,
        "point_renderer": point_renderer,
        "line_go":        line_go,
        "line_renderer":  line_renderer,
    }

def on_update(dt: float) -> None:
    global sim_time
    sim_time += timedelta(seconds=dt * SIM_SPEED)

    for icao24, group in grouped.items():
        row = get_row_at(group, sim_time)
        if row is None:
            continue

        alt_m = row["geoaltitude"]
        if pd.isna(alt_m):
            alt_m = 0.0

        pos = latlon_to_local(row["lat"], row["lon"], alt_m)

        gos = aircraft_gos[icao24]
        gos["point_go"].transform.position = pos

        trail = trails[icao24]
        trail.append(pos)

        if len(trail) >= 2:
            ordered = list(trail)
            n = len(ordered)
            fade_colors: list[Color] = []
            for i, _ in enumerate(ordered):
                t_val = i / (n - 1)                         # 0.0 … 1.0
                brightness = 0.15 + 0.85 * t_val            # 0.15 … 1.0
                fade_colors.append(Color(
                    0.1 * brightness,
                    0.6 * brightness,
                    1.0 * brightness,
                ))

            line_renderer = gos["line_renderer"]
            line_renderer.colors = None
            expanded_pts: list[Vec3]   = []
            expanded_cols: list[Color] = []
            for i in range(len(ordered) - 1):
                expanded_pts  += [ordered[i],       ordered[i + 1]]
                expanded_cols += [fade_colors[i],   fade_colors[i + 1]]

            line_renderer.points = expanded_pts
            line_renderer.colors = expanded_cols
            if line_renderer._vao is not None:
                glDeleteVertexArrays(1, [line_renderer._vao])
                glDeleteBuffers(1,     [line_renderer._vbo])
                line_renderer._vao = None
                line_renderer._vbo = None


engine.on_update = on_update


def on_gui() -> None:
    hud = engine.hud
    pos = engine.camera.position

    hud.text(f"X: {pos.x:.1f}  Y: {pos.y:.1f}  Z: {pos.z:.1f}", 10, 10, size=18, shadow=True)
    hud.text(f"SIM  {sim_time.strftime('%H:%M:%S')} UTC", 10, 35, size=16, shadow=True)

    hud.crosshair()


engine.on_gui = on_gui
engine.run()