from pathlib import Path
import re
import numpy as np
from PIL import Image
from pyproj import Transformer

from utils.airport import Airport
from utils.arinc_runway import ArincRunway
from utils.clients.airport_client import OpenStreetMapAirportClient
from utils.clients.osm_water_client import OpenStreetMapWaterClient

from engine.utils.engine import Engine
from engine.utils.color import Color
from engine.utils.camera import FlyCamera
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
from engine.utils_project.water_texture_renderer import WaterTextureRenderer

from dotenv import load_dotenv

load_dotenv()

ICAO = "CYVR"

WINDOW_W = 1280
WINDOW_H = 720

LOCAL_SCALE = 0.01

USE_SATELLITE_TEXTURE = False
TERRAIN_TEX_SIZE = 8192

PREFERRED_BEARING_DEG = -90.0

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

engine = Engine(WINDOW_W, WINDOW_H, f"Airport Viewer - {ICAO}")
scene = engine.scene


def _runway_group_key(identifier: str) -> frozenset:
    n, suf = int(re.match(r"^(\d{1,2})([LRC]?)$", identifier.upper()).groups()[0]), \
             re.match(r"^(\d{1,2})([LRC]?)$", identifier.upper()).group(2)
    n_recip = n + 18 if n <= 18 else n - 18
    suf_recip = {"L": "R", "R": "L", "C": "C", "": ""}[suf]
    return frozenset({(n, suf), (n_recip, suf_recip)})


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

for runway in runways_one_per_direction:
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

airport_elev_m = runways[0].runway_altitude * 0.3048 if runways else 0.0
cx, cz = geo.to_local(airport.lat, airport.lon)
cy = geo.altitude_to_local(airport_elev_m)

camera = FlyCamera(speed=100.0)
camera.position = Vec3(cx, cy + 300, cz + 800)
engine.camera = camera

engine.run()