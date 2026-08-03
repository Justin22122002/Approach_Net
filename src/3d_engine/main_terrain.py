from pathlib import Path

from pyproj import Transformer

from utils.clients.open_airport_map_client import get_bbox_simple
from utils.texture import Texture
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
from utils.lineRenderer import LineRenderer

icao_vancouver = {
    "icao": "CYVR", # CYYZ
}

icao_frankfurt = {
    "icao": "EDDF",
}

icao = icao_vancouver

# --- load airport data section ---
basePath: Path = Path("../data")
airport = Airport.load_from_folder(basePath, icao["icao"])

radius_km = 25.0  # depending on the airport choose -> 20 - 30 km
bounds = get_bbox_simple(airport.lat, airport.lon, radius_km)

crs_to: str = GeoTransformer.utm_epsg(airport.lat, airport.lon)

transformer: Transformer = Transformer.from_crs(
    crs_from="EPSG:4326",
    crs_to=crs_to,
    always_xy=True
)

# instead of loading the runway data from the client via: AiracNetClient().get_airport_full(icao["icao"])
# we load the pre saved data from the base path
runways: list[ArincRunway] = airport.arinc_424_runway

email = "justin.groh@study.thws.de"
password = "Copernicus2026!"

delta = 0.2

LAT_MIN = airport.lat - delta
LAT_MAX = airport.lat + delta
LON_MIN = airport.lon - delta
LON_MAX = airport.lon + delta

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
vertical_scale = 0.01

geo = GeoTransformer(
    lat=airport.lat,
    lon=airport.lon,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    transformer=transformer
)

sat_array = provider.get_satellite_image(
    lat_min=LAT_MIN,
    lon_min=LON_MIN,
    lat_max=LAT_MAX,
    lon_max=LON_MAX,
    width=2048,
    height=2048
)
sat_texture = Texture.from_array(sat_array)

terrain_project_mesh = TerrainProjectMesh(
    grid=grid,
    lat_min=LAT_MIN,
    lon_min=LON_MIN,
    lat_max=LAT_MAX,
    lon_max=LON_MAX,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    transformer=transformer,
    texture=sat_texture
)

engine = Engine(
    1280,
    720,
    f"Terrain - {icao['icao']}"
)

scene = engine.scene

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
        transformer=transformer,
    )

    builder.spawn(scene, geo)

scene.ambient_light = Color(0.25, 0.25, 0.3)

scene.directional_light = DirectionalLight(
    direction=Vec3(-1, -2, -1),
    color=Color(1.0, 0.95, 0.85),
    intensity=1.0
)

engine.camera = FlyCamera(speed=100.0)

engine.camera.position = Vec3(
    rw.x,
    geo.altitude_to_local(terrain_project_mesh.altitude) + 50,
    rw.z
)

bbox_lat_min, bbox_lon_min, bbox_lat_max, bbox_lon_max = bounds  # get_bbox_simple: (lat_min, lon_min, lat_max, lon_max)

bbox_corners_latlon = [
    (bbox_lat_min, bbox_lon_min),
    (bbox_lat_min, bbox_lon_max),
    (bbox_lat_max, bbox_lon_max),
    (bbox_lat_max, bbox_lon_min),
    (bbox_lat_min, bbox_lon_min),
]

bbox_height = geo.altitude_to_local(terrain_project_mesh.altitude + 100.0)

bbox_points = []
for lat, lon in bbox_corners_latlon:
    local_x, local_z = geo.to_local(lat, lon)
    bbox_points.append(Vec3(local_x, bbox_height, local_z))

bbox_go = scene.spawn("DataBoundingBox")
bbox_lr = LineRenderer.strip(bbox_points, color=Color(1.0, 1.0, 0.0))
bbox_go.add_component(bbox_lr)

def on_gui():
    hud = engine.hud
    pos = engine.camera.position
    hud.text(f"X: {pos.x:.1f}  Y: {pos.y:.1f}  Z: {pos.z:.1f}", 10, 10, size=18, shadow=True)
    hud.crosshair()

engine.on_gui = on_gui
engine.run()