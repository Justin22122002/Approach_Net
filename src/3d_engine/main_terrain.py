from pathlib import Path

from pyproj import Transformer

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

icao_vancouver = {
    "icao": "CYVR",
    "lat": 49.1967,
    "lon": -123.1815
}

icao_frankfurt = {
    "icao": "EDDF",
    "lat": 50.0379,
    "lon": 8.5622
}

icao = icao_frankfurt

crs_to: str = GeoTransformer.utm_epsg(icao["lat"], icao["lon"])

transformer: Transformer = Transformer.from_crs(
    crs_from="EPSG:4326",
    crs_to=crs_to,
    always_xy=True
)

# --- load airport data section ---
basePath: Path = Path("../data")
frankfurt = Airport.load_from_folder(basePath, icao["icao"])

# instead of loading the runway data from the client via: AiracNetClient().get_airport_full(icao["icao"])
# we load the pre saved data from the base path
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
vertical_scale = 0.01

geo = GeoTransformer(
    lat=0.0,
    lon=0.0,
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

def on_gui():
    hud = engine.hud
    pos = engine.camera.position
    hud.text(f"X: {pos.x:.1f}  Y: {pos.y:.1f}  Z: {pos.z:.1f}", 10, 10, size=18, shadow=True)
    hud.crosshair()

engine.on_gui = on_gui
engine.run()