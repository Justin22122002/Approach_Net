from utils.engine import Engine
from utils.meshRenderer import MeshRenderer
from utils.terrain_client import CopernicusDEMProvider
from utils.material import Material
from utils.color import Color
from utils.directionalLight import DirectionalLight
from utils.camera import FlyCamera
from utils.vec3 import Vec3

from utils_project.project_mesh import (
    GeoTransformer,
    TerrainProjectMesh
)

email = "justin.groh@study.thws.de"
password = "Copernicus2026!"

icao = {
    "icao": "EDDF",
    "lat": 50.0379,
    "lon": 8.5622
}

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
vertical_scale = 0.03

terrain_project_mesh = TerrainProjectMesh(
    grid=grid,
    lat_min=LAT_MIN,
    lon_min=LON_MIN,
    lat_max=LAT_MAX,
    lon_max=LON_MAX,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale
)

terrain_mesh = terrain_project_mesh.generate_mesh()

geo = GeoTransformer(
    lat=icao["lat"],
    lon=icao["lon"],
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale
)

dx, dz = geo.to_local(LAT_MIN, LON_MIN)

terrain_position = Vec3(
    dx,
    geo.altitude_to_local(terrain_project_mesh.base_altitude),
    dz
)

engine = Engine(
    1280,
    720,
    f"Terrain - {icao['icao']}"
)

scene = engine.scene

terrain_go = scene.spawn("Terrain")

terrain_go.add_component(
    MeshRenderer(
        terrain_mesh,
        Material(
            color=Color(0.4, 0.55, 0.3)
        )
    )
)

terrain_go.transform.position = terrain_position

scene.ambient_light = Color(0.25, 0.25, 0.3)

scene.directional_light = DirectionalLight(
    direction=Vec3(-1, -2, -1),
    color=Color(1.0, 0.95, 0.85),
    intensity=1.0
)

engine.camera = FlyCamera(speed=10.0)

engine.camera.position = Vec3(
    terrain_position.x,
    geo.altitude_to_local(terrain_project_mesh.base_altitude) + 50,
    terrain_position.z
)

engine.run()