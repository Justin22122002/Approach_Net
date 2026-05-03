from utils.engine import Engine
from utils.meshRenderer import MeshRenderer
from utils.terrain_client import CopernicusDEMProvider
from utils.vec3 import Vec3
from utils.color import Color
from utils.directionalLight import DirectionalLight
from utils.camera import FlyCamera
from utils.mesh import Mesh
from utils.material import Material

password = "Copernicus2026!"
email = "justin.groh@study.thws.de"

icao_data = {
    "icao": "EDDF",
    "lat": 50.0379,
    "lon": 8.5622
}

provider = CopernicusDEMProvider(email, password)
delta = 0.05

LAT_MIN = icao_data["lat"] - delta
LAT_MAX = icao_data["lat"] + delta
LON_MIN = icao_data["lon"] - delta
LON_MAX = icao_data["lon"] + delta

grid = provider.get_terrain(
    lat_min=LAT_MIN,
    lon_min=LON_MIN,
    lat_max=LAT_MAX,
    lon_max=LON_MAX,
    width=255,
    height=255
)

engine = Engine(800, 600, f"Terrain – {icao_data['icao']}")
scene  = engine.scene

terrain_mesh = Mesh.from_heightmap(
    grid,
    size=50.0,
    lat_min=LAT_MIN,
    lon_min=LON_MIN,
    lat_max=LAT_MAX,
    lon_max=LON_MAX
)

terrain_go = scene.spawn("Terrain")
terrain_mat = Material(color=Color(0.4, 0.55, 0.3))
terrain_go.add_component(MeshRenderer(terrain_mesh, terrain_mat))
terrain_go.transform.position = Vec3(0, 0, 0)

scene.ambient_light = Color(0.25, 0.25, 0.3)
scene.directional_light = DirectionalLight(
    direction=Vec3(-1, -2, -1),
    color=Color(1.0, 0.95, 0.85),
    intensity=1.0
)

engine.camera = FlyCamera(speed=8.0)
engine.camera.position = Vec3(0, 8, 20)

engine.run()