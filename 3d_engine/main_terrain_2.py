import math
from pyproj import Transformer, Geod

from utils.engine import Engine
from utils.meshRenderer import MeshRenderer
from utils.terrain_client import CopernicusDEMProvider
from utils.vec3 import Vec3
from utils.color import Color
from utils.directionalLight import DirectionalLight
from utils.camera import FlyCamera
from utils.mesh import Mesh
from utils.material import Material
from utils.component_helpers import quick_box
from src.utils.airac_net_client import AiracNetClient
from src.utils.arinc_runway import ArincRunway

ICAO = "CYVR"
LAT = 49.1967
LON = -123.1815
RADIUS_KM = 20.0
WORLD_SIZE = 50.0

ICAO = "EDDF"
LAT = 50.0379
LON = 8.5622
RADIUS_KM = 20.0
WORLD_SIZE = 50.0

geod = Geod(ellps="WGS84")

def latlon_to_utm(lat, lon):
    return wgs84_to_utm.transform(lon, lat)

radius_m = RADIUS_KM * 1000

lon1, lat1, _ = geod.fwd(LON, LAT, 225, radius_m)
lon2, lat2, _ = geod.fwd(LON, LAT, 45, radius_m)

LAT_MIN = min(lat1, lat2)
LAT_MAX = max(lat1, lat2)
LON_MIN = min(lon1, lon2)
LON_MAX = max(lon1, lon2)

utm_zone = int((LON + 180) / 6) + 1
epsg_utm = 32600 + utm_zone
wgs84_to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True)

origin_x, origin_y = latlon_to_utm(LAT, LON)

corner_min_x, corner_min_y = latlon_to_utm(LAT_MIN, LON_MIN)
corner_max_x, corner_max_y = latlon_to_utm(LAT_MAX, LON_MAX)

real_w_m = corner_max_x - corner_min_x
real_h_m = corner_max_y - corner_min_y
real_extent_m = max(real_w_m, real_h_m)

m_to_wu = WORLD_SIZE / real_extent_m


def geo_to_world(lat, lon):
    ux, uy = latlon_to_utm(lat, lon)
    x = (ux - origin_x) * m_to_wu
    z = -(uy - origin_y) * m_to_wu
    return x, z


provider = CopernicusDEMProvider(
    "justin.groh@study.thws.de",
    "Copernicus2026!"
)

grid = provider.get_terrain(
    lat_min=LAT_MIN,
    lon_min=LON_MIN,
    lat_max=LAT_MAX,
    lon_max=LON_MAX,
    width=255,
    height=255
)

runways: list[ArincRunway] = AiracNetClient().get_airport_full(ICAO)
print(f"ARINC Runways: {len(runways)}")


engine = Engine(800, 600, f"Terrain – {ICAO}")
scene = engine.scene


# Terrain Mesh
terrain_go = scene.spawn("Terrain")
terrain_go.add_component(MeshRenderer(
    Mesh.from_heightmap(
        grid,
        size=WORLD_SIZE,
        lat_min=LAT_MIN,
        lon_min=LON_MIN,
        lat_max=LAT_MAX,
        lon_max=LON_MAX,
    ),
    Material(color=Color(0.4, 0.55, 0.3))
))

for runway in runways:
    start, end = runway.centerline_coordinates

    mid_lat = (start.lat + end.lat) / 2
    mid_lon = (start.lon + end.lon) / 2

    cx, cz = geo_to_world(mid_lat, mid_lon)
    cy = (runway.runway_altitude * 0.3048) * m_to_wu

    corners = runway.polygon_coordinates

    _, _, rwy_len_m = geod.inv(start.lon, start.lat, end.lon, end.lat)
    _, _, rwy_wid_m = geod.inv(
        corners[0].lon, corners[0].lat,
        corners[3].lon, corners[3].lat
    )

    rwy_len_wu = rwy_len_m / real_extent_m * WORLD_SIZE
    rwy_wid_wu = rwy_wid_m / real_extent_m * WORLD_SIZE

    bearing, _, _ = geod.inv(start.lon, start.lat, end.lon, end.lat)
    bearing = bearing % 360

    rwy_go = quick_box(
        scene,
        position=Vec3(cx, cy + 0.02, cz),
        size=Vec3(rwy_wid_wu, 0.05, rwy_len_wu),
        color=Color(0.25, 0.25, 0.25),
        name=f"Runway_{runway.runway_identifier}",
        auto_collision=False
    )

    rwy_go.transform.rotation = Vec3(0, -bearing, 0)

    print(f"  {runway.runway_identifier}: "
          f"len={rwy_len_m:.0f}ft ({rwy_len_wu:.2f} wu)  "
          f"wid={rwy_wid_m:.0f}ft ({rwy_wid_wu:.2f} wu)  "
          f"brg={bearing:.1f}°")


scene.ambient_light = Color(0.25, 0.25, 0.3)
scene.directional_light = DirectionalLight(
    direction=Vec3(-1, -2, -1),
    color=Color(1.0, 0.95, 0.85),
    intensity=1.0
)


engine.camera = FlyCamera(speed=8.0)
engine.camera.position = Vec3(0, 10, 25)

engine.run()