import math
import time
from typing import Any

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
from src.utils.open_sky_rest_client import OpenSkyRestClient

from utils.aircraft import AircraftData, Aircraft


# CONFIG
ICAO = "EDDF"
LAT = 50.0379
LON = 8.5622
RADIUS_KM = 20.0
WORLD_SIZE = 50.0

geod = Geod(ellps="WGS84")

utm_zone = int((LON + 180) / 6) + 1
epsg_utm = 32600 + utm_zone

wgs84_to_utm = Transformer.from_crs(
    "EPSG:4326",
    f"EPSG:{epsg_utm}",
    always_xy=True
)

def latlon_to_utm(lat: float, lon: float):
    return wgs84_to_utm.transform(lon, lat)


# BOUNDS
radius_m = RADIUS_KM * 1000

lon1, lat1, _ = geod.fwd(LON, LAT, 225, radius_m)
lon2, lat2, _ = geod.fwd(LON, LAT, 45, radius_m)

LAT_MIN, LAT_MAX = min(lat1, lat2), max(lat1, lat2)
LON_MIN, LON_MAX = min(lon1, lon2), max(lon1, lon2)

origin_x, origin_y = latlon_to_utm(LAT, LON)

corner_min_x, corner_min_y = latlon_to_utm(LAT_MIN, LON_MIN)
corner_max_x, corner_max_y = latlon_to_utm(LAT_MAX, LON_MAX)

real_extent_m = max(
    corner_max_x - corner_min_x,
    corner_max_y - corner_min_y
)

m_to_wu = WORLD_SIZE / real_extent_m


def geo_to_world(lat: float, lon: float):
    ux, uy = latlon_to_utm(lat, lon)
    x = (ux - origin_x) * m_to_wu
    z = -(uy - origin_y) * m_to_wu
    return x, z


def safe_float(v):
    try:
        if v is None:
            return None
        if v != v:  # NaN check
            return None
        return float(v)
    except Exception:
        return None


HEIGHT_SCALE = WORLD_SIZE / real_extent_m

def aircraft_to_world(lat, lon, geoalt_m):
    x, z = geo_to_world(lat, lon)

    y = geoalt_m * HEIGHT_SCALE

    return Vec3(x, y, z)


# TERRAIN
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


# RUNWAYS
runways: list[ArincRunway] = AiracNetClient().get_airport_full(ICAO)
print(f"Runways: {len(runways)}")


# ENGINE
engine = Engine(800, 600, f"Terrain + Aircraft – {ICAO}")
scene = engine.scene


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


# RUNWAYS
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
    bearing %= 360

    rwy = quick_box(
        scene,
        position=Vec3(cx, cy + 0.02, cz),
        size=Vec3(rwy_wid_wu, 0.05, rwy_len_wu),
        color=Color(0.25, 0.25, 0.25),
        name=f"Runway_{runway.runway_identifier}",
        auto_collision=False
    )

    rwy.transform.rotation = Vec3(0, -bearing, 0)


# LIGHT + CAMERA
scene.ambient_light = Color(0.25, 0.25, 0.3)

scene.directional_light = DirectionalLight(
    direction=Vec3(-1, -2, -1),
    color=Color(1.0, 0.95, 0.85),
    intensity=1.0
)

engine.camera = FlyCamera(speed=8.0)
engine.camera.position = Vec3(0, 10, 25)


# AIRCRAFT SYSTEM
opensky_client = OpenSkyRestClient()

aircraft_cache: dict[str, Any] = {}
aircraft_objects: list[Aircraft] = []

last_update_time = 0.0
UPDATE_INTERVAL = 5.0
half_edge_km = 10


def update_aircraft():
    global last_update_time, aircraft_objects

    now = time.time()
    if now - last_update_time < UPDATE_INTERVAL:
        return

    last_update_time = now

    df = opensky_client.get_flights_in_area(
        half_edge_km,
        LAT,
        LON
    )

    print(f"Flüge aktualisiert: {len(df) if df is not None else 0}")

    aircraft_objects = []

    if df is None or df.empty:
        return

    for _, row in df.iterrows():

        icao24 = row.get("icao24")
        lat = safe_float(row.get("latitude"))
        lon = safe_float(row.get("longitude"))
        geoalt = row.get("geoaltitude")

        if lat is None or lon is None or icao24 is None:
            continue

        if geoalt is None or str(geoalt) == "<NA>":
            geoalt = 10.0 # set to 10, so they dont get rendered in the ground 

        data = AircraftData(
            icao24=icao24,
            callsign=row.get("callsign"),
            origin_country=row.get("origin_country"),
            last_position=row.get("last_position"),
            timestamp=row.get("timestamp"),
            longitude=lon,
            latitude=lat,
            altitude=row.get("altitude"),
            onground=row.get("onground"),
            groundspeed=row.get("groundspeed"),
            track=row.get("track"),
            vertical_rate=row.get("vertical_rate"),
            sensors=row.get("sensors"),
            geoaltitude=geoalt,
            squawk=row.get("squawk"),
            spi=row.get("spi"),
            position_source=row.get("position_source"),
            category=row.get("category"),
        )

        aircraft_objects.append(Aircraft(data))


def sync_aircraft_to_scene():
    global aircraft_cache

    alive_ids = set()

    for aircraft in aircraft_objects:

        data = aircraft.data

        if data.latitude is None or data.longitude is None:
            continue

        icao = data.icao24
        if icao is None:
            continue

        alive_ids.add(icao)

        pos = aircraft_to_world(
            data.latitude,
            data.longitude,
            data.geoaltitude or 0.0
        )

        if icao not in aircraft_cache:
            go = scene.spawn(f"Aircraft_{icao}")
            go.add_component(
                MeshRenderer(
                    Mesh.sphere(radius=0.1),
                    Material(color=Color.red())
                )
            )
            aircraft_cache[icao] = go

        aircraft_cache[icao].transform.position = pos

    to_delete = set(aircraft_cache.keys()) - alive_ids

    for icao in to_delete:
        go = aircraft_cache[icao]
        scene.destroy(go)
        del aircraft_cache[icao]


def on_update(dt: float):
    update_aircraft()
    sync_aircraft_to_scene()

engine.on_update = on_update

engine.run()