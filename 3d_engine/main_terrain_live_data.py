import time
from pathlib import Path
from typing import Any

import pandas as pd
from pyproj import Transformer

from utils.aircraft import AircraftData
from utils.airport import Airport
from utils.arinc_runway import ArincRunway
from utils.engine import Engine
from utils.meshRenderer import MeshRenderer
from utils.material import Material
from utils.color import Color
from utils.directionalLight import DirectionalLight
from utils.camera import FlyCamera
from utils.open_sky_rest_client import OpenSkyRestClient
from utils.pointRenderer import PointRenderer
from utils.terrain_client import CopernicusDEMProvider
from utils.vec3 import Vec3
from utils_project.aircraft_project_mesh import AircraftProjectMesh
from utils_project.geo_transformer import GeoTransformer
from utils_project.runway_project_mesh import RunwayProjectMesh
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
basePath: Path = Path("../src/data")
airport_data = Airport.load_from_folder(basePath, icao["icao"])

# instead of loading the runway data from the client via: AiracNetClient().get_airport_full(icao["icao"])
# we load the pre saved data from the base path
runways: list[ArincRunway] = airport_data.arinc_424_runway

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

terrain_mesh = terrain_project_mesh.generate_mesh()

geo = GeoTransformer(
    lat=0.0,
    lon=0.0,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    transformer=transformer
)

dx, dz = geo.to_local(terrain_project_mesh.lat, terrain_project_mesh.lon)

terrain_position = Vec3(
    dx,
    geo.altitude_to_local(terrain_project_mesh.altitude),
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
rw = None
for runway in runways:
    runway_mesh_builder = RunwayProjectMesh(
        lat=runway.latitude,
        lon=runway.longitude,
        length_ft=runway.length,
        width_ft=runway.width,
        bearing_deg=runway.bearing,
        altitude_ft=runway.runway_altitude,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        displaced_threshold_ft=runway.displaced_threshold,
        transformer=transformer
    )

    runway_mesh = runway_mesh_builder.generate_mesh()

    cx, cz = geo.to_local(runway_mesh_builder.lat, runway_mesh_builder.lon)
    cy = geo.altitude_to_local(runway_mesh_builder.altitude)

    runway_go = scene.spawn(f"Runway_{runway.runway_identifier}")

    runway_go.transform.position = Vec3(cx, cy, cz)

    rw = Vec3(cx, cy, cz)
    runway_go.add_component(
        MeshRenderer(
            runway_mesh,
            Material(color=Color(1.0, 1.0, 1.0))
        )
    )

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

### fetch live aircraft data
half_edge_km = 20
opensky_client = OpenSkyRestClient()
last_update_time = 0
aircraft_points: dict[str, Any] = {}

def on_update(dt: float) -> None:
    global last_update_time

    current_time = time.time()

    if current_time - last_update_time >= 5:
        flight_df = opensky_client.get_flights_in_area(half_edge_km, icao["lat"], icao["lon"])
        last_update_time = current_time
        print(f"Flüge aktualisiert: {len(flight_df) if flight_df is not None else 0}")

        aircraft_objects: list[AircraftProjectMesh] = []
        if flight_df is not None and not flight_df.empty:
            for idx, row in flight_df.iterrows():
                # todo: move mapping in an extra class
                icao24 = row.get("icao24")
                callsign = row.get("callsign")
                origin_country = row.get("origin_country")
                last_position = row.get("last_position")
                timestamp = row.get("timestamp")
                longitude = row.get("longitude")
                latitude = row.get("latitude")
                altitude = row.get("altitude")
                onground = row.get("onground")
                groundspeed = row.get("groundspeed")
                track = row.get("track")
                vertical_rate = row.get("vertical_rate")
                sensors = row.get("sensors")
                geoaltitude = row.get("geoaltitude")
                squawk = row.get("squawk")
                spi = row.get("spi")
                position_source = row.get("position_source")
                category = row.get("category")

                if pd.isna(geoaltitude):
                     geoaltitude = airport_data.arinc_424_runway[0].runway_altitude * 0.3048

                data = AircraftData(
                    icao24=icao24,
                    callsign=callsign,
                    origin_country=origin_country,
                    last_position=last_position,
                    timestamp=timestamp,
                    longitude=longitude,
                    latitude=latitude,
                    altitude=altitude,
                    onground=onground,
                    groundspeed=groundspeed,
                    track=track,
                    vertical_rate=vertical_rate,
                    sensors=sensors,
                    geoaltitude=geoaltitude,
                    squawk=squawk,
                    spi=spi,
                    position_source=position_source,
                    category=category,
                )
                aircraft_object = AircraftProjectMesh(
                    aircraft_data=data,
                    horizontal_scale=horizontal_scale,
                    vertical_scale=vertical_scale,
                    transformer=transformer,
                )

                aircraft_objects.append(aircraft_object)

                if aircraft_object.lat is not None and aircraft_object.lon is not None:

                    x, z = geo.to_local(aircraft_object.lat, aircraft_object.lon)
                    y = geo.altitude_to_local(aircraft_object.altitude)

                    if icao24 not in aircraft_points:
                        aircraft_go = scene.spawn(f"Aircraft_{icao24}")

                        color: Color = Color(0.0, 1.0, 0.0)
                        if (
                                aircraft_object.aircraft_data.onground
                                and aircraft_object.aircraft_data.groundspeed == 0.0
                        ):
                            color = Color(0.5, 0.5, 0.5)

                        elif (
                                aircraft_object.aircraft_data.onground
                                and aircraft_object.aircraft_data.groundspeed > 0.0
                        ):
                            color = Color(1.0, 1.0, 0.0)

                        aircraft_go.add_component(
                            PointRenderer(
                                color=color,
                                size=200.0
                            )
                        )

                        aircraft_points[icao24] = aircraft_go

                    aircraft_points[icao24].transform.position = Vec3(x, y, z)


def on_gui():
    hud = engine.hud
    pos = engine.camera.position
    hud.text(f"X: {pos.x:.1f}  Y: {pos.y:.1f}  Z: {pos.z:.1f}", 10, 10, size=18, shadow=True)
    hud.crosshair()

engine.on_gui = on_gui
engine.on_update = on_update
engine.run()