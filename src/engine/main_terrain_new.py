from pathlib import Path

from pyproj import Transformer

from utils.airport import Airport
from utils.arinc_runway import ArincRunway
from engine.utils.engine import Engine
from engine.utils.color import Color
from engine.utils.directionalLight import DirectionalLight
from engine.utils.camera import FlyCamera
from engine.utils.vec3 import Vec3
from engine.utils_project.approach_project_mesh import ApproachProjectMesh
from engine.utils_project.geo_transformer import GeoTransformer
from engine.utils_project.rnav_lpv_funnel_project_mesh import RnavLpvFunnelProjectMesh
from engine.utils_project.runway_project_mesh import RunwayProjectMesh
from engine.utils_project.separation_geometry import SeparationGeometry
from engine.utils_project.separation_project_mesh import SeparationProjectMesh
from engine.utils_project.taxiway_project_mesh import TaxiwayProjectMesh
from engine.utils_project.terrain_project_mesh import TerrainProjectMesh

icao_vancouver = {
    "icao": "CYVR",
}

icao_frankfurt = {
    "icao": "EDDF",
}

icao = icao_frankfurt

# --- load airport data section ---
basePath: Path = Path("../data")
airport = Airport.load_from_folder(basePath, icao["icao"])

crs_to: str = GeoTransformer.utm_epsg(airport.lat, airport.lon)

transformer: Transformer = Transformer.from_crs(
    crs_from="EPSG:4326",
    crs_to=crs_to,
    always_xy=True
)

# instead of loading the runway data from the client via: AiracNetClient().get_airport_full(icao["icao"])
# we load the pre saved data from the base path
runways: list[ArincRunway] = airport.arinc_424_runway

horizontal_scale = 0.01
vertical_scale = 0.01

geo = GeoTransformer(
    lat=airport.lat,
    lon=airport.lon,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    transformer=transformer
)

terrain_project_mesh = TerrainProjectMesh(
    grid=airport.terrain.elevation_grid,
    lat_min=airport.terrain.lat_min,
    lon_min=airport.terrain.lon_min,
    lat_max=airport.terrain.lat_max,
    lon_max=airport.terrain.lon_max,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    transformer=transformer
)

engine = Engine(
    1280,
    720,
    f"Terrain - {icao['icao']}"
)

scene = engine.scene

terrain_project_mesh.spawn(scene=scene, geo=geo)

rw = None
spawned_separation = False
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

        funnel_mesh = RnavLpvFunnelProjectMesh(
            approach=approach,
            transformer=transformer,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            runway=runway,
        )
        funnel_mesh.spawn(scene, geo)

        _e, faf_fix, _rw = approach.get_final_approach_fixes()
        if faf_fix is not None and not spawned_separation:
            faf_x, faf_y = transformer.transform(faf_fix.longitude, faf_fix.latitude)
            SeparationProjectMesh(
                center_xy=(faf_x, faf_y),
                center_h=faf_fix.altitude * 0.3048,
                transformer=transformer,
                horizontal_scale=horizontal_scale,
                vertical_scale=vertical_scale,
                separation=SeparationGeometry(),
            ).spawn(scene, geo)
            spawned_separation = True

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

def on_gui():
    hud = engine.hud
    pos = engine.camera.position
    hud.text(f"X: {pos.x:.1f}  Y: {pos.y:.1f}  Z: {pos.z:.1f}", 10, 10, size=18, shadow=True)
    hud.crosshair()

engine.on_gui = on_gui
engine.run()