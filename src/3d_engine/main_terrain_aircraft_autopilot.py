from pathlib import Path

import numpy as np
from pyproj import Transformer

from physics.aircraft_environment import AircraftEnvironment
from physics.aircraft_goal import AircraftGoal
from physics.aircraft_type import AircraftType
from physics.aircraft_kinematics_state import AircraftKinematicsState
from utils.airport import Airport
from utils.arinc_runway import ArincRunway
from utils.engine import Engine
from utils.lineRenderer import LineRenderer
from utils.color import Color
from utils.directionalLight import DirectionalLight
from utils.camera import OrbitCamera
from utils.pointRenderer import PointRenderer
from utils.clients.terrain_client import CopernicusDEMProvider
from utils.vec3 import Vec3
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
basePath: Path = Path("../data")
airport = Airport.load_from_folder(basePath, icao["icao"])

# instead of loading the runway data from the client via: AiracNetClient().get_airport_full(icao["icao"])
# we load the pre saved data from the base path
runways: list[ArincRunway] = airport.arinc_424_runway

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

scene.ambient_light = Color(0.25, 0.25, 0.3)

scene.directional_light = DirectionalLight(
    direction=Vec3(-1, -2, -1),
    color=Color(1.0, 0.95, 0.85),
    intensity=1.0
)

cam = OrbitCamera(distance=5.0)
cam.pitch = 15.0
engine.camera = cam

x, y = transformer.transform(icao["lon"], icao["lat"])
b737 = AircraftKinematicsState(
    x=x, y=y, h=airport.arinc_424_runway[0].runway_altitude * 0.3048 + 200,
    v=220.0, gamma=0.0, chi=0.0,
    mass=65000.0,
    C_D0=0.028,
    area=124.6,
)

constraints = AircraftType(
    cl_max=3.0
)

goal = AircraftGoal(
    x=b737.x + 10000.0,
    y=b737.y + 10000.0,
    h=1500.0
)
env  = AircraftEnvironment(state=b737, goal=goal, constraints=constraints, dt=0.1)

thrust      = 240000.0
load_factor = 1.0
roll        = 0.0

THRUST_STEP      = 20000.0
ROLL_STEP        = 25.0
LF_STEP          = 0.4
PHYS_DT          = 0.1
MAX_TRAIL_POINTS = 600
MAX_TRAIL        = True

trail_pts: list[Vec3] = []
_phys_acc: float      = 0.0

aircraft_go = scene.spawn("Aircraft")

x, z = geo.to_local_in_m(
    b737.x,
    b737.y,
)
y = geo.altitude_to_local(b737.h)

aircraft_go.transform.position = Vec3(x, y, z)

aircraft_go.add_component(PointRenderer(color=Color(1.0, 0.85, 0.0), size=120.0))

vel_go = scene.spawn("VelocityVector")
vel_lr = LineRenderer([Vec3.zero(), Vec3.zero()], color=Color(0.1, 1.0, 0.3))
vel_go.add_component(vel_lr)

trail_go = scene.spawn("Trail")
trail_lr = LineRenderer([Vec3.zero(), Vec3.zero()], color=Color(0.3, 0.75, 1.0))
trail_go.add_component(trail_lr)

goal_go = scene.spawn("Goal")

gx, gz = geo.to_local_in_m(
    goal.x,
    goal.y,
)
gy = geo.altitude_to_local(goal.h)

goal_go.transform.position = Vec3(gx, gy, gz)

goal_go.add_component(
    PointRenderer(
        color=Color(1.0, 0.0, 0.0),
        size=180.0
    )
)

def on_update(dt: float) -> None:
    global thrust, load_factor, roll
    global _phys_acc, trail_pts

    _phys_acc += dt

    while _phys_acc >= PHYS_DT:
        thrust, load_factor, roll = env.autopilot()

        thrust = np.clip(thrust, 0.0, 350000.0)
        roll = np.clip(roll, -60.0, 60.0)
        load_factor = np.clip(load_factor, 0.0, 3.0)

        env.apply_action(
            thrust=thrust,
            load_factor=load_factor,
            roll=roll,
        )

        _phys_acc -= PHYS_DT

    # world position of the aircraft
    x, z = geo.to_local_in_m(
        b737.x,
        b737.y,
    )
    y = geo.altitude_to_local(b737.h)

    wp = Vec3(x, y, z)

    # move aircraft marker
    aircraft_go.transform.position = wp

    # move cam to target
    cam.target = wp

    # append trajectory
    trail_pts.append(Vec3(wp.x, wp.y, wp.z))
    if len(trail_pts) > MAX_TRAIL_POINTS and MAX_TRAIL:
        trail_pts.pop(0)

    if len(trail_pts) >= 2:
        trail_lr.set_points_strip(trail_pts)

    # velocity vector
    v_raw = b737.velocity_vector()        # [East, North, Up] in m/s
    vel_tip = Vec3(
        wp.x + v_raw[0] * horizontal_scale,
        wp.y + v_raw[2] * vertical_scale,
        wp.z - v_raw[1] * horizontal_scale
    )
    vel_lr.set_points_strip([wp, vel_tip])

def on_gui() -> None:
    hud = engine.hud
    fps = engine._clock.get_fps() if engine._clock else 0.0

    hud.panel(10, 10, 285, 170, border_radius=8)
    hud.text("flight simulation",       20, 16, size=17, color=(255, 210, 60), shadow=True)

    # Separator
    hud.line(20, 38, 285, 38, color=(80, 80, 100, 180))

    v_kts  = b737.v   * 1.944          # m/s → knots
    h_ft   = b737.h   * 3.281          # m   → Fuß


    hud.text(f"Velocity.  {b737.v:6.1f} m/s  ({v_kts:5.0f} kts)",   20,  44, size=15)
    hud.text(f"height     {b737.h:6.0f} m    ({h_ft:6.0f} ft)",     20,  64, size=15)
    hud.text(f"chi        {b737.chi % 360:5.1f}°",                  20,  84, size=15)
    hud.text(f"gamma      {b737.gamma:+5.1f}°",                     20, 104, size=15)
    hud.line(20, 145, 285, 145, color=(80, 80, 100, 180))
    hud.text(f"FPS        {fps:5.0f}",                              20, 150, size=13, color=(160, 160, 180))

    px = hud.width - 295

    hud.panel(px, 10, 285, 160, border_radius=8)

    hud.text("autopilot",px + 10,16,size=17,color=(255, 210, 60),shadow=True)
    hud.line(px + 10,38,px + 275,38,color=(80, 80, 100, 180))

    hud.text(f"Thrust        {thrust / 1000:6.1f} kN",px + 10,46,size=15)
    hud.bar(px + 10,64,265,12,value=thrust / 350000.0,fg_color=(255, 160, 30, 220))

    hud.text(f"Roll          {roll:+6.1f}°",px + 10,84,size=15)
    hud.bar(px + 10,102,265,12,value=(roll + 35.0) / 70.0,fg_color=(80, 180, 255, 220))

    hud.text(f"Load Factor   {load_factor:4.2f} g",px + 10,122,size=15)
    hud.bar(px + 10, 140, 265, 12, value=load_factor / 2.5, fg_color=(100, 220, 120, 220))


engine.on_gui = on_gui
engine.on_update = on_update
engine.run()