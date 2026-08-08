import os
from pathlib import Path
import numpy as np
from pyproj import Transformer, Geod
from utils.clients.open_airport_map_client import get_bbox_simple

from physics.aircraft_environment_casadi import AircraftEnvironmentCasadi
from physics.aircraft_dynamics import AircraftDynamics
from physics.aircraft_kinematics_state_casadi import AircraftKinematicsStateCasadi
from physics.landing_optimizer import LandingTrajectoryOptimizer, state_to_casadi
from utils.airport import Airport
from utils.arinc_runway import ArincRunway
from utils.engine import Engine
from utils.color import Color
from utils.directionalLight import DirectionalLight
from utils.camera import OrbitCamera
from utils.pointRenderer import PointRenderer
from utils.lineRenderer import LineRenderer
from utils.clients.terrain_client import CopernicusDEMProvider
from utils.vec3 import Vec3
from utils_project.geo_transformer import GeoTransformer
from utils_project.runway_project_mesh import RunwayProjectMesh
from utils_project.terrain_project_mesh import TerrainProjectMesh
from dotenv import load_dotenv

load_dotenv()

icao_vancouver = {
    "icao": "CYVR",
}

icao_frankfurt = {
    "icao": "EDDF",
}

icao = icao_vancouver

# --- load airport data section ---
basePath: Path = Path("../data")
airport = Airport.load_from_folder(basePath, icao["icao"])

crs_to      = GeoTransformer.utm_epsg(airport.lat, airport.lon)
transformer = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)

radius_km = 25.0  # depending on the airport choose -> 20 - 30 km
bounds = get_bbox_simple(airport.lat, airport.lon, radius_km)

runways: list[ArincRunway] = airport.arinc_424_runway

provider = CopernicusDEMProvider(
   os.environ["COPERNICUS_USERNAME"],
   os.environ["COPERNICUS_PASSWORD"],
)
delta    = 0.2
grid     = provider.get_terrain(
    airport.lat - delta, airport.lon - delta,
    airport.lat + delta, airport.lon + delta,
    1024, 1024,
)

horizontal_scale = 0.01
vertical_scale   = 0.01
geo = GeoTransformer(airport.lat, airport.lon, horizontal_scale, vertical_scale, transformer)

engine = Engine(1280, 720, "CasADi Landing")
scene  = engine.scene
TerrainProjectMesh(
    grid=grid,
    lat_min=airport.lat - delta, lon_min=airport.lon - delta,
    lat_max=airport.lat + delta, lon_max=airport.lon + delta,
    horizontal_scale=horizontal_scale, vertical_scale=vertical_scale,
    transformer=transformer,
).spawn(scene, geo)

scene.ambient_light     = Color(0.25, 0.25, 0.3)
scene.directional_light = DirectionalLight(Vec3(-1,-2,-1), Color(1,0.95,0.85), 1)

cam = OrbitCamera(distance=6)
engine.camera = cam

for runway in runways:
    RunwayProjectMesh(
        runway=runway,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        transformer=transformer) \
        .spawn(scene, geo)

runway         = airport.arinc_424_runway[0]
goal_x, goal_y = transformer.transform(runway.longitude, runway.latitude)
goal_h         = runway.runway_altitude * 0.3048
goal_chi_deg   = float(runway.bearing)

print(f"Runway {runway}  χ={goal_chi_deg:.1f}°  "
      f"threshold ({goal_x:.0f}, {goal_y:.0f}, {goal_h:.1f} m)")

goal_go = scene.spawn("Goal")
goal_go.add_component(PointRenderer(color=Color(1, 0, 0), size=140))
gx, gz = geo.to_local_in_m(goal_x, goal_y)
goal_go.transform.position = Vec3(gx, geo.altitude_to_local(goal_h), gz)

constraints = AircraftDynamics(
    cl_max=3.096,
    mass=77800.0,
    C_D0=0.0200,
    area=128.0,
    k=0.0410,
    max_thrust=266900,
    max_load_factor=1.5,
    min_load_factor=0.5,
    max_roll_angle=60,
)

geod = Geod(ellps="WGS84")
approach_bearing_deg = goal_chi_deg % 360.0

start_lon, start_lat, _ = geod.fwd(
    airport.lon, airport.lat,
    az=approach_bearing_deg,
    dist=radius_km * 1000.0,
)
x0_start, y0_start = transformer.transform(start_lon, start_lat)

START_ALT_AGL_M = 3000.0

state = AircraftKinematicsStateCasadi(
    x=x0_start,
    y=y0_start,
    h=goal_h + START_ALT_AGL_M,
    v=90.0,
    gamma=-2.5,
    chi=-goal_chi_deg,
    aircraft_type=constraints,
)

print("Optimising trajectory with IPOPT (free end-time) …")

N_casadi = 100
optimizer = LandingTrajectoryOptimizer(
    state           = state,
    N               = N_casadi,
    T_min           = 60.0,
    T_max           = 600.0,
)

x0_vec = state_to_casadi(state)
goal   = np.array([goal_x, goal_y, goal_h])

X_opt, U_opt, T_opt, _ = optimizer.solve(
    x0_vec,
    goal,
    goal_chi_deg,
)

dt_casadi = T_opt / N_casadi

print(f"Optimisation done.  T = {T_opt:.1f} s  dt = {dt_casadi:.3f} s")
print(f"  pos  ({X_opt[0,-1]:.0f}, {X_opt[1,-1]:.0f}, {X_opt[2,-1]:.1f}) m")
print(f"  v    {float(X_opt[3,-1]):.1f} m/s")

# Re-use AircraftEnvironmentCasadi to replay the optimized trajectory.
# float() on state fields: after apply_action the attributes become CasADi DM.
env = AircraftEnvironmentCasadi(
    state       = state,
    goal        = None,
    dt          = dt_casadi,
)

print(f"Environment ready: {N_casadi} steps × {dt_casadi:.3f} s = {T_opt:.1f} s")

aircraft_go = scene.spawn("Aircraft")
aircraft_go.add_component(PointRenderer(color=Color(1, 0.85, 0), size=120))

plan_pts = []
for k in range(N_casadi + 1):
    px, pz = geo.to_local_in_m(X_opt[0, k], X_opt[1, k])
    py     = geo.altitude_to_local(X_opt[2, k])
    plan_pts.append(Vec3(px, py, pz))

plan_go = scene.spawn("Plan")
plan_lr = LineRenderer(plan_pts, color=Color(1, 1, 0))
plan_go.add_component(plan_lr)

trail_go = scene.spawn("Trail")
trail_lr = LineRenderer([Vec3.zero(), Vec3.zero()], color=Color(0.3, 0.9, 1.0))
trail_go.add_component(trail_lr)
trail_pts: list[Vec3] = []

cam.target = Vec3(0, 0, 0)

_step_idx = 0
_phys_acc = 0.0


def on_update(dt_frame: float) -> None:
    global _step_idx, _phys_acc

    _phys_acc += dt_frame
    while _phys_acc >= dt_casadi:
        if _step_idx < N_casadi:
            env.apply_action(
                thrust      = float(U_opt[0, _step_idx]),
                load_factor = float(U_opt[1, _step_idx]),
                roll        = float(U_opt[2, _step_idx]),
            )
            _step_idx += 1
        _phys_acc -= dt_casadi

    # float() extracts scalar from CasADi DM after apply_action mutates state fields
    lx, lz = geo.to_local_in_m(float(state.x), float(state.y))
    ly      = geo.altitude_to_local(float(state.h))
    wp      = Vec3(lx, ly, lz)

    aircraft_go.transform.position = wp
    cam.target = wp

    trail_pts.append(Vec3(wp.x, wp.y, wp.z))
    if len(trail_pts) > 600:
        trail_pts.pop(0)
    if len(trail_pts) >= 2:
        trail_lr.set_points_strip(trail_pts)


def on_gui() -> None:
    hud  = engine.hud
    fps  = engine._clock.get_fps() if engine._clock else 0.0
    prog = min(_step_idx / max(N_casadi, 1), 1.0)

    # float() extracts scalar from CasADi DM for arithmetic / string formatting
    sx = float(state.x);  sy = float(state.y)
    sv = float(state.v);  sh = float(state.h)
    schi = float(state.chi); sgamma = float(state.gamma)

    dx   = sx - goal_x
    dy   = sy - goal_y
    dist = np.sqrt(dx*dx + dy*dy)
    dchi = (schi - goal_chi_deg + 180) % 360 - 180

    hud.panel(10, 10, 310, 230, border_radius=8)
    hud.text("CasADi Landing",                          20, 16,  size=17, color=(255,210,60), shadow=True)
    hud.line(20, 38, 310, 38,                           color=(80,80,100,180))
    hud.text(f"Speed    {sv:6.1f} m/s",         20, 44,  size=15)
    hud.text(f"Altitude {sh:6.0f} m   (goal {goal_h:.0f})",        20, 64,  size=15)
    hud.text(f"Dist     {dist/1000:5.2f} km",                       20, 84,  size=15)
    hud.text(f"χ error  {dchi:+6.1f}°  (goal {goal_chi_deg:.0f}°)", 20, 104, size=15)
    hud.text(f"γ        {sgamma:+5.1f}°",                           20, 124, size=15)
    hud.text(f"T opt    {T_opt:6.1f} s  (dt {dt_casadi:.2f} s)",   20, 144, size=15)
    hud.line(20, 168, 310, 168,                         color=(80,80,100,180))
    hud.text(f"Progress {prog*100:4.0f}%  step {_step_idx}/{N_casadi}", 20, 173, size=14)
    hud.bar(20, 190, 270, 10, value=prog,               fg_color=(100,220,120,220))
    hud.text(f"FPS      {fps:5.0f}",                    20, 207, size=13, color=(160,160,180))


engine.on_update = on_update
engine.on_gui    = on_gui
engine.run()