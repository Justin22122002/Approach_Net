from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from pyproj import Transformer

from physics.aircraft_environment_casadi import AircraftEnvironmentCasadi
from physics.aircraft_kinematics_max_constrains import AircraftKinematicsMaxConstrains
from physics.aircraft_kinematics_state_casadi import AircraftKinematicsStateCasadi
from physics.mulit_landing_optimizer import MultiLandingTrajectoryOptimizer, state_to_casadi
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


icao = {"icao": "EDDF", "lat": 50.0379, "lon": 8.5622}
crs_to      = GeoTransformer.utm_epsg(icao["lat"], icao["lon"])
transformer = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)

basePath = Path("../data")
airport  = Airport.load_from_folder(basePath, icao["icao"])
runways: list[ArincRunway] = airport.arinc_424_runway

provider = CopernicusDEMProvider("justin.groh@study.thws.de", "Copernicus2026!")
delta    = 0.2
grid     = provider.get_terrain(
    icao["lat"] - delta, icao["lon"] - delta,
    icao["lat"] + delta, icao["lon"] + delta,
    1024, 1024,
)

horizontal_scale = 0.01
vertical_scale   = 0.01
geo = GeoTransformer(0.0, 0.0, horizontal_scale, vertical_scale, transformer)

engine = Engine(1280, 720, "CasADi Multi-Landing")
scene  = engine.scene
TerrainProjectMesh(
    grid=grid,
    lat_min=icao["lat"] - delta, lon_min=icao["lon"] - delta,
    lat_max=icao["lat"] + delta, lon_max=icao["lon"] + delta,
    horizontal_scale=horizontal_scale, vertical_scale=vertical_scale,
    transformer=transformer,
).spawn(scene, geo)

scene.ambient_light     = Color(0.25, 0.25, 0.3)
scene.directional_light = DirectionalLight(Vec3(-1,-2,-1), Color(1,0.95,0.85), 1)

cam = OrbitCamera(distance=8)
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
V_TD           = 70.0

print(f"Runway {runway}  χ={goal_chi_deg:.1f}°  "
      f"threshold ({goal_x:.0f}, {goal_y:.0f}, {goal_h:.1f} m)")

goal_go = scene.spawn("Goal")
goal_go.add_component(PointRenderer(color=Color(1, 0, 0), size=140))
gx, gz = geo.to_local_in_m(goal_x, goal_y)
goal_go.transform.position = Vec3(gx, geo.altitude_to_local(goal_h), gz)

x0_utm, y0_utm = transformer.transform(icao["lon"], icao["lat"])

AIRCRAFT_CONFIGS = [
    # (dx, dy, dh, v0, gamma_deg, dchi_deg, color)
    ( 3000, -2000, 1200,  90.0, -2.5,   0, Color(1.0, 0.85, 0.0)),
    ( 2500, -4500,  900,  95.0, -2.0,  20, Color(1.0, 0.0,  1.0)),
]

states_list      = []
constraints_list = []
for dx, dy, dh, v0, gam0, dchi, _ in AIRCRAFT_CONFIGS:
    states_list.append(AircraftKinematicsStateCasadi(
        x=x0_utm + dx, y=y0_utm + dy, h=goal_h + dh,
        v=v0, gamma=gam0, chi=-goal_chi_deg + dchi,
        mass=65_000.0, C_D0=0.028, area=124.6,
    ))
    constraints_list.append(AircraftKinematicsMaxConstrains(cl_max=3.0))

M        = len(states_list)
N_CASADI = 100
goal_vec = np.array([goal_x, goal_y, goal_h, V_TD])

print(f"Optimising {M} aircraft simultaneously …")

optimizer = MultiLandingTrajectoryOptimizer(
    states          = states_list,
    max_constraints = constraints_list,
    N               = N_CASADI,
    T_min           = 60.0,
    T_max           = 600.0,
    D_horizontal    = 4600.0, # 1000.0
    D_vertical      = 300.0,
    sep_stride      = 5,
)

x0_inputs       = [state_to_casadi(s) for s in states_list]
goals_inputs    = [goal_vec for _ in range(M)]
headings_inputs = [goal_chi_deg for _ in range(M)]

X_opt, U_opt, T_vals = optimizer.solve(x0_inputs, goals_inputs, headings_inputs)

dt_list = [T_vals[i] / N_CASADI for i in range(M)]

print("\nresult:")
for i in range(M):
    si = i * 6
    print(f"  AC{i}: T={T_vals[i]:.1f}s  dt={dt_list[i]:.3f}s, end=({X_opt[si,-1]:.0f}, {X_opt[si+1,-1]:.0f}, {X_opt[si+2,-1]:.1f}) m")

print("\nseparation diagnose:")
for i in range(M):
    for j in range(i + 1, M):
        si = i * 6; sj = j * 6
        dh = np.sqrt((X_opt[si]-X_opt[sj])**2 + (X_opt[si+1]-X_opt[sj+1])**2)
        dv = np.abs(X_opt[si+2] - X_opt[sj+2])
        print(f"  AC{i}–AC{j}: min_horiz={dh.min():.0f} m  min_vert={dv.min():.0f} m")

environments = [
    AircraftEnvironmentCasadi(
        state       = states_list[i],
        goal        = None,
        constraints = constraints_list[i],
        dt          = dt_list[i],
    )
    for i in range(M)
]

@dataclass
class AircraftData:
    idx:        int
    color:      Color
    step_idx:   int   = 0
    phys_acc:   float = 0.0
    landed:     bool  = False
    aircraft_go: object = None
    trail_pts:  list  = field(default_factory=list)
    trail_lr:   object = None

aircraft_list = [AircraftData(idx=i, color=AIRCRAFT_CONFIGS[i][6]) for i in range(M)]

for ac in aircraft_list:
    i = ac.idx
    go = scene.spawn(f"AC{i}")
    go.add_component(PointRenderer(color=ac.color, size=120))
    ac.aircraft_go = go

    si = i * 6
    plan_pts = []
    for k in range(N_CASADI + 1):
        px, pz = geo.to_local_in_m(X_opt[si, k], X_opt[si+1, k])
        py     = geo.altitude_to_local(X_opt[si+2, k])
        plan_pts.append(Vec3(px, py, pz))
    plan_go = scene.spawn(f"AC{i}_plan")
    plan_go.add_component(LineRenderer(plan_pts, color=ac.color))

    trail_go = scene.spawn(f"AC{i}_trail")
    dim = Color(ac.color.r*0.55, ac.color.g*0.55, ac.color.b*0.55)
    ac.trail_lr = LineRenderer([Vec3.zero(), Vec3.zero()], color=dim)
    trail_go.add_component(ac.trail_lr)

cam.target = Vec3(0, 0, 0)

def on_update(dt_frame: float) -> None:
    for ac in aircraft_list:
        if ac.landed:
            continue
        i = ac.idx
        ac.phys_acc += dt_frame
        while ac.phys_acc >= dt_list[i]:
            if ac.step_idx < N_CASADI:
                ci = i * 3
                environments[i].apply_action(
                    thrust      = float(U_opt[ci,   ac.step_idx]),
                    load_factor = float(U_opt[ci+1, ac.step_idx]),
                    roll        = float(U_opt[ci+2, ac.step_idx]),
                )
                ac.step_idx += 1
            else:
                ac.landed = True
            ac.phys_acc -= dt_list[i]

        s = states_list[i]
        lx, lz = geo.to_local_in_m(float(s.x), float(s.y))
        ly      = geo.altitude_to_local(float(s.h))
        wp      = Vec3(lx, ly, lz)
        ac.aircraft_go.transform.position = wp

        ac.trail_pts.append(Vec3(wp.x, wp.y, wp.z))
        if len(ac.trail_pts) > 600:
            ac.trail_pts.pop(0)
        if len(ac.trail_pts) >= 2:
            ac.trail_lr.set_points_strip(ac.trail_pts)

    flying = [ac for ac in aircraft_list if not ac.landed]
    if flying:
        # float() needed: state fields are CasADi DM after first simulation step
        cx = np.mean([geo.to_local_in_m(float(states_list[ac.idx].x), float(states_list[ac.idx].y))[0] for ac in flying])
        cz = np.mean([geo.to_local_in_m(float(states_list[ac.idx].x), float(states_list[ac.idx].y))[1] for ac in flying])
        cy = np.mean([geo.altitude_to_local(float(states_list[ac.idx].h)) for ac in flying])
        cam.target = Vec3(float(cx), float(cy), float(cz))

def on_gui() -> None:
    hud = engine.hud
    fps = engine._clock.get_fps() if engine._clock else 0.0

    panel_h = 50 + M * 72
    hud.panel(10, 10, 330, panel_h, border_radius=8)
    hud.text("CasADi Multi-Landing", 20, 16, size=16, color=(255, 210, 60), shadow=True)
    hud.line(20, 38, 330, 38, color=(80, 80, 100, 180))

    y = 44
    for ac in aircraft_list:
        i = ac.idx
        s = states_list[i]
        cr = int(ac.color.r*255); cg = int(ac.color.g*255); cb = int(ac.color.b*255)

        sv = float(s.v); sh = float(s.h); sgamma = float(s.gamma)

        status = "Landed" if ac.landed else f"Step {ac.step_idx}/{N_CASADI}"
        hud.text(f"AC{i}  T={T_vals[i]:.0f}s  {status}", 20, y, size=14, color=(cr, cg, cb), shadow=True)
        y += 18
        hud.text(f"  v={sv:.0f} m/s  h={sh:.0f} m  γ={sgamma:+.1f}°", 20, y, size=13)
        y += 18

        for j, other in enumerate(aircraft_list):
            if j <= i:
                continue
            if ac.landed or other.landed:
                continue
            sj = states_list[j]
            dh = float(np.sqrt((float(s.x)-float(sj.x))**2 + (float(s.y)-float(sj.y))**2))
            dv = float(abs(float(s.h) - float(sj.h)))
            ok = dh >= 1000.0 or dv >= 300.0
            col = (100, 220, 100) if ok else (255, 80, 80)
            hud.text(f"  sep AC{j}: {dh:.0f}m / {dv:.0f}m vert", 20, y, size=12, color=col)
            y += 16

        hud.line(20, y+2, 330, y+2, color=(60, 60, 80, 150))
        y += 8

    hud.text(f"FPS {fps:.0f}", 20, y + 2, size=12, color=(160, 160, 180))


engine.on_update = on_update
engine.on_gui    = on_gui
engine.run()