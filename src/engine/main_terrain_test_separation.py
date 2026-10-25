from dataclasses import dataclass, field

import torch

from engine.physics.aircraft_environment import AircraftEnvironment
from engine.physics.aircraft_kinematics_state import AircraftKinematicsState

from engine.utils.engine import Engine
from engine.utils.lineRenderer import LineRenderer
from engine.utils.color import Color
from engine.utils.directionalLight import DirectionalLight
from engine.utils.camera import OrbitCamera, FlyCamera
from engine.utils.pointRenderer import PointRenderer
from engine.utils.vec3 import Vec3
from engine.utils_project.separation_geometry import SeparationGeometry
from ml.utils.models.constraint_joint_speration_layer_cvxpy import AircraftSceneSeparationQPLayer

PHYS_DT = 1.0
SUB_STEPS = 5

RENDER_SCALE = 0.01

U_NOMINAL = (80_000.0, 1.0, 0.0)


@dataclass
class FakeAircraftDynamics:
    mass: float = 65000.0
    C_D0: float = 0.028
    area: float = 124.6
    cl_max: float = 2.4
    k: float = 0.045
    max_thrust: float = 240e3
    min_load_factor: float = 0.3
    max_load_factor: float = 1.5
    max_roll_angle: float = 30.0


def _to_local(x: float, y: float, h: float) -> Vec3:
    return Vec3(x * RENDER_SCALE, h * RENDER_SCALE, -y * RENDER_SCALE)


def spawn_separation_mesh(
        scene,
        separation: SeparationGeometry,
        label: str,
        cylinder_color=Color(0.2, 0.9, 0.4),
        ellipsoid_color=Color(1.0, 0.9, 0.2)
):
    cyl_lrs, ellip_lrs = [], []
    dummy_center = (0.0, 0.0, 0.0)
    for i, edge in enumerate(separation.cylinder_edges(dummy_center)):
        pts = [_to_local(*p) for p in edge]
        go = scene.spawn(f"{label}Cyl_{i}")
        lr = LineRenderer.strip(pts, color=cylinder_color)
        go.add_component(lr)
        cyl_lrs.append(lr)
    for i, ring in enumerate(separation.ellipsoid_rings(dummy_center)):
        pts = [_to_local(*p) for p in ring]
        go = scene.spawn(f"{label}Ellip_{i}")
        lr = LineRenderer.strip(pts, color=ellipsoid_color)
        go.add_component(lr)
        ellip_lrs.append(lr)
    return cyl_lrs, ellip_lrs


def update_separation_mesh(separation: SeparationGeometry, center_xy: tuple[float, float], center_h: float,
                            cyl_lrs: list, ellip_lrs: list) -> None:
    center = (center_xy[0], center_xy[1], center_h)
    for lr, edge in zip(cyl_lrs, separation.cylinder_edges(center)):
        lr.set_points_strip([_to_local(*p) for p in edge])
    for lr, ring in zip(ellip_lrs, separation.ellipsoid_rings(center)):
        lr.set_points_strip([_to_local(*p) for p in ring])


_full_sep = SeparationGeometry()
separation_geom = SeparationGeometry(
    d_horizontal_m=_full_sep.d_h / 2.0,
    d_vertical_m=_full_sep.d_v / 2.0,
    n=_full_sep.n,
    contain_cylinder=_full_sep.contain_cylinder,
)


@dataclass
class ActiveAircraft:
    name: str
    env: AircraftEnvironment
    state: AircraftKinematicsState
    dynamics: FakeAircraftDynamics
    landed: bool = False
    thrust: float = 0.0
    load_factor: float = 1.0
    roll: float = 0.0
    tag: str = ""
    go: object = None
    trail_go: object = None
    trail_lr: object = None
    trail_pts: list = field(default_factory=list)
    sep_cyl_lrs: list = field(default_factory=list)
    sep_ellip_lrs: list = field(default_factory=list)

layer = AircraftSceneSeparationQPLayer(
    solver=None,            # None -> cvxpy default; cp.MOREAU nur in WSL
    rho_slack=1e5,
    debug=False,
)
layer.eval()

engine = Engine(1280, 720, "Scene Separation-Layer Test")
scene = engine.scene

scene.ambient_light = Color(0.25, 0.25, 0.3)
scene.directional_light = DirectionalLight(direction=Vec3(-1, -2, -1), color=Color(1.0, 0.95, 0.85), intensity=1.0)
cam = FlyCamera(speed=100.0)
cam.pitch = 15.0
engine.camera = cam

D = 15000.0
init = [
    ("A", dict(x=0.0, y=-D, h=10000.0, v=70.0, gamma=0.0, chi=0.0)),
    ("B", dict(x=-D, y=0.0, h=10000.0, v=70.0, gamma=0.0, chi=90.0)),
]

active_aircraft: list[ActiveAircraft] = []
for name, kw in init:
    dyn = FakeAircraftDynamics()
    state = AircraftKinematicsState(**kw, aircraft_type=dyn)
    env = AircraftEnvironment(state=state, goal=None, dt=PHYS_DT / SUB_STEPS)

    go = scene.spawn(f"Aircraft_{name}")
    go.transform.position = Vec3(kw["x"] * RENDER_SCALE, kw["h"] * RENDER_SCALE, -kw["y"] * RENDER_SCALE)
    go.add_component(PointRenderer(color=Color(1.0, 0.85, 0.0), size=100.0))

    trail_go = scene.spawn(f"Trail_{name}")
    trail_lr = LineRenderer([Vec3.zero(), Vec3.zero()], color=Color(0.3, 0.75, 1.0))
    trail_go.add_component(trail_lr)

    # if name == "A":
    sep_cyl, sep_ellip = spawn_separation_mesh(scene, separation_geom, label=name)

    active_aircraft.append(ActiveAircraft(
        name=name, env=env, state=state, dynamics=dyn,
        go=go, trail_go=trail_go, trail_lr=trail_lr,
        sep_cyl_lrs=sep_cyl, sep_ellip_lrs=sep_ellip,
    ))


def predict_actions(aircraft: list[ActiveAircraft]) -> None:
    living = [a for a in aircraft if not a.landed]
    if not living:
        return

    aircraft_state = torch.tensor([[a.state.h, a.state.v, a.state.gamma] for a in living], dtype=torch.float32)
    aircraft_pos = torch.tensor([[a.state.x, a.state.y, a.state.chi] for a in living], dtype=torch.float32)
    aircraft_dyn = torch.tensor([[a.dynamics.mass, a.dynamics.C_D0, a.dynamics.area, a.dynamics.cl_max, a.dynamics.k] for a in living], dtype=torch.float32)
    aircraft_limits = torch.tensor([[a.dynamics.max_thrust, a.dynamics.min_load_factor, a.dynamics.max_load_factor, a.dynamics.max_roll_angle] for a in living], dtype=torch.float32)
    u_ref = torch.tensor([U_NOMINAL] * len(living), dtype=torch.float32)

    with torch.no_grad():
        out = layer(u_ref, aircraft_state, aircraft_dyn, aircraft_limits, aircraft_pos).numpy()

    active = layer.last_active_mask.tolist()
    for a, (t, n, phi), is_active in zip(living, out, active):
        a.thrust, a.load_factor, a.roll = float(t), float(n), float(phi)
        a.tag = "SEP" if is_active else ""

_phys_acc = 0.0

def on_update(dt: float) -> None:
    global _phys_acc
    _phys_acc += 1.0
    while _phys_acc >= PHYS_DT:
        predict_actions(active_aircraft)
        for _ in range(SUB_STEPS):
            for a in active_aircraft:
                if a.landed:
                    continue
                a.env.apply_action(thrust=a.thrust, load_factor=a.load_factor, roll=a.roll)
                if a.state.h <= 0.0:
                    a.landed = True
        _phys_acc -= PHYS_DT

    living_positions: list[Vec3] = []
    for a in active_aircraft:
        if a.landed:
            continue
        wp = Vec3(a.state.x * RENDER_SCALE, a.state.h * RENDER_SCALE, -a.state.y * RENDER_SCALE)
        a.go.transform.position = wp
        living_positions.append(wp)

        a.trail_pts.append(wp)
        if len(a.trail_pts) >= 2:
            a.trail_lr.set_points_strip(a.trail_pts)

        update_separation_mesh(separation_geom, (a.state.x, a.state.y), a.state.h,
                                a.sep_cyl_lrs, a.sep_ellip_lrs)

    if living_positions:
        cam.target = Vec3(
            sum(p.x for p in living_positions) / len(living_positions),
            sum(p.y for p in living_positions) / len(living_positions),
            sum(p.z for p in living_positions) / len(living_positions),
        )

def on_gui() -> None:
    hud = engine.hud
    hud.panel(10, 10, 420, 80 + 18 * len(active_aircraft), border_radius=8)
    hud.text("Scene Separation-Layer Test", 20, 16, size=14, color=(60, 210, 255))
    hud.line(20, 38, 420, 38, color=(80, 80, 100, 180))
    y = 44
    for a in active_aircraft:
        hud.text(f"{a.name}  h={a.state.h:5.0f}m  v={a.state.v:4.0f}m/s  T={a.thrust/1e3:.0f}kN  {a.tag}",
                  20, y, size=12, color=(255, 170, 60) if a.tag else (220, 220, 230))
        y += 18
    s = layer.last_stats
    hud.text(f"QP active scenes: {s.get('scenes_active', 0)}  failed: {s.get('scenes_failed', 0)}",
             20, y + 4, size=12, color=(160, 160, 180))

engine.on_gui = on_gui
engine.on_update = on_update
engine.run()