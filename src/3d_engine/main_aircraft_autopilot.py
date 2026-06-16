"""
coordinate-Mapping  (Aircraft → Engine):
    East    (x)  →  world-X
    Altitude(h)  →  world-Y
    North   (y)  →  world-Z (negative, cause opeGl is right handed with z in front)
"""

import numpy as np

from physics.aircraft_environment import AircraftEnvironment
from physics.aircraft_goal import AircraftGoal
from physics.aircraft_type import AircraftType
from physics.aircraft_kinematics_state import AircraftKinematicsState

from utils.engine import Engine
from utils.camera import OrbitCamera
from utils.vec3 import Vec3
from utils.color import Color
from utils.directionalLight import DirectionalLight
from utils.lineRenderer import LineRenderer
from utils.pointRenderer import PointRenderer

# scale -> 1 world unit in opengl = 100 meter
SCALE = 1.0 / 100.0

def to_world(x: float, y: float, h: float) -> Vec3:
    return Vec3(x * SCALE, h * SCALE, -y * SCALE)

engine = Engine(1280, 720, "flight simulation", clear_color=Color(0.05, 0.08, 0.18))
scene  = engine.scene

scene.ambient_light = Color(0.2, 0.22, 0.28)
scene.directional_light = DirectionalLight(
    direction=Vec3(-1, -2, -0.5),
    color=Color(1.0, 0.95, 0.85),
    intensity=1.0,
)

cam = OrbitCamera(distance=25.0)
cam.pitch = 15.0
engine.camera = cam

# chi is here set to 0.0, and so the plane will fly in y direction and so in opengl in -z
constraints = AircraftType(
    cl_max=3.0,
    mass=65000.0,
    C_D0=0.028,
    area=124.6,
)

b737 = AircraftKinematicsState(
    x=0.0,
    y=0.0,
    h=3000.0,
    v=220.0,
    gamma=0.0,
    chi=0.0,
    aircraft_type=constraints,
)

goal = AircraftGoal(x=10000.0, y=10000.0, h=4000.0)
env  = AircraftEnvironment(state=b737, goal=goal, dt=0.1)

thrust      = 240000.0
load_factor = 1.0
roll        = 0.0

THRUST_STEP      = 20000.0
ROLL_STEP        = 25.0
LF_STEP          = 0.4
PHYS_DT          = 0.1
MAX_TRAIL_POINTS = 600
MAX_TRAIL        = True


ground_go = scene.spawn("Ground")
ground_go.add_component(
    LineRenderer.grid(size=400.0, divisions=40, color=Color(0.2, 0.2, 0.25))
)

axes_go = scene.spawn("Axes")
axes_go.add_component(LineRenderer.axes(length=5.0))
axes_go.transform.position = Vec3(0, 0.1, 0)

aircraft_go = scene.spawn("Aircraft")
aircraft_go.transform.position = to_world(b737.x, b737.y, b737.h)
aircraft_go.add_component(PointRenderer(color=Color(1.0, 0.85, 0.0), size=120.0))

vel_go = scene.spawn("VelocityVector")
vel_lr = LineRenderer([Vec3.zero(), Vec3.zero()], color=Color(0.1, 1.0, 0.3))
vel_go.add_component(vel_lr)

trail_go = scene.spawn("Trail")
trail_lr = LineRenderer([Vec3.zero(), Vec3.zero()], color=Color(0.3, 0.75, 1.0))
trail_go.add_component(trail_lr)

goal_go = scene.spawn("Goal")
goal_go.transform.position = to_world(
    goal.x,
    goal.y,
    goal.h
)
goal_go.add_component(
    PointRenderer(
        color=Color(1.0, 0.0, 0.0),
        size=200.0
    )
)

trail_pts: list[Vec3] = []
_phys_acc: float      = 0.0

def on_update(dt: float) -> None:
    global thrust, load_factor, roll
    global _phys_acc, trail_pts

    _phys_acc += dt

    while _phys_acc >= PHYS_DT:
        thrust, load_factor, roll = env.autopilot()

        thrust = np.clip(thrust, 0.0, 350000.0)
        roll = np.clip(roll, -60.0, 60.0)

        env.apply_action(
            thrust=thrust,
            load_factor=load_factor,
            roll=roll,
        )

        _phys_acc -= PHYS_DT

    wp = to_world(b737.x, b737.y, b737.h)

    aircraft_go.transform.position = wp
    cam.target = wp

    trail_pts.append(Vec3(wp.x, wp.y, wp.z))
    if len(trail_pts) > 600:
        trail_pts.pop(0)

    if len(trail_pts) >= 2:
        trail_lr.set_points_strip(trail_pts)

    v = b737.velocity_vector()

    vel_lr.set_points_strip([
        wp,
        Vec3(
            wp.x + v[0] * SCALE,
            wp.y + v[2] * SCALE,
            wp.z - v[1] * SCALE
        )
    ])

engine.on_update = on_update

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
engine.run()