"""
coordinate-Mapping  (Aircraft → Engine):
    East    (x)  →  world-X
    Altitude(h)  →  world-Y
    North   (y)  →  world-Z (negative, cause opeGl is right handed with z in front)
"""

import math
import pygame

from physics.aircraft_environment import AircraftEnvironment
from physics.aircraft_goal import AircraftGoal
from physics.aircraft_dynamics import AircraftDynamics
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


constraints = AircraftDynamics(
    cl_max=3.0,
    mass=65000.0,
    C_D0=0.028,
    area=124.6,
    max_thrust=240000,
    max_load_factor=2.5,
    min_load_factor=0.3,
    max_roll_angle=60,
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

goal = AircraftGoal(x=40000.0, y=25000.0, h=1000.0)
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

trail_pts: list[Vec3] = []
_phys_acc: float      = 0.0

def on_update(dt: float) -> None:
    global thrust, load_factor, roll, _phys_acc

    inp = engine.input

    if inp.get_key(pygame.K_w):
        thrust = min(thrust + THRUST_STEP * dt, 500_000.0)
    if inp.get_key(pygame.K_s):
        thrust = max(thrust - THRUST_STEP * dt, 0.0)

    if inp.get_key(pygame.K_a):
        roll = max(roll - ROLL_STEP * dt, -60.0)
    if inp.get_key(pygame.K_d):
        roll = min(roll + ROLL_STEP * dt,  60.0)

    if inp.get_key(pygame.K_r):
        roll = 0.0

    if inp.get_key(pygame.K_q):
        load_factor = max(load_factor - LF_STEP * dt, 0.0)
    if inp.get_key(pygame.K_e):
        load_factor = min(load_factor + LF_STEP * dt, 3.0)

    _phys_acc += dt
    while _phys_acc >= PHYS_DT:
        env.apply_action(
            thrust=thrust,
            load_factor=load_factor,
            roll=roll,
        )
        _phys_acc -= PHYS_DT

    # world position of the aircraft
    wp = to_world(b737.x, b737.y, b737.h)

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
        wp.x + v_raw[0] * SCALE,
        wp.y + v_raw[2] * SCALE,          # Up: Y
        wp.z - v_raw[1] * SCALE,          # North: -Z
    )
    vel_lr.set_points_strip([wp, vel_tip])


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

    # steering panel
    px = hud.width - 295
    hud.panel(px, 10, 285, 140, border_radius=8)
    hud.text("steering",  px + 10, 16, size=17, color=(255, 210, 60), shadow=True)
    hud.line(px + 10, 38, px + 275, 38, color=(80, 80, 100, 180))

    # Thrust
    hud.text(f"Thrust        {thrust/1000:5.1f} kN   W/S", px + 10, 44, size=15)
    hud.bar(px + 10, 62, 265, 10, value=thrust / 130_000, fg_color=(255, 160, 30, 220))

    # roll
    hud.text(f"Roll          {roll:+5.1f}°       A/D  (R=level)", px + 10, 78, size=15)
    hud.bar(px + 10, 96, 265, 10, value=(roll + 60) / 120, fg_color=(80, 180, 255, 220))

    # Load-Factor
    hud.text(f"n-oad-Factor  {load_factor:4.2f}       Q/E", px + 10, 112, size=15)
    hud.bar(px + 10, 130, 265, 10, value=load_factor / 3.0, fg_color=(100, 220, 120, 220))

engine.on_gui = on_gui
engine.run()