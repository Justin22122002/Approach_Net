import math

from engine.utils.engine import Engine
from engine.utils.camera import FlyCamera
from engine.utils.pointRenderer import PointRenderer
from engine.utils.vec3 import Vec3
from engine.utils.color import Color
from engine.utils.directionalLight import DirectionalLight
from engine.utils.lineRenderer import LineRenderer
from engine.utils.component_helpers import quick_cube

engine = Engine(1280, 720, "LineRenderer Demo", clear_color=Color(0.07, 0.07, 0.10))
scene  = engine.scene

engine.camera = FlyCamera(speed=8.0)
engine.camera.position = Vec3(0, 6, 14)

scene.ambient_light = Color(0.15, 0.15, 0.18)
scene.directional_light = DirectionalLight(
    direction=Vec3(-1, -2, -1),
    color=Color(1.0, 0.95, 0.85),
    intensity=1.0,
)

grid_go = scene.spawn("Grid")
grid_go.add_component(LineRenderer.grid(size=30.0, divisions=30, color=Color(0.2, 0.2, 0.25)))
grid_go.transform.position = Vec3(0, -0.1, 0)

axes_go = scene.spawn("Axes")
axes_go.add_component(LineRenderer.axes(length=4.0))

wire_go = scene.spawn("WireBox")
wire_go.add_component(
    LineRenderer.wire_box(Vec3(-1, 0, -1), Vec3(1, 2, 1), color=Color(1.0, 0.8, 0.0))
)
wire_go.transform.position = Vec3(-5, 0, -3)

circle_go = scene.spawn("Circles")
circle_go.add_component(LineRenderer.circle(radius=1.5, segments=64, axis="y", color=Color.red()))
circle_go.add_component(LineRenderer.circle(radius=1.5, segments=64, axis="x", color=Color.green()))
circle_go.add_component(LineRenderer.circle(radius=1.5, segments=64, axis="z", color=Color.blue()))
circle_go.transform.position = Vec3(5, 1.5, -3)

go = scene.spawn("MyPoint")
go.transform.position = Vec3(2, 1, -4)
go.add_component(PointRenderer(color=Color.red(), size=50.0))

cloud = scene.spawn("Cloud")
cloud.add_component(PointRenderer(
    points=[Vec3(0,0,0), Vec3(1,1,0), Vec3(2,0,1)],
    colors=[Color.red(), Color.green(), Color.blue()],
    size=50.0,
))

def make_spiral(turns: int = 4, points_per_turn: int = 60, radius: float = 1.2, height: float = 4.0):
    total = turns * points_per_turn
    pts: list[Vec3]  = []
    cols: list[Color] = []
    for i in range(total + 1):
        t   = i / total
        ang = 2 * math.pi * turns * t
        x   = math.cos(ang) * radius
        z   = math.sin(ang) * radius
        y   = height * t
        pts.append(Vec3(x, y, z))
        cols.append(Color(t, 1.0 - t, 1.0))
    return LineRenderer.strip(pts, colors=cols)

spiral_go = scene.spawn("Spiral")
spiral_go.add_component(make_spiral())
spiral_go.transform.position = Vec3(0, 0, -6)

sine_pts: list[Vec3]  = []
sine_cols: list[Color] = []
steps = 120
for i in range(steps + 1):
    t = i / steps
    x = (t - 0.5) * 14          # von -7 bis +7
    y = math.sin(t * 4 * math.pi) * 1.5
    z = math.cos(t * 2 * math.pi) * 2.0
    sine_pts.append(Vec3(x, y + 1.5, z))
    sine_cols.append(Color(t, 0.5, 1.0 - t))

sine_go = scene.spawn("SineWave")
sine_go.add_component(LineRenderer.strip(sine_pts, colors=sine_cols))
sine_go.transform.position = Vec3(0, 0, 4)

cross_go = scene.spawn("Cross")
cross_go.add_component(
    LineRenderer.segments(
        [
            (Vec3(-1, 0, -1), Vec3(1, 2,  1)),
            (Vec3( 1, 0, -1), Vec3(-1, 2, 1)),
            (Vec3( 0, 0, -1), Vec3(0,  2, 1)),
        ],
        color=Color(1.0, 0.5, 0.0),
    )
)
cross_go.transform.position = Vec3(7, 0, 2)

ref_cube = quick_cube(scene, position=Vec3(0, 0.5, -6), color=Color(0.3, 0.3, 0.9), scale=1.0, name="RefCube")

def on_update(dt: float) -> None:
    t = engine.time
    spiral_go.transform.rotation = Vec3(0, t * 40, 0)
    circle_go.transform.rotation = Vec3(t * 20, t * 30, t * 15)
    wire_go.transform.rotation = Vec3(0, t * 25, 0)
    sine_go.transform.rotation = Vec3(math.sin(t) * 10, 0, 0)

def on_gui():
    hud = engine.hud
    pos = engine.camera.position
    hud.text(f"X: {pos.x:.1f}  Y: {pos.y:.1f}  Z: {pos.z:.1f}", 10, 10, size=18, shadow=True)
    hud.crosshair()

engine.on_gui = on_gui
engine.on_update = on_update
engine.run()