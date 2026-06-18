from utils.engine import Engine
from utils.meshRenderer import MeshRenderer
from utils.sky import Sky
from utils.text_renderer import TextRenderer
from utils.vec3 import Vec3
from utils.color import Color
from utils.directionalLight import DirectionalLight
from utils.camera import FlyCamera
from utils.mesh import Mesh
from utils.texture import Texture
from utils.simplematerial import SimpleMaterial

engine = Engine(800, 600, "My Game")
scene  = engine.scene

cube1 = scene.spawn("Cube")
wood_texture  = Texture("textures/wood.png")
wood_material = SimpleMaterial(texture=wood_texture)
cube1.add_component(MeshRenderer(Mesh.cube(), material=wood_material))
cube1.transform.position = Vec3(0, 0, -5)

cube2 = scene.spawn("Cube")
cube2.add_component(MeshRenderer(Mesh.cube(), material=None))
cube2.transform.position = Vec3(0, 0, -8)

label = scene.spawn("Label")
label.transform.position = Vec3(2, 1.5, 0)
label.add_component(TextRenderer(
    text="",
    font_size=48,
    color=Color.white(),
    world_height=0.6,
    billboard=True,
))

# Lighting
scene.ambient_light = Color(0.2, 0.2, 0.2)
scene.directional_light = DirectionalLight(
    direction=Vec3(-1, -2, -1),
    color=Color(1.0, 0.95, 0.85)
)

# Camera
engine.camera = FlyCamera(speed=5.0)

engine.sky = Sky(
    sun_elevation=30.0,
    sun_azimuth=45.0,
    cloud_density=1.0,
    cloud_speed=0.012,
    zenith_color=(0.07, 0.22, 0.68),
    horizon_color=(0.52, 0.75, 0.94),
)
scene.directional_light = engine.sky.directional_light

state = {
    "paused":      False,
    "show_panel":  True,
    "speed_index": 1,       # 0=slow 1=normal 2=fast
    "auto_sun":    True,    # if False, sun_elevation only moves via the slider
}
SPEEDS = [1.0, 3.0, 8.0]
SPEED_LABELS = ["Slow", "Normal", "Fast"]

def on_pause_toggle():
    state["paused"] = not state["paused"]

def on_speed_cycle():
    state["speed_index"] = (state["speed_index"] + 1) % len(SPEEDS)
    engine.sky.cloud_speed = 0.012 * SPEEDS[state["speed_index"]]

def on_toggle_panel():
    state["show_panel"] = not state["show_panel"]

def on_toggle_auto_sun():
    state["auto_sun"] = not state["auto_sun"]

def on_quit():
    engine.quit()

def on_update(dt: float):
    if not state["paused"]:
        label.get_component(TextRenderer).set_text(f"Zeit: {engine.time:.1f}s")

        if state["auto_sun"]:
            engine.sky.sun_elevation += dt * 3.0
            if engine.sky.sun_elevation > 180.0:
                engine.sky.sun_elevation = -10.0

        scene.directional_light.direction = engine.sky.get_light_direction()

        elev = engine.sky.sun_elevation

        if elev > 25.0:
            scene.directional_light.color = Color(1.0, 0.95, 0.85)
        elif elev > 5.0:
            scene.directional_light.color = Color(1.0, 0.7, 0.45)
        else:
            scene.directional_light.color = Color(1.0, 0.35, 0.2)

engine.on_update = on_update

def on_gui():
    hud = engine.hud

    hud.button(
        "Panel",
        x=10, y=10, w=90, h=30,
        callback=on_toggle_panel,
        color=(30, 30, 50, 200),
        hover_color=(60, 60, 100, 230),
    )

    if not state["show_panel"]:
        return

    hud.panel(x=10, y=50, w=220, h=350)

    # Status label
    status = "PAUSED" if state["paused"] else "RUNNING"
    hud.text(status, x=20, y=60, size=16, color=(200, 230, 255))

    hud.button(
        "Resume" if state["paused"] else "Pause",
        x=20, y=90, w=200, h=36,
        callback=on_pause_toggle,
        color=(140, 60, 60, 220) if not state["paused"] else (60, 140, 60, 220),
        hover_color=(180, 80, 80, 240) if not state["paused"] else (80, 180, 80, 240),
    )

    current_speed = SPEED_LABELS[state["speed_index"]]
    hud.button(
        f"Speed: {current_speed}",
        x=20, y=136, w=200, h=36,
        callback=on_speed_cycle,
    )

    hud.text("Sonnenstand", x=20, y=180, size=14, color=(180, 200, 220))
    engine.sky.sun_elevation = hud.slider(
        "sun_elevation",
        x=20, y=198, w=200, h=18,
        value=engine.sky.sun_elevation,
        min_value=-10.0, max_value=180.0,
        show_value=True,
        value_fmt="{:.0f}°",
    )

    hud.text("Wolkendichte", x=20, y=226, size=14, color=(180, 200, 220))
    engine.sky.cloud_density = hud.slider(
        "cloud_density",
        x=20, y=244, w=200, h=18,
        value=engine.sky.cloud_density,
        min_value=0.0, max_value=2.0,
        show_value=True,
    )

    hud.text("Ambient", x=20, y=272, size=14, color=(180, 200, 220))
    ambient_b = hud.slider(
        "ambient_brightness",
        x=20, y=290, w=200, h=18,
        value=scene.ambient_light.r,
        min_value=0.0, max_value=1.0,
        show_value=True,
    )
    scene.ambient_light = Color(ambient_b, ambient_b, ambient_b)
    # -----------------------------------------------------------------------

    hud.button(
        "Auto-Sonne: An" if state["auto_sun"] else "Auto-Sonne: Aus",
        x=20, y=316, w=200, h=32,
        callback=on_toggle_auto_sun,
        color=(60, 100, 140, 220) if state["auto_sun"] else (90, 90, 90, 220),
        hover_color=(80, 130, 180, 240) if state["auto_sun"] else (120, 120, 120, 240),
    )

    # Quit button
    hud.button(
        "Quit",
        x=20, y=352, w=200, h=36,
        callback=on_quit,
        color=(100, 30, 30, 220),
        hover_color=(160, 50, 50, 240),
        border_color=(200, 80, 80, 255),
        hover_border_color=(255, 120, 120, 255),
    )

engine.on_gui = on_gui

engine.run()