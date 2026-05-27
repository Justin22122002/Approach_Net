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
from utils.material import Material

engine = Engine(800, 600, "My Game")
scene  = engine.scene

# Add a cube
cube1 = scene.spawn("Cube")
wood_texture = Texture("textures/wood.png")
wood_material = Material(texture=wood_texture)
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

def on_update(dt: float):
    label.get_component(TextRenderer).set_text(f"Zeit: {engine.time:.1f}s")

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

engine.run()