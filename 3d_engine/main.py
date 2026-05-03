from utils.engine import Engine
from utils.meshRenderer import MeshRenderer
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

# Lighting
scene.ambient_light = Color(0.2, 0.2, 0.2)
scene.directional_light = DirectionalLight(
    direction=Vec3(-1, -2, -1),
    color=Color(1.0, 0.95, 0.85)
)

# Camera
engine.camera = FlyCamera(speed=5.0)

engine.run()