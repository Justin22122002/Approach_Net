from engine.utils.vec3 import Vec3
from engine.utils.simplematerial import SimpleMaterial
from engine.utils.meshRenderer import MeshRenderer
from engine.utils.color import Color
from engine.utils.mesh import Mesh
from engine.utils.texture import Texture
from engine.utils.boxCollider import BoxCollider
from engine.utils.gameObject import GameObject
from engine.utils.scene import Scene

def quick_cube(scene: Scene,
               position: Vec3 = None,
               color:    Color = None,
               texture: Texture | None = None,
               scale:    float = 1.0,
               name:     str = "Cube",
               auto_collision: bool = True) -> GameObject:
    go = scene.spawn(name)
    mat = SimpleMaterial(texture=texture, color=color or Color.white())
    go.add_component(MeshRenderer(Mesh.cube(), mat))
    if auto_collision:
        go.add_component(BoxCollider(Vec3(scale*0.5, scale*0.5, scale*0.5)))
    if position:
        go.transform.position = position
    go.transform.scale = Vec3(scale, scale, scale)
    return go


def quick_plane(scene: Scene,
                position: Vec3 = None,
                size: float = 10.0,
                color: Color = None,
                texture: Texture | None = None,
                name: str = "Plane",
                auto_collision: bool = True) -> GameObject:
    go = scene.spawn(name)
    mat = SimpleMaterial(texture=texture, color=color or Color(0.4, 0.4, 0.4))
    go.add_component(MeshRenderer(Mesh.plane(size=size, divisions=4), mat))
    if auto_collision:
        go.add_component(BoxCollider(Vec3(size*0.5, 0.05, size*0.5)))
    if position:
        go.transform.position = position
    return go


def quick_box(scene: Scene,
              position: Vec3 = None,
              size: Vec3 = None,
              color: Color = None,
              texture: Texture | None = None,
              name: str = "Box",
              auto_collision: bool = True) -> GameObject:
    go = scene.spawn(name)
    sz = size or Vec3(1, 1, 1)
    mat = SimpleMaterial(texture=texture, color=color or Color.white())
    go.add_component(MeshRenderer(Mesh.cube(), mat))
    if auto_collision:
        go.add_component(BoxCollider(Vec3(sz.x*0.5, sz.y*0.5, sz.z*0.5)))
    if position:
        go.transform.position = position
    go.transform.scale = sz
    return go