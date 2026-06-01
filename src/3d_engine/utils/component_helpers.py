from typing import Optional

from utils.vec3 import Vec3
from utils.material import Material
from utils.meshRenderer import MeshRenderer
from utils.color import Color
from utils.mesh import Mesh
from utils.texture import Texture
from utils.boxCollider import BoxCollider
from utils.gameObject import GameObject
from utils.scene import Scene

def quick_cube(scene: Scene,
               position: Vec3 = None,
               color:    Color = None,
               texture:  Optional[Texture] = None,
               scale:    float = 1.0,
               name:     str = "Cube",
               auto_collision: bool = True) -> GameObject:
    """Spawn a coloured (or textured) cube in one line.
    auto_collision=True automatically adds a BoxCollider sized to the cube.
    Set auto_collision=False for purely decorative objects.
    """
    go = scene.spawn(name)
    mat = Material(texture=texture, color=color or Color.white())
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
                texture: Optional[Texture] = None,
                name: str = "Plane",
                auto_collision: bool = True) -> GameObject:
    """Spawn a flat ground plane in one line.
    auto_collision=True automatically adds a BoxCollider covering the plane.
    Set auto_collision=False for purely decorative/background planes.
    """
    go = scene.spawn(name)
    mat = Material(texture=texture, color=color or Color(0.4, 0.4, 0.4))
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
              texture: Optional[Texture] = None,
              name: str = "Box",
              auto_collision: bool = True) -> GameObject:
    """Spawn an arbitrarily-sized box (non-uniform scale) in one line.
    Unlike quick_cube which uses a uniform scale, this lets you set
    width/height/depth independently — useful for platforms and walls.
    auto_collision=True adds a matching BoxCollider automatically.
    """
    go = scene.spawn(name)
    sz = size or Vec3(1, 1, 1)
    mat = Material(texture=texture, color=color or Color.white())
    go.add_component(MeshRenderer(Mesh.cube(), mat))
    if auto_collision:
        go.add_component(BoxCollider(Vec3(sz.x*0.5, sz.y*0.5, sz.z*0.5)))
    if position:
        go.transform.position = position
    go.transform.scale = sz
    return go