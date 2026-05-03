from typing import Optional

from utils.color import Color
from utils.gameObject import GameObject
from utils.directionalLight import DirectionalLight
from utils.vec3 import Vec3
from utils.boxCollider import BoxCollider

class Scene:
    def __init__(self):
        self._objects: list[GameObject] = []
        self.ambient_light    = Color(0.2, 0.2, 0.25, 1.0)
        self.directional_light: Optional[DirectionalLight] = DirectionalLight(
            direction=Vec3(-1,-2,-1),
            color=Color(1.0, 0.95, 0.85),
            intensity=1.0
        )

    def spawn(self, name="GameObject") -> GameObject:
        go = GameObject(name)
        self._objects.append(go)
        return go

    def destroy(self, go: GameObject):
        go.destroy()
        self._objects.remove(go)

    def find(self, name: str) -> Optional[GameObject]:
        for o in self._objects:
            if o.name == name:
                return o
        return None

    def find_by_tag(self, tag: str) -> list[GameObject]:
        return [o for o in self._objects if tag in o.tags]

    def get_colliders(self) -> list[BoxCollider]:
        result = []
        for o in self._objects:
            c = o.get_component(BoxCollider)
            if c: result.append(c)
        return result

    def check_collision(self, a: BoxCollider) -> list[BoxCollider]:
        return [b for b in self.get_colliders() if b is not a and a.overlaps(b)]

    def _start_all(self):
        for o in self._objects:
            o._start()

    def _update_all(self, dt):
        for o in list(self._objects):
            o._update(dt)

    def _draw_all(self, view_mat, proj_mat):
        for o in self._objects:
            o._draw(view_mat, proj_mat, self)