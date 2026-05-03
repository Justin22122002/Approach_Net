from utils.component import Component
from utils.transform import Transform
from utils.meshRenderer import MeshRenderer

class GameObject:
    def __init__(self, name="GameObject"):
        self.name       = name
        self.active     = True
        self.transform  = Transform()
        self._components: list[Component] = []
        self.tags: list[str] = []

    def add_component(self, comp: Component) -> Component:
        comp.owner = self
        self._components.append(comp)
        return comp

    def get_component(self, cls):
        for c in self._components:
            if isinstance(c, cls):
                return c
        return None

    def _start(self):
        for c in self._components:
            c.on_start()

    def _update(self, dt):
        if not self.active: return
        for c in self._components:
            c.on_update(dt)

    def _draw(self, view_mat, proj_mat, scene):
        if not self.active: return
        for c in self._components:
            if isinstance(c, MeshRenderer):
                c.draw(view_mat, proj_mat, scene)

    def destroy(self):
        for c in self._components:
            c.on_destroy()