import numpy as np
from numpy.typing import NDArray

from engine.utils.component import Component
from engine.utils.renderer import Renderer
from engine.utils.transform import Transform

class GameObject:
    def __init__(self, name: str = "GameObject"):
        self.name: str                    = name
        self.active: bool                 = True
        self.transform: Transform         = Transform()
        self._components: list[Component] = []
        self.tags: list[str]              = []

    def add_component(self, comp: Component) -> Component:
        comp.owner = self
        self._components.append(comp)
        return comp

    def get_component[T: Component](self, cls: type[T]) -> T | None:
        for c in self._components:
            if isinstance(c, cls):
                return c
        return None

    def _start(self) -> None:
        for c in self._components:
            c.on_start()

    def _update(self, dt) -> None:
        if not self.active: return
        for c in self._components:
            c.on_update(dt)

    def _draw(
        self,
        view_mat: NDArray[np.float32],
        proj_mat: NDArray[np.float32],
        scene: "Scene"
    ) -> None:
        if not self.active: return
        for c in self._components:
            if isinstance(c, Renderer): # e.g. MeshRenderer, LineRenderer, PointRenderer
                c.draw(view_mat, proj_mat, scene)

    def destroy(self) -> None:
        for c in self._components:
            c.on_destroy()