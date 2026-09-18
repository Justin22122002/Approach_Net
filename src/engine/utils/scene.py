from collections.abc import Generator

import numpy as np
from numpy.typing import NDArray

from engine.utils.color import Color
from engine.utils.gameObject import GameObject
from engine.utils.directionalLight import DirectionalLight, PointLight, SpotLight
from engine.utils.meshRenderer import MeshRenderer
from engine.utils.vec3 import Vec3
from engine.utils.boxCollider import BoxCollider
from engine.utils.shadow_map import ShadowMap
from utils.mesh import Mesh


class Scene:
    def __init__(self) -> None:
        self._objects: list[GameObject] = []
        self.ambient_light: Color       = Color(0.2, 0.2, 0.25, 1.0)
        self.directional_light: DirectionalLight | None = DirectionalLight(
            direction=Vec3(-1, -2, -1),
            color=Color(1.0, 0.95, 0.85),
            intensity=1.0
        )

        # those are only used for the complex material, not for the simple material
        self.point_lights: list[PointLight] = []
        self.spot_lights:  list[SpotLight]  = []

        # directional-light shadow map
        self.shadow_map: ShadowMap = ShadowMap()
        self.shadow_enabled: bool  = True
        self.shadow_radius: float  = 25.0

    def spawn(self, name: str = "GameObject") -> GameObject:
        go = GameObject(name)
        self._objects.append(go)
        return go

    def destroy(self, go: GameObject):
        go.destroy()
        self._objects.remove(go)

    def find(self, name: str) -> GameObject | None:
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

    # -> model_mat, mesh
    def shadow_casters(self) -> Generator[tuple[np.ndarray, Mesh], None, None]:
        for o in self._objects:
            if not o.active:
                continue
            mr = o.get_component(MeshRenderer)
            if mr and mr.cast_shadow:
                yield o.transform.matrix(), mr.mesh

    def _render_shadow_pass(self, focus: Vec3) -> None:
        if not self.shadow_enabled or self.directional_light is None:
            return
        self.shadow_map.render(
            light_dir=self.directional_light.direction,
            center=focus,
            radius=self.shadow_radius,
            casters=self.shadow_casters(),
        )

    def _start_all(self) -> None:
        for o in self._objects:
            o._start()

    def _update_all(self, dt: float) -> None:
        for o in list(self._objects):
            o._update(dt)

    def _draw_all(
        self,
        view_mat: NDArray[np.float32],
        proj_mat: NDArray[np.float32]
    ) -> None:
        opaque, transparent = [], []
        for go in self._objects:
            if not go.active:
                continue
            mr = go.get_component(MeshRenderer)
            (transparent if mr and mr.material.is_transparent else opaque).append(go)

        for go in opaque:
            go._draw(view_mat, proj_mat, self)

        if transparent:
            cam_pos = np.linalg.inv(view_mat)[:3, 3]
            transparent.sort(key=lambda go: -np.linalg.norm(go.transform.position.to_np() - cam_pos))
            for go in transparent:
                go._draw(view_mat, proj_mat, self)