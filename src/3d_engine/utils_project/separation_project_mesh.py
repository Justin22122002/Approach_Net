from typing import override

import numpy as np
from pyproj import Transformer

from utils.color import Color
from utils.gameObject import GameObject
from utils.lineRenderer import LineRenderer
from utils.mesh import Mesh
from utils.scene import Scene
from utils.vec3 import Vec3
from utils_project.geo_transformer import GeoTransformer
from utils_project.project_mesh import ProjectMesh
from utils_project.separation_geometry import SeparationGeometry


def _to_vec3(geo: GeoTransformer, x: float, y: float, h: float) -> Vec3:
    lx, lz = geo.to_local_in_m(x, y)
    return Vec3(lx, geo.altitude_to_local(h), lz)


class SeparationProjectMesh(ProjectMesh):
    def __init__(
        self,
        center_xy: tuple[float, float],
        center_h: float,
        transformer: Transformer,
        horizontal_scale: float,
        vertical_scale: float,
        separation: SeparationGeometry,
        cylinder_color: Color = Color(0.2, 0.9, 0.4),
        ellipsoid_color: Color = Color(1.0, 0.9, 0.2),
        label: str = "Sep",
    ) -> None:
        self.center = (center_xy[0], center_xy[1], center_h)
        self.separation = separation
        self.cylinder_color = cylinder_color
        self.ellipsoid_color = ellipsoid_color
        self.label = label

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=0.0, lon=0.0, altitude=0.0,
            transformer=transformer,
        )

    @override
    def spawn(self, scene: Scene, geo: GeoTransformer) -> list[GameObject]:
        objects: list[GameObject] = []

        for i, edge in enumerate(self.separation.cylinder_edges(self.center)):
            pts = [_to_vec3(geo, *p) for p in edge]
            go = scene.spawn(f"{self.label}Cyl_{i}")
            go.add_component(LineRenderer.strip(pts, color=self.cylinder_color))
            objects.append(go)

        for i, ring in enumerate(self.separation.ellipsoid_rings(self.center)):
            pts = [_to_vec3(geo, *p) for p in ring]
            go = scene.spawn(f"{self.label}Ellip_{i}")
            go.add_component(LineRenderer.strip(pts, color=self.ellipsoid_color))
            objects.append(go)

        print(f"[separation] D_h={self.separation.d_h:.0f}m  D_v={self.separation.d_v:.0f}m")
        return objects

    @override
    def generate_mesh(self) -> Mesh:
        return Mesh(np.empty(0, dtype=np.float32), np.empty(0, dtype=np.uint32))