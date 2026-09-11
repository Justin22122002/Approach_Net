from typing import override

import numpy as np
from pyproj import Transformer

from utils.geo_types import Coordinate

from engine.utils.gameObject import GameObject
from engine.utils.lineRenderer import LineRenderer
from engine.utils.mesh import Mesh
from engine.utils.meshRenderer import MeshRenderer
from engine.utils.simplematerial import SimpleMaterial
from engine.utils.color import Color
from engine.utils.scene import Scene
from engine.utils.vec3 import Vec3
from engine.utils_project.geo_transformer import GeoTransformer
from engine.utils_project.project_mesh import ProjectMesh


class TaxiwayProjectMesh(ProjectMesh):

    def __init__(
        self,
        coordinates: list[Coordinate],           # list[Coordinate] with .lat / .lon
        airport_lat: float,
        airport_lon: float,
        airport_altitude_ft: float,
        horizontal_scale: float,
        vertical_scale: float,
        transformer: Transformer,
        color: Color = None,
        use_mesh: bool = False,
        taxiway_width_m: float = 20.0,
    ) -> None:
        self.coordinates: list[Coordinate]  = coordinates
        self.color: Color                   = color or Color(0.9, 0.9, 0.1)
        self.use_mesh: bool                 = use_mesh
        self.taxiway_width_m: float         = taxiway_width_m

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=airport_lat,
            lon=airport_lon,
            altitude=airport_altitude_ft * self.FT_TO_M,
            transformer=transformer,
        )

        self.origin_x, self.origin_y = self.transformer.transform(self.lon, self.lat)

    def _local_xz(self, lat: float, lon: float) -> tuple[float, float]:
        x, y = self.transformer.transform(lon, lat)
        dx = (x - self.origin_x) * self.horizontal_scale
        dz = (self.origin_y - y) * self.horizontal_scale
        return dx, dz

    def _local_points(self) -> list[Vec3]:
        return [Vec3(dx, 0.0, dz) for dx, dz in (self._local_xz(c.lat, c.lon) for c in self.coordinates)]

    @override
    def spawn(self, scene: Scene, geo: GeoTransformer) -> list[GameObject]:
        cx, cz = geo.to_local(self.lat, self.lon)
        cy = geo.altitude_to_local(self.altitude)

        go = scene.spawn("Taxiway")
        go.transform.position = Vec3(cx, cy, cz)

        if self.use_mesh:
            mesh = self.generate_mesh()
            if mesh.vertices.size == 0:
                return []
            go.add_component(MeshRenderer(mesh, SimpleMaterial(color=self.color)))
        else:
            go.add_component(LineRenderer.strip(self._local_points(), color=self.color, depth_offset=True))

        return [go]

    @override
    def generate_mesh(self) -> Mesh:
        half_w = (self.taxiway_width_m * self.horizontal_scale) / 2.0
        local_pts = [(p.x, p.y, p.z) for p in self._local_points()]

        verts: list[float] = []
        idx: list[int] = []

        for p0, p1 in zip(local_pts, local_pts[1:]):
            dir_x = p1[0] - p0[0]
            dir_z = p1[2] - p0[2]
            length = (dir_x ** 2 + dir_z ** 2) ** 0.5
            if length < 1e-9:
                continue
            dir_x /= length
            dir_z /= length

            perp_x = -dir_z * half_w
            perp_z = dir_x * half_w

            l0 = (p0[0] + perp_x, p0[1], p0[2] + perp_z)
            r0 = (p0[0] - perp_x, p0[1], p0[2] - perp_z)
            l1 = (p1[0] + perp_x, p1[1], p1[2] + perp_z)
            r1 = (p1[0] - perp_x, p1[1], p1[2] - perp_z)

            base = len(verts) // 8
            for (x, y, z), (u, v) in ((l0, (0, 0)), (r0, (1, 0)), (r1, (1, 1)), (l1, (0, 1))):
                verts += [x, y, z, 0.0, 1.0, 0.0, u, v]

            idx += [base, base + 2, base + 1, base, base + 3, base + 2]

        if not idx:
            return Mesh(np.empty(0, dtype=np.float32), np.empty(0, dtype=np.uint32))

        return Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))