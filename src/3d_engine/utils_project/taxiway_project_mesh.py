from typing import override

import numpy as np
from pyproj import Transformer

from utils.gameObject import GameObject
from utils.geo_types import Coordinate
from utils.lineRenderer import LineRenderer
from utils.mesh import Mesh
from utils.meshRenderer import MeshRenderer
from utils.simplematerial import SimpleMaterial
from utils.color import Color
from utils.scene import Scene
from utils.vec3 import Vec3
from utils_project.geo_transformer import GeoTransformer
from utils_project.project_mesh import ProjectMesh


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
            transformer=transformer
        )

    @override
    def spawn(self, scene: Scene, geo: GeoTransformer) -> list[GameObject]:
        if self.use_mesh:
            return self._spawn_mesh(scene, geo)
        return self._spawn_line(scene, geo)

    def _spawn_line(self, scene: Scene, geo: GeoTransformer) -> list[GameObject]:
        dy: float = geo.altitude_to_local(self.altitude)

        points: list[Vec3] = [
            Vec3(dx, dy, dz)
            for c in self.coordinates
            for dx, dz in (geo.to_local(c.lat, c.lon),)
        ]

        go = scene.spawn(f"Taxiway")
        go.add_component(LineRenderer.strip(points, color=self.color, depth_offset=True))
        return [go]

    def _spawn_mesh(self, scene: Scene, geo: GeoTransformer) -> list[GameObject]:
        dy: float = geo.altitude_to_local(self.altitude)
        half_w = (self.taxiway_width_m * self.horizontal_scale) / 2.0

        local_pts = [
            (dx, dy, dz)
            for c in self.coordinates
            for dx, dz in (geo.to_local(c.lat, c.lon),)
        ]

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
                # flat up-normal, consistent with RoadProjectMesh
                verts += [x, y, z, 0.0, 1.0, 0.0, u, v]

            # CCW as seen from +Y so the front face points up
            idx += [base, base + 2, base + 1, base, base + 3, base + 2]

        if not idx:
            return []

        mesh = Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))

        go = scene.spawn(f"Taxiway")
        go.add_component(MeshRenderer(mesh, SimpleMaterial(color=self.color)))
        return [go]

    @override
    def generate_mesh(self) -> Mesh:
        """
        cause the aircraft is represented as a point, no mesh is generated
        """
        pass
