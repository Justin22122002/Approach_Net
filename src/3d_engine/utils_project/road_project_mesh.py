from typing import override

import numpy as np
from pyproj import Transformer

from utils.color import Color
from utils.gameObject import GameObject
from utils.geo_types import Coordinate
from utils.simplematerial import SimpleMaterial
from utils.mesh import Mesh
from utils.meshRenderer import MeshRenderer
from utils.scene import Scene
from utils.vec3 import Vec3
from utils_project.geo_transformer import GeoTransformer
from utils_project.project_mesh import ProjectMesh
from utils_project.terrain_project_mesh import TerrainProjectMesh


class RoadProjectMesh(ProjectMesh):
    def __init__(
        self,
        roads: list[list[Coordinate]],
        terrain: TerrainProjectMesh,
        horizontal_scale: float,
        vertical_scale: float,
        transformer: Transformer,
        color: Color = None,
        road_width_m: float = 6.0,
        height_offset_m: float = 5.0,
    ) -> None:
        self.roads: list[list[Coordinate]] = roads
        self.terrain: TerrainProjectMesh = terrain
        self.color: Color = color or Color(0.25, 0.25, 0.25)
        self.road_width_m: float = road_width_m
        self.height_offset_m: float = height_offset_m

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=terrain.lat_min,
            lon=terrain.lon_min,
            altitude=terrain.altitude,
            transformer=transformer,
        )

        self.origin_x, self.origin_y = self.transformer.transform(self.lon, self.lat)

    def _local_xz(self, lat: float, lon: float) -> tuple[float, float]:
        x, y = self.transformer.transform(lon, lat)
        dx = (x - self.origin_x) * self.horizontal_scale
        dz = (self.origin_y - y) * self.horizontal_scale
        return dx, dz

    @override
    def spawn(self, scene: Scene, geo: GeoTransformer) -> list[GameObject]:
        mesh = self.generate_mesh()
        if mesh.vertices.size == 0:
            return []

        cx, cz = geo.to_local(self.lat, self.lon)
        cy = geo.altitude_to_local(self.altitude)

        go = scene.spawn("Roads")
        go.transform.position = Vec3(cx, cy, cz)
        go.add_component(MeshRenderer(mesh, SimpleMaterial(color=self.color)))
        return [go]

    @override
    def generate_mesh(self) -> Mesh:
        half_w = (self.road_width_m * self.horizontal_scale) / 2.0

        verts: list[float] = []
        idx: list[int] = []

        for line in self.roads:
            if len(line) < 2:
                continue

            local_pts = []
            for c in line:
                dx, dz = self._local_xz(c.lat, c.lon)
                h_m = self.terrain.sample_height(c.lat, c.lon)
                dy = (h_m - self.altitude + self.height_offset_m) * self.vertical_scale
                local_pts.append((dx, dy, dz))

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