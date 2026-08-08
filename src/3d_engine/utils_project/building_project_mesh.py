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
from utils_project.geo_transformer import GeoTransformer
from utils_project.project_mesh import ProjectMesh
from utils_project.terrain_project_mesh import TerrainProjectMesh


def _signed_area_xz(pts: list[tuple[float, float]]) -> float:
    a = 0.0
    n = len(pts)
    for i in range(n):
        x0, z0 = pts[i]
        x1, z1 = pts[(i + 1) % n]
        a += x0 * z1 - x1 * z0
    return a * 0.5


def _ear_clip(pts: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    n = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    def cross2(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def point_in_tri(p, a, b, c):
        d1 = cross2(a, b, p)
        d2 = cross2(b, c, p)
        d3 = cross2(c, a, p)
        neg = d1 < 0 or d2 < 0 or d3 < 0
        pos = d1 > 0 or d2 > 0 or d3 > 0
        return not (neg and pos)

    indices = list(range(n))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(indices) > 3 and guard < n * n + 10:
        guard += 1
        ear_found = False
        m = len(indices)
        for k in range(m):
            i_prev = indices[k - 1]
            i_curr = indices[k]
            i_next = indices[(k + 1) % m]
            if cross2(pts[i_prev], pts[i_curr], pts[i_next]) <= 0:
                continue  # reflex vertex, can't be an ear
            if any(
                j not in (i_prev, i_curr, i_next)
                and point_in_tri(pts[j], pts[i_prev], pts[i_curr], pts[i_next])
                for j in indices
            ):
                continue
            triangles.append((i_prev, i_curr, i_next))
            del indices[k]
            ear_found = True
            break
        if not ear_found:
            break
    if len(indices) >= 3:
        for k in range(1, len(indices) - 1):
            triangles.append((indices[0], indices[k], indices[k + 1]))
    return triangles


class BuildingProjectMesh(ProjectMesh):
    def __init__(
        self,
        buildings: list[list[Coordinate]],
        terrain: TerrainProjectMesh,
        horizontal_scale: float,
        vertical_scale: float,
        transformer: Transformer,
        color: Color = None,
        building_height_m: float = 8.0,
        height_offset_m: float = 0.1,
        min_footprint_m: float = 2.0,
    ) -> None:
        self.buildings: list[list[Coordinate]] = buildings
        self.terrain: TerrainProjectMesh = terrain
        self.color: Color = color or Color.white()
        self.building_height_m: float = building_height_m
        self.height_offset_m: float = height_offset_m
        self.min_footprint_m: float = min_footprint_m

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=terrain.lat_min,
            lon=terrain.lon_min,
            altitude=0.0,
            transformer=transformer,
        )

    @override
    def spawn(self, scene: Scene, geo: GeoTransformer) -> list[GameObject]:
        verts: list[float] = []
        idx: list[int] = []

        for ring in self.buildings:
            if len(ring) < 3:
                continue

            pts_xz = [geo.to_local(c.lat, c.lon) for c in ring]
            if len(pts_xz) > 1 and pts_xz[0] == pts_xz[-1]:
                pts_xz = pts_xz[:-1]
            if len(pts_xz) < 3:
                continue

            xs = [p[0] for p in pts_xz]
            zs = [p[1] for p in pts_xz]
            width_m = (max(xs) - min(xs)) / self.horizontal_scale
            depth_m = (max(zs) - min(zs)) / self.horizontal_scale
            if width_m < self.min_footprint_m or depth_m < self.min_footprint_m:
                continue

            cx = sum(xs) / len(xs)
            cz = sum(zs) / len(zs)

            base_h_m = max(self.terrain.sample_height(c.lat, c.lon) for c in ring)
            y0 = geo.altitude_to_local(base_h_m) + self.height_offset_m * self.vertical_scale
            y1 = y0 + self.building_height_m * self.vertical_scale

            n = len(pts_xz)

            for i in range(n):
                x0, z0 = pts_xz[i]
                x1, z1 = pts_xz[(i + 1) % n]
                if (x0 - x1) ** 2 + (z0 - z1) ** 2 < 1e-12:
                    continue

                v0 = (x0, y0, z0)
                v1 = (x1, y0, z1)
                v2 = (x1, y1, z1)
                v3 = (x0, y1, z0)

                mx, mz = (x0 + x1) / 2.0, (z0 + z1) / 2.0
                outward = (mx - cx, mz - cz)

                base = len(verts) // 8
                for (vx, vy, vz), (u, v) in zip((v0, v1, v2, v3), ((0, 0), (1, 0), (1, 1), (0, 1))):
                    verts += [vx, vy, vz, 0.0, 0.0, 0.0, u, v]  # normal patched below

                ax, ay, az = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
                bx, by, bz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
                nx = ay * bz - az * by
                nz = ax * by - ay * bx
                nlen = (nx * nx + nz * nz) ** 0.5 or 1.0
                nx, nz = nx / nlen, nz / nlen
                for k in range(4):
                    verts[(base + k) * 8 + 3] = nx
                    verts[(base + k) * 8 + 5] = nz

                if nx * outward[0] + nz * outward[1] >= 0:
                    idx += [base, base + 1, base + 2, base, base + 2, base + 3]
                else:
                    idx += [base, base + 2, base + 1, base, base + 3, base + 2]

            roof_pts = pts_xz if _signed_area_xz(pts_xz) > 0 else list(reversed(pts_xz))
            tris = _ear_clip(roof_pts)
            if tris:
                roof_base = len(verts) // 8
                for x, z in roof_pts:
                    verts += [x, y1, z, 0.0, 1.0, 0.0, 0.0, 0.0]

                a0, b0, c0 = tris[0]
                ax_, az_ = roof_pts[b0][0] - roof_pts[a0][0], roof_pts[b0][1] - roof_pts[a0][1]
                bx_, bz_ = roof_pts[c0][0] - roof_pts[a0][0], roof_pts[c0][1] - roof_pts[a0][1]
                ny = az_ * bx_ - ax_ * bz_
                flip = ny < 0

                for a, b, c in tris:
                    if flip:
                        idx += [roof_base + a, roof_base + c, roof_base + b]
                    else:
                        idx += [roof_base + a, roof_base + b, roof_base + c]

        if not idx:
            return []

        mesh = Mesh(np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32))

        go = scene.spawn("Buildings")
        go.add_component(MeshRenderer(mesh, SimpleMaterial(color=self.color)))
        return [go]

    @override
    def generate_mesh(self) -> Mesh:
        pass