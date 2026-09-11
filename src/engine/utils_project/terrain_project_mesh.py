from typing import override

import numpy as np
from pyproj import Transformer

from engine.utils.color import Color
from engine.utils.gameObject import GameObject
from engine.utils.simplematerial import SimpleMaterial
from engine.utils.mesh import Mesh
from engine.utils.meshRenderer import MeshRenderer
from engine.utils.scene import Scene
from engine.utils.texture import Texture
from engine.utils.vec3 import Vec3
from engine.utils_project.geo_transformer import GeoTransformer
from engine.utils_project.project_mesh import ProjectMesh


class TerrainProjectMesh(ProjectMesh):

    def __init__(
        self,
        grid: np.ndarray,
        lat_min: float,
        lon_min: float,
        lat_max: float,
        lon_max: float,
        horizontal_scale: float,
        vertical_scale: float,
        transformer: Transformer,
        texture: Texture | None = None,
    ) -> None:
        self.grid: np.ndarray = grid
        self.lat_min: float = lat_min
        self.lon_min: float = lon_min
        self.lat_max: float = lat_max
        self.lon_max: float = lon_max
        self._texture: Texture | None = texture

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=lat_min,
            lon=lon_min,
            altitude=float(grid.min()),
            transformer=transformer
        )

        # lat_max(y1)    ____________________
        #               |                    |
        #               |                    |
        #               |                    |
        #               |                    |
        # lat_min (y0)  |____________________|
        #          lon_min (x0)              lon_max (x1)
        #
        self.x0, self.y0 = self.transformer.transform(lon_min, lat_min)
        self.x1, _  = self.transformer.transform(lon_max, lat_min)
        _, self.y1  = self.transformer.transform(lon_min, lat_max)

        self.width_m: float = abs(self.x1 - self.x0)
        self.height_m: float = abs(self.y1 - self.y0)

    def sample_height(self, lat: float, lon: float) -> float:
        rows, cols = self.grid.shape

        j = (self.lat_max - lat) / (self.lat_max - self.lat_min) * (rows - 1)
        i = (lon - self.lon_min) / (self.lon_max - self.lon_min) * (cols - 1)

        j = max(0.0, min(float(rows - 1), j))
        i = max(0.0, min(float(cols - 1), i))

        j0, i0 = int(j), int(i)
        j1, i1 = min(j0 + 1, rows - 1), min(i0 + 1, cols - 1)
        tj, ti = j - j0, i - i0

        h00 = float(self.grid[j0, i0])
        h10 = float(self.grid[j0, i1])
        h01 = float(self.grid[j1, i0])
        h11 = float(self.grid[j1, i1])

        h0 = h00 * (1.0 - ti) + h10 * ti
        h1 = h01 * (1.0 - ti) + h11 * ti
        return h0 * (1.0 - tj) + h1 * tj

    @override
    def spawn(self, scene: Scene, geo: GeoTransformer) -> list[GameObject]:
        dx, dz = geo.to_local(self.lat, self.lon)
        dy = geo.altitude_to_local(self.altitude)

        go = scene.spawn("Terrain")
        go.transform.position = Vec3(dx, dy, dz)

        texture = SimpleMaterial(texture=self._texture) if self._texture is not None \
            else SimpleMaterial(color=Color(0.4, 0.55, 0.3))

        go.add_component(MeshRenderer(
            self.generate_mesh(),
            texture

        ))
        return [go]

    @override
    def generate_mesh(self) -> Mesh:
        rows, cols = self.grid.shape
        min_h = float(self.grid.min())

        verts = []
        idx = []

        def h(r, c):
            r = max(0, min(rows - 1, r))
            c = max(0, min(cols - 1, c))
            return float(self.grid[r, c]) - min_h

        for j in range(rows):
            for i in range(cols):
                lon = self.lon_min + (i / (cols - 1)) * (self.lon_max - self.lon_min)

                lat = self.lat_max - (j / (rows - 1)) * (self.lat_max - self.lat_min)

                ux, uy = self.transformer.transform(lon, lat)

                x = (ux - self.x0) * self.horizontal_scale
                z = -(uy - self.y0) * self.horizontal_scale

                y = h(j, i) * self.vertical_scale

                # compute change in height for normals of the mesh
                dx = self.width_m / (cols - 1)
                dz = self.height_m / (rows - 1)

                hl = h(j, i - 1) * self.vertical_scale
                hr = h(j, i + 1) * self.vertical_scale
                hd = h(j + 1, i) * self.vertical_scale
                hu = h(j - 1, i) * self.vertical_scale

                sx = (hr - hl) / (2.0 * dx * self.horizontal_scale)
                sz = (hd - hu) / (2.0 * dz * self.horizontal_scale)

                n = Vec3(-sx, 1.0, -sz).normalized()

                verts += [
                    x, y, z,
                    n.x, n.y, n.z,
                    i / (cols - 1),
                    j / (rows - 1)
                ]

        for j in range(rows - 1):
            for i in range(cols - 1):
                tl = j * cols + i
                idx += [
                    tl, tl + cols, tl + 1,
                        tl + 1, tl + cols, tl + cols + 1
                ]

        return Mesh(
            np.array(verts, dtype=np.float32),
            np.array(idx, dtype=np.uint32)
        )

# old normal code:
# def height_raw(r, c):
#     r = max(0, min(rows - 1, r))
#     c = max(0, min(cols - 1, c))
#     return float(self.grid[r, c])
#
# hx = height_raw(j, i + 1) - height_raw(j, i - 1)
# hz = height_raw(j + 1, i) - height_raw(j - 1, i)
# n = Vec3(-hx, 2.0, -hz).normalized()