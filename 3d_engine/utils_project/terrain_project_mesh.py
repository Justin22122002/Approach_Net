from typing import override

import numpy as np
from pyproj import Transformer

from utils.mesh import Mesh
from utils.vec3 import Vec3
from utils_project.project_mesh import ProjectMesh


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
    ) -> None:
        self.grid: np.ndarray = grid
        self.lat_min: float = lat_min
        self.lon_min: float = lon_min
        self.lat_max: float = lat_max
        self.lon_max: float = lon_max

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

        def height_raw(r, c):
            r = max(0, min(rows - 1, r))
            c = max(0, min(cols - 1, c))
            return float(self.grid[r, c])

        for j in range(rows):
            for i in range(cols):
                lon = self.lon_min + (i / (cols - 1)) * (self.lon_max - self.lon_min)

                lat = self.lat_max - (j / (rows - 1)) * (self.lat_max - self.lat_min)

                ux, uy = self.transformer.transform(lon, lat)

                x = (ux - self.x0) * self.horizontal_scale
                z = -(uy - self.y0) * self.horizontal_scale

                y = h(j, i) * self.vertical_scale

                hx = height_raw(j, i + 1) - height_raw(j, i - 1)
                hz = height_raw(j + 1, i) - height_raw(j - 1, i)

                n = Vec3(-hx, 2.0, -hz).normalized()

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