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
        horizontal_scale: float = 1.0,
        vertical_scale: float = 1.0
    ) -> None:
        self.grid = grid
        self.lat_min = lat_min
        self.lon_min = lon_min
        self.lat_max = lat_max
        self.lon_max = lon_max

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=lat_min,
            lon=lon_min,
            altitude=float(grid.min())
        )

        # WGS 84 / UTM zone xxx
        utm = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)


        # lat_max(y1)    ____________________
        #               |                    |
        #               |                    |
        #               |                    |
        #               |                    |
        # lat_min (y0)  |____________________|
        #          lon_min (x0)              lon_max (x1)
        #
        x0, y0 = utm.transform(lon_min, lat_min)
        x1, _  = utm.transform(lon_max, lat_min)
        _, y1  = utm.transform(lon_min, lat_max)

        self.width_m = abs(x1 - x0)
        self.height_m = abs(y1 - y0)
        self.base_altitude = float(grid.min())

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
                x = (i / (cols - 1)) * self.width_m * self.horizontal_scale
                z = -((rows - 1 - j) / (rows - 1)) * self.height_m * self.horizontal_scale
                y = h(j, i) * self.vertical_scale

                hx = height_raw(j, i + 1) - height_raw(j, i - 1)
                hz = height_raw(j + 1, i) - height_raw(j - 1, i)

                n = Vec3(-hx, 2.0, -hz).normalized() # or remove .normalized()

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