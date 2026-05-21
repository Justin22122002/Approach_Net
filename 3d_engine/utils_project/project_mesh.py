import math
from abc import ABCMeta, abstractmethod
from typing import override

import numpy as np

from utils.mesh import Mesh
from pyproj import Transformer, Geod

from utils.vec3 import Vec3

class ProjectMesh(metaclass=ABCMeta):
    """
    Georeferenced mesh builder.

    Holds:
    - geographic anchor (lat/lon)
    - scaling policy
    - mesh generation logic
    """

    def __init__(
            self,
            horizontal_scale: float,
            vertical_scale: float,
            lat: float,
            lon: float,
            altitude: float,
    ) -> None:
        """
        :param horizontal_scale:the height scale converts the height in meter to the scale into the mesh
        :param vertical_scale: the plane scale converts the plane lat and lon into the scale of the mesh
        :param lat: the geographic anchor latitude should be in Wsg84
        :param lon: the geographic anchor longitude should be in Wsg84

        note: the lat and lon should serve as a reference point for later placing the mesh into the world
        """
        self.horizontal_scale: float = horizontal_scale
        self.vertical_scale: float = vertical_scale
        # reference point for the mesh
        self.lat: float = lat
        self.lon: float = lon
        # the altitude serves as an offset for the mesh in which it should be places later on,
        # e.g. for the plane the actual height or for the terrain the lowest point of the terrain
        self.altitude: float = altitude

    @abstractmethod
    def generate_mesh(self) -> Mesh:
        pass


class GeoTransformer:
    def __init__(
            self,
            lat: float,
            lon: float,
            horizontal_scale: float,
            vertical_scale: float,
    ) -> None:
        self.transformer = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:32632",
            always_xy=True
        )

        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale

        self.origin_x, self.origin_y = self.latlon_to_utm(lat, lon)

    def latlon_to_utm(self, lat: float, lon: float):
        return self.transformer.transform(lon, lat)

    def to_local(self, lat: float, lon: float):
        x, y = self.latlon_to_utm(lat, lon)

        dx = (x - self.origin_x) * self.horizontal_scale
        dz = (self.origin_y - y) * self.horizontal_scale

        return dx, dz

    def altitude_to_local(self, alt_m: float):
        return alt_m * self.vertical_scale

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

class RunwayProjectMesh(ProjectMesh):
    FT_TO_M = 0.3048

    def __init__(
        self,
        lat: float,
        lon: float,
        length_ft: float,
        width_ft: float,
        bearing_deg: float,
        altitude_ft: float,
        horizontal_scale: float,
        vertical_scale: float,
    ) -> None:
        self.geod = Geod(ellps="WGS84")

        self.length_m = length_ft * self.FT_TO_M
        self.width_m = width_ft * self.FT_TO_M
        self.altitude_m = altitude_ft * self.FT_TO_M
        self.bearing = bearing_deg

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=lat,
            lon=lon,
            altitude=self.altitude_m
        )

    def _compute_endpoints(self):
        start_lat = self.lat
        start_lon = self.lon

        end_lon, end_lat, _ = self.geod.fwd(
            start_lon,
            start_lat,
            self.bearing,
            self.length_m
        )

        return (start_lat, start_lon), (end_lat, end_lon)

    @override
    def generate_mesh(self) -> Mesh:
        (lat0, lon0), (lat1, lon1) = self._compute_endpoints()

        az, _, _ = self.geod.inv(lon0, lat0, lon1, lat1)

        half_width = self.width_m / 2.0

        left_az = az - 90
        right_az = az + 90

        p0_lon, p0_lat, _ = self.geod.fwd(lon0, lat0, left_az, half_width)
        p1_lon, p1_lat, _ = self.geod.fwd(lon0, lat0, right_az, half_width)

        p2_lon, p2_lat, _ = self.geod.fwd(lon1, lat1, right_az, half_width)
        p3_lon, p3_lat, _ = self.geod.fwd(lon1, lat1, left_az, half_width)

        geo = GeoTransformer(
            lat=self.lat,
            lon=self.lon,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale
        )

        def proj(lat, lon):
            x, z = geo.to_local(lat, lon)
            return Vec3(x, 0.0, z)

        v0 = proj(p0_lat, p0_lon)
        v1 = proj(p1_lat, p1_lon)
        v2 = proj(p2_lat, p2_lon)
        v3 = proj(p3_lat, p3_lon)

        verts = []
        idx = [0, 1, 2, 0, 2, 3]

        for v in [v0, v1, v2, v3]:
            verts += [
                v.x, 0.0, v.z,
                0.0, 1.0, 0.0,
                0.0, 0.0
            ]

        return Mesh(
            np.array(verts, dtype=np.float32),
            np.array(idx, dtype=np.uint32)
        )

def utm_epsg(lat: float, lon: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"