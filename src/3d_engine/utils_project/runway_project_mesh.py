from typing import override

import numpy as np

from utils.arinc_runway import ArincRunway
from utils.color import Color
from utils.gameObject import GameObject
from utils.simplematerial import SimpleMaterial
from utils.mesh import Mesh
from pyproj import Transformer, Geod

from utils.meshRenderer import MeshRenderer
from utils.scene import Scene
from utils.vec3 import Vec3
from utils_project.geo_transformer import GeoTransformer
from utils_project.project_mesh import ProjectMesh


class RunwayProjectMesh(ProjectMesh):
    def __init__(
        self,
        runway: ArincRunway,
        horizontal_scale: float,
        vertical_scale: float,
        transformer: Transformer,
    ) -> None:
        self.geod = Geod(ellps="WGS84")
        self.runway: ArincRunway = runway
        self.length_m: float = runway.length * self.FT_TO_M
        self.width_m: float = runway.width * self.FT_TO_M
        self.displaced_threshold_m: float = runway.displaced_threshold * self.FT_TO_M
        self.bearing: float = runway.bearing

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=runway.latitude,
            lon=runway.longitude,
            altitude=runway.runway_altitude * self.FT_TO_M,
            transformer=transformer
        )

    def _compute_endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        start_lat = self.lat
        start_lon = self.lon

        start_lon, start_lat, _ = self.geod.fwd( # this is for api.flightplandatabase.com
            start_lon,
            start_lat,
            self.bearing,
            self.displaced_threshold_m
        )

        end_lon, end_lat, _ = self.geod.fwd(
            start_lon,
            start_lat,
            self.bearing,
            self.length_m
        )

        return (start_lat, start_lon), (end_lat, end_lon)

    @override
    def spawn(self, scene: Scene, geo: GeoTransformer) -> list[GameObject]:
        cx, cz = geo.to_local(self.lat, self.lon)
        cy = geo.altitude_to_local(self.altitude)

        go = scene.spawn("Runway")
        go.transform.position = Vec3(cx, cy, cz)
        go.add_component(MeshRenderer(self.generate_mesh(), SimpleMaterial(color=Color.gray())))
        return [go]

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

        def proj(lat, lon):
            x, y = self.transformer.transform(lon, lat)

            x0, y0 = self.transformer.transform(self.lon, self.lat)

            dx = (x - x0) * self.horizontal_scale
            dz = (y - y0) * self.horizontal_scale

            return Vec3(dx, 0.0, -dz)

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
