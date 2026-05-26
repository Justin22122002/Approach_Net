from pyproj import Transformer
from typing_extensions import override

from utils.geo_types import Coordinate
from utils.lineRenderer import LineRenderer
from utils.mesh import Mesh
from utils.color import Color
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
    ) -> None:
        self.coordinates: list[Coordinate]  = coordinates
        self.color: Color                   = color or Color(0.9, 0.9, 0.1)

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=airport_lat,
            lon=airport_lon,
            altitude=airport_altitude_ft * self.FT_TO_M,
            transformer=transformer
        )

    def generate_renderer(
            self,
            geo_transformer: GeoTransformer
    ) -> LineRenderer:
        if len(self.coordinates) < 2:
            return LineRenderer([Vec3.zero(), Vec3.zero()], color=self.color)

        dy: float = geo_transformer.altitude_to_local(self.altitude)

        points: list[Vec3] = [
            Vec3(dx, dy, dz)
            for c in self.coordinates
            for dx, dz in (geo_transformer.to_local(c.lat, c.lon),)
        ]

        return LineRenderer.strip(points, color=self.color)

    @override
    def generate_mesh(self) -> Mesh:
        """
        cause the aircraft is represented as a point, no mesh is generated
        """
        pass
