from pyproj import Transformer
from typing_extensions import override

from utils.color import Color
from utils.mesh import Mesh
from utils.pointRenderer import PointRenderer
from utils.vec3 import Vec3

from utils.approch import Approach
from utils_project.geo_transformer import GeoTransformer
from utils_project.project_mesh import ProjectMesh


class ApproachProjectMesh(ProjectMesh):
    def __init__(
        self,
        approach: Approach,
        airport_lat: float,
        airport_lon: float,
        airport_altitude_ft: float,
        horizontal_scale: float,
        vertical_scale: float,
        transformer: Transformer,
        color: Color = None,
    ) -> None:
        self.approach: Approach = approach

        self.color: Color = color or Color(1.0, 0.0, 0.0)

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=airport_lat,
            lon=airport_lon,
            altitude=airport_altitude_ft * self.FT_TO_M, # note: this is the airport height, cause one approach has multiple heights for each fix/way-point
            transformer=transformer
        )

    # todo: add abstract method in ProjectMesh with Renderer as return type with generic specification of this method
    def generate_renderer(
        self,
        geo_transformer: GeoTransformer
    ) -> PointRenderer:
        points: list[Vec3] = []

        for fix in self.approach.fixes:
            if fix.latitude is None or fix.longitude is None:
                print(f"missing fix: {fix}")
                continue

            dx, dz = geo_transformer.to_local(
                fix.latitude,
                fix.longitude
            )

            altitude_m = fix.altitude * self.FT_TO_M

            dy = geo_transformer.altitude_to_local(
                altitude_m
            )

            points.append(
                Vec3(dx, dy, dz)
            )

        if len(points) < 2:
            return PointRenderer(
                [Vec3.zero(), Vec3.zero()],
                color=self.color,
                size=200.0
            )

        return PointRenderer(
            points=points,
            color=self.color,
            size=200.0
        )

    @override
    def generate_mesh(self) -> Mesh:
        """
        approaches are rendered as lines only
        """
        pass