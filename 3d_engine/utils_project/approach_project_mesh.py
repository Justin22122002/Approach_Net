from pyproj import Transformer
from typing_extensions import override

from utils.color import Color
from utils.gameObject import GameObject
from utils.mesh import Mesh
from utils.pointRenderer import PointRenderer
from utils.scene import Scene
from utils.text_renderer import TextRenderer
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

    @override
    def spawn(self, scene: Scene, geo_transformer: GeoTransformer) -> list[GameObject]:
        spawned = []
        points: list[Vec3] = []

        for fix in self.approach.fixes:
            if fix.latitude is None or fix.longitude is None:
                continue

            dx, dz = geo_transformer.to_local(
                fix.latitude,
                fix.longitude
            )

            dy = geo_transformer.altitude_to_local(
                fix.altitude * self.FT_TO_M
            )

            points.append(Vec3(dx, dy, dz))

            label_go = scene.spawn(f"label_{fix.fix_identifier}")
            label_go.transform.position = Vec3(dx, dy + 1, dz)
            label_go.add_component(TextRenderer(
                text=fix.fix_identifier,
                font_size = 48,
                color=self.color,
                billboard=True,
                world_height=0.6,
            ))
            spawned.append(label_go)

        approach_go = scene.spawn(f"Approach_{self.approach.procedure_identifier}")
        approach_go.add_component(PointRenderer(
            points=points,
            color=self.color,
            size=200.0,
        ))
        spawned.append(approach_go)
        return spawned

    @override
    def generate_mesh(self) -> Mesh:
        """
        approaches are rendered as lines only
        """
        pass