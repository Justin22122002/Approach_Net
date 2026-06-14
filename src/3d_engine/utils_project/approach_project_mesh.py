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
            lat=0.0,
            lon=0.0,
            altitude=0.0, # note: this is just a default value cause the approch consist out of multiple points
            transformer=transformer
        )

    @override
    def spawn(self, scene: Scene, geo_transformer: GeoTransformer) -> list[GameObject]:
        spawned = []
        points: list[Vec3] = []

        entry_fixes, fix_before, rw_fix = self.approach.get_final_approach_fixes()

        special_colors: dict[str, Color] = {}
        if rw_fix:
            special_colors[rw_fix.fix_identifier] = Color(0.0, 0.0, 0.0)
        if fix_before:
            special_colors[fix_before.fix_identifier] = Color(1.0, 1.0, 0.0)
        for entry_fix in entry_fixes:
            special_colors[entry_fix.fix_identifier] = Color(0.0, 0.5, 1.0)

        for fix in self.approach.fixes:
            if fix.latitude is None or fix.longitude is None:
                continue
            if not fix.fix_identifier:
                continue
            if fix.latitude == 0.0 and fix.longitude == 0.0:
                continue

            dx, dz = geo_transformer.to_local(
                fix.latitude,
                fix.longitude
            )

            dy = geo_transformer.altitude_to_local(
                fix.altitude * self.FT_TO_M
            )

            points.append(Vec3(dx, dy, dz))

            fix_color = special_colors.get(fix.fix_identifier, self.color)

            label_go = scene.spawn(f"label_{fix.fix_identifier}")
            label_go.transform.position = Vec3(dx, dy + 1, dz)
            label_go.add_component(TextRenderer(
                text=f"{fix.fix_identifier}", # self.approach.procedure_identifier
                font_size=48,
                color=fix_color,
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