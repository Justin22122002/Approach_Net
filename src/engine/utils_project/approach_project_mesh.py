from pyproj import Transformer
from typing_extensions import override

from engine.utils.color import Color
from engine.utils.gameObject import GameObject
from engine.utils.lineRenderer import LineRenderer
from engine.utils.mesh import Mesh
from engine.utils.pointRenderer import PointRenderer
from engine.utils.scene import Scene
from engine.utils.text_renderer import TextRenderer
from engine.utils.vec3 import Vec3
from engine.utils_project.geo_transformer import GeoTransformer
from engine.utils_project.project_mesh import ProjectMesh

from utils.approch import Approach
from utils.fix_identifier import FixIdentifier


class ApproachProjectMesh(ProjectMesh):
    def __init__(
        self,
        approach: Approach,
        horizontal_scale: float,
        vertical_scale: float,
        transformer: Transformer,
        color: Color = None,
        connect_fixes: bool = False,
        line_color: Color | None = None,
        line_width: float = 2.0,
        transition_depth: int | None = None,
    ) -> None:
        self.approach: Approach = approach

        self.color: Color = color or Color(1.0, 0.0, 0.0)
        self.connect_fixes: bool = connect_fixes
        self.line_color: Color = line_color or self.color
        self.line_width: float = line_width
        self.transition_depth: int | None = transition_depth

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=0.0,
            lon=0.0,
            altitude=0.0, # note: this is just a default value cause the approch consist out of multiple points
            transformer=transformer
        )

    @staticmethod
    def _is_valid(fix: FixIdentifier) -> bool:
        if fix.latitude is None or fix.longitude is None:
            return False
        if not fix.fix_identifier:
            return False
        if fix.latitude == 0.0 and fix.longitude == 0.0:
            return False
        return True

    def _fix_to_vec3(self, fix: FixIdentifier, geo_transformer: GeoTransformer) -> Vec3:
        dx, dz = geo_transformer.to_local(fix.latitude, fix.longitude)
        dy = geo_transformer.altitude_to_local(fix.altitude * self.FT_TO_M)
        return Vec3(dx, dy, dz)

    def _spawn_final_lines(
        self,
        scene: Scene,
        geo_transformer: GeoTransformer,
        entry_fixes: list[FixIdentifier],
        faf_fix: FixIdentifier | None,
        rw_fix: FixIdentifier | None,
    ) -> list[GameObject]:
        if_fix = next((f for f in entry_fixes if f.get_fix_type() == "IF"), None)

        chain = [f for f in (if_fix, faf_fix, rw_fix) if f is not None and self._is_valid(f)]

        polylines: list[tuple[str, list[FixIdentifier]]] = []
        if len(chain) >= 2:
            polylines.append(("final", chain))

        if chain:
            anchor = chain[0]
            seen: set[tuple[str, ...]] = set()
            for i, segment in enumerate(self.approach._split_into_segments()):
                j = next((k for k, f in enumerate(segment) if f.fix_identifier == anchor.fix_identifier), None)
                if not j:  # anchor not in this segment or nothing before it
                    continue
                start = 0 if self.transition_depth is None else max(0, j - self.transition_depth)
                path = [f for f in segment[start:j] if self._is_valid(f)] + [anchor]
                key = tuple(f.fix_identifier for f in path)
                if len(path) >= 2 and key not in seen:
                    seen.add(key)
                    polylines.append((f"transition_{i}", path))

        spawned: list[GameObject] = []
        for name, fixes in polylines:
            pts = [self._fix_to_vec3(f, geo_transformer) for f in fixes]
            go = scene.spawn(f"ApproachLine_{self.approach.procedure_identifier}_{name}")
            go.add_component(LineRenderer.strip(pts, color=self.line_color, line_width=self.line_width))
            spawned.append(go)
        return spawned

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
            if not self._is_valid(fix):
                continue

            pos = self._fix_to_vec3(fix, geo_transformer)
            points.append(pos)

            fix_color = special_colors.get(fix.fix_identifier, self.color)

            label_go = scene.spawn(f"label_{fix.fix_identifier}")
            label_go.transform.position = Vec3(pos.x, pos.y + 1, pos.z)
            label_go.add_component(TextRenderer(
                text=f"{fix.fix_identifier}", # self.approach.procedure_identifier
                font_size=48,
                color=fix_color,
                billboard=True,
                world_height=0.6,
            ))
            spawned.append(label_go)

        if self.connect_fixes:
            spawned += self._spawn_final_lines(scene, geo_transformer, entry_fixes, fix_before, rw_fix)

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
        approaches are rendered as points / lines only
        """
        pass