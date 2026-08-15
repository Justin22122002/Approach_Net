from typing import override

from pyproj import Transformer

from utils.airport_terminal import AirportTerminal
from utils.color import Color
from utils.gameObject import GameObject
from utils.scene import Scene
from utils_project.building_project_mesh import BuildingProjectMesh
from utils_project.geo_transformer import GeoTransformer
from utils_project.terrain_project_mesh import TerrainProjectMesh


class TerminalProjectMesh(BuildingProjectMesh):
    def __init__(
        self,
        terminals: list[AirportTerminal],
        terrain: TerrainProjectMesh,
        horizontal_scale: float,
        vertical_scale: float,
        transformer: Transformer,
        color: Color = None,
        building_height_m: float = 15.0,
        height_offset_m: float = 0.15,
        min_footprint_m: float = 2.0,
    ) -> None:
        super().__init__(
            buildings=[t.coordinates for t in terminals],
            terrain=terrain,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            transformer=transformer,
            color=color or Color(0.447, 0.235, 0.110),
            building_height_m=building_height_m,
            height_offset_m=height_offset_m,
            min_footprint_m=min_footprint_m,
        )

    @override
    def spawn(self, scene: Scene, geo: GeoTransformer) -> list[GameObject]:
        objects = super().spawn(scene, geo)
        for go in objects:
            go.name = "Terminals"
        return objects