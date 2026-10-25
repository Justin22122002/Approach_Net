from typing import override

from pyproj import Transformer

from utils.aircraft import Aircraft

from engine.utils.color import Color
from engine.utils.gameObject import GameObject
from engine.utils.mesh import Mesh
from engine.utils.pointRenderer import PointRenderer
from engine.utils.scene import Scene
from engine.utils.vec3 import Vec3
from engine.utils_project.geo_transformer import GeoTransformer
from engine.utils_project.project_mesh import ProjectMesh


class AircraftProjectMesh(ProjectMesh):
    def __init__(
            self,
            aircraft_data: Aircraft,
            horizontal_scale: float,
            vertical_scale: float,
            transformer: Transformer,
    ) -> None:
        assert aircraft_data.latitude is not None
        assert aircraft_data.longitude is not None

        self.aircraft_data: Aircraft = aircraft_data

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=aircraft_data.latitude,
            lon=aircraft_data.longitude,
            altitude=aircraft_data.geoaltitude,
            transformer=transformer
        )

    @override
    def spawn(self, scene: Scene, geo: GeoTransformer) -> list[GameObject]:
        x, z = geo.to_local(self.lat, self.lon)
        y = geo.altitude_to_local(self.altitude)

        go = scene.spawn(f"Aircraft_{self.aircraft_data.icao24}")
        go.transform.position = Vec3(x, y, z)
        go.add_component(PointRenderer(color=self._color(), size=200.0))
        return [go]

    def _color(self) -> Color:
        d = self.aircraft_data
        if d.onground and d.groundspeed == 0.0:  return Color(0.5, 0.5, 0.5)  # parked
        if d.onground and d.groundspeed > 0.0:  return Color(1.0, 1.0, 0.0)  # rolls on ground
        return Color(0.0, 1.0, 0.0) # in air

    def update_position(self, go: GameObject, geo: GeoTransformer) -> None:
        x, z = geo.to_local(self.lat, self.lon)
        y = geo.altitude_to_local(self.altitude)
        go.transform.position = Vec3(x, y, z)

    @override
    def generate_mesh(self) -> Mesh:
        """
        cause the aircraft is represented as a point, no mesh is generated
        """
        pass