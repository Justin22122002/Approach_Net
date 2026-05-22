from typing import override

from pyproj import Transformer

from utils.aircraft import AircraftData
from utils.mesh import Mesh
from utils_project.project_mesh import ProjectMesh


class AircraftProjectMesh(ProjectMesh):
    def __init__(
            self,
            aircraft_data: AircraftData,
            horizontal_scale: float,
            vertical_scale: float,
            transformer: Transformer,
    ) -> None:
        assert aircraft_data.latitude is not None
        assert aircraft_data.longitude is not None

        aircraft_data: AircraftData = aircraft_data

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=aircraft_data.latitude,
            lon=aircraft_data.longitude,
            altitude=aircraft_data.geoaltitude,
            transformer=transformer
        )

    @override
    def generate_mesh(self) -> Mesh:
        pass