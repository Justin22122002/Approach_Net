from pathlib import Path
from typing import override

from pyproj import Transformer, crs

from utils.aircraft import AircraftData
from utils.mesh import Mesh
from utils_project.project_mesh import ProjectMesh


class AircraftProjectMesh(ProjectMesh):
    """
    The vertical reference datum is the Earth Gravitational Model 20081 (EGM2008; EPSG 3855).

    PROJ_DOWNLOAD_DIR="$(python -c 'import pyproj; print(pyproj.datadir.get_user_data_dir())')"
    cd "$PROJ_DOWNLOAD_DIR"
    curl -O 'https://cdn.proj.org/us_nga_egm08_25.tif'

    see: https://gis.stackexchange.com/questions/454317/how-to-convert-wgs84-from-ellipsoidal-height-to-mean-sea-level-i-e-height-above
    """
    def __init__(
            self,
            aircraft_data: AircraftData,
            horizontal_scale: float,
            vertical_scale: float,
            transformer: Transformer,
    ) -> None:
        assert aircraft_data.latitude is not None
        assert aircraft_data.longitude is not None

        self.aircraft_data: AircraftData = aircraft_data

        tform = Transformer.from_crs(
            "epsg:4979",
            "epsg:4326+3855",
            always_xy=True
        )

        _, _, new_height = tform.transform(
            aircraft_data.longitude,
            aircraft_data.latitude,
            aircraft_data.geoaltitude
        )

        print(f"new_height: {new_height}, old_height: {aircraft_data.geoaltitude}")
        aircraft_data.geoaltitude = new_height

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
        """
        cause aircrafts are represented as points, no mesh is generated
        """
        pass