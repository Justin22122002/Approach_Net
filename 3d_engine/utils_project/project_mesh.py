from abc import ABCMeta, abstractmethod

from pyproj import Transformer

from utils.gameObject import GameObject
from utils.mesh import Mesh
from utils.scene import Scene
from utils_project.geo_transformer import GeoTransformer


class ProjectMesh(metaclass=ABCMeta):
    """
    Georeferenced mesh builder.

    Holds:
    - geographic anchor (lat/lon)
    - scaling policy
    - mesh generation logic

    note: every mesh or object should use wsg84 lat and lon and msl for the height reference
    """
    FT_TO_M: float = 0.3048

    def __init__(
            self,
            horizontal_scale: float,
            vertical_scale: float,
            lat: float,
            lon: float,
            altitude: float,
            transformer: Transformer
    ) -> None:
        """
        :param horizontal_scale:the height scale converts the height in meter to the scale into the mesh
        :param vertical_scale: the plane scale converts the plane lat and lon into the scale of the mesh
        :param lat: the geographic anchor latitude should be in Wsg84
        :param lon: the geographic anchor longitude should be in Wsg84

        note: the lat and lon should serve as a reference point for later placing the mesh into the world
        note: also note that the mesh should take 0,0,0 as a reference point, and later it can be
              translated with the GeoTransformer class to the right position
        """
        self.horizontal_scale: float = horizontal_scale
        self.vertical_scale: float = vertical_scale
        # reference point for the mesh
        self.lat: float = lat
        self.lon: float = lon
        # the altitude serves as an offset for the mesh in which it should be places later on,
        # e.g. for the plane the actual height or for the terrain the lowest point of the terrain
        # note: this height should be in meters
        self.altitude: float = altitude

        # WGS 84 / UTM zone xxx
        self.transformer: Transformer = transformer

    @abstractmethod
    def generate_mesh(self) -> Mesh:
        pass

    @abstractmethod
    def spawn(self, scene: Scene, geo: GeoTransformer) -> list[GameObject]:
        pass
