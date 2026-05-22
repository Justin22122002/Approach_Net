from abc import ABCMeta, abstractmethod

from utils.mesh import Mesh

class ProjectMesh(metaclass=ABCMeta):
    """
    Georeferenced mesh builder.

    Holds:
    - geographic anchor (lat/lon)
    - scaling policy
    - mesh generation logic
    """

    def __init__(
            self,
            horizontal_scale: float,
            vertical_scale: float,
            lat: float,
            lon: float,
            altitude: float,
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
        self.altitude: float = altitude

    @abstractmethod
    def generate_mesh(self) -> Mesh:
        pass



def utm_epsg(lat: float, lon: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"