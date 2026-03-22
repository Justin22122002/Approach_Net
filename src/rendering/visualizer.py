import pygame
from pyproj import Transformer

from src.rendering.renderable import Renderable

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

class Visualizer:
    def __init__(
            self,
            width: float = 1080,
            height: float = 1080,
    ) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("OpenSky Radar")
        self.width: float  = width
        self.height : float = height
        self.zoom: float = 1.0
        self.offset_x: int = 0
        self.offset_y: int = 0

    def geo_to_screen(
            self,
            lat: float,
            lon: float,
            lamin: float,
            lamax: float,
            lomin: float,
            lomax: float,
    ) -> tuple[int, int]:
        x = (lon - lomin) / (lomax - lomin) * self.width * self.zoom + self.offset_x
        y = self.height * self.zoom - ((lat - lamin) / (lamax - lamin) * self.height * self.zoom) + self.offset_y
        return int(x), int(y)

    def geo_to_screen_mercator(
            self,
            lat: float,
            lon: float,
            lamin: float,
            lamax: float,
            lomin: float,
            lomax: float,
    ) -> tuple[int, int]:
        x, y = transformer.transform(lon, lat)
        xmin, ymin = transformer.transform(lomin, lamin)
        xmax, ymax = transformer.transform(lomax, lamax)

        scale = min(self.width / (xmax - xmin), self.height / (ymax - ymin))
        scale *= self.zoom

        x_screen = (x - xmin) * scale + self.offset_x
        y_screen = - ((y - ymin) * scale) + self.offset_y

        return int(x_screen), int(y_screen)

    def render_objects(
            self,
            objects: list[Renderable],
            lamin: float,
            lamax: float,
            lomin: float,
            lomax: float,
    ) -> None:
        self.screen.fill((10, 10, 20))

        geo_fn = lambda lat, lon: self.geo_to_screen_mercator(
            lat, lon, lamin, lamax, lomin, lomax
        )

        group = pygame.sprite.Group(objects)

        group.update(geo_fn)


        for obj in objects:
            obj.render(
                self.screen,
                geo_fn,
            )
        pygame.display.flip()
