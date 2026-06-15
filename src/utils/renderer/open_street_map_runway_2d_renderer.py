from typing import override, Callable

import pygame
from pygame import Surface

from rendering.renderable import Renderable
from src.utils.open_street_map_runway import OpenStreetMapRunway


class OpenStreetMapRunway2dRenderer(Renderable):
    def __init__(self, data: OpenStreetMapRunway, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.runway: OpenStreetMapRunway = data

    @override
    def render(
            self,
            screen: Surface,
            geo_to_screen: Callable[[float, float], tuple[float, float]],
            **kwargs,
    ) -> None:
        if len(self.runway.polygon_coordinates) < 4 or self.runway.width is None:
            return

        points = [geo_to_screen(c.lat, c.lon) for c in self.runway.polygon_coordinates]

        pygame.draw.polygon(
            surface=screen,
            color=(105, 105, 105),
            points=points,
            width=0,
        )

        points = [geo_to_screen(c.lat, c.lon) for c in self.runway.centerline_coordinates]
        if len(points) > 1:
            pygame.draw.lines(
                surface=screen,
                color=(255, 0, 0),
                closed=False,
                points=points,
                width=1,
            )

        if self.runway.ref:
            font = pygame.font.SysFont("Arial", 14)
            start = self.runway.polygon_coordinates[0]
            end = self.runway.polygon_coordinates[-2]

            sx, sy = geo_to_screen(start.lat, start.lon)
            ex, ey = geo_to_screen(end.lat, end.lon)

            txt_start = font.render(self.runway.ref, True, (255, 255, 0))
            txt_end = font.render(self.runway.ref, True, (255, 255, 0))

            offset = 5

            screen.blit(txt_start, (sx + offset, sy + offset))
            screen.blit(txt_end, (ex + offset, ey + offset))
