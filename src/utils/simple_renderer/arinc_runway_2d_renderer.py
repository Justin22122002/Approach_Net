from typing import override

import pygame
from pygame import Surface

from rendering.renderable import Renderable, GeoToScreenFunction
from src.utils.arinc_runway import ArincRunway
from src.utils.simple_renderer.approach_2d_renderer import Approach2dRenderer


class ArincRunway2dRenderer(Renderable):
    def __init__(self, data: ArincRunway, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.runway: ArincRunway = data

    @override
    def render(self, screen: Surface, geo_to_screen: GeoToScreenFunction, **kwargs) -> None:
        start, end = self.runway.centerline_coordinates
        pygame.draw.line(
            screen,
            (255, 255, 255),
            geo_to_screen(start.lat, start.lon),
            geo_to_screen(end.lat, end.lon),
            2
        )
        pygame.draw.circle(screen, (0, 0, 255), (geo_to_screen(start.lat, start.lon)), 5)
        pygame.draw.circle(screen, (255, 0, 0), (geo_to_screen(end.lat, end.lon)), 5)

        for approach in self.runway.approaches:
            Approach2dRenderer(approach).render(screen, geo_to_screen)
