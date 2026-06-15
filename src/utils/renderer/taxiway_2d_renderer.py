from typing import override, Callable

import pygame
from pygame import Surface

from rendering.renderable import Renderable
from src.utils.taxiway import Taxiway


class Taxiway2dRenderer(Renderable):
    def __init__(self, data: Taxiway, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.taxiway: Taxiway = data

    @override
    def render(
            self,
            screen: Surface,
            geo_to_screen: Callable[[float, float], tuple[float, float]],
            **kwargs,
    ) -> None:
        if len(self.taxiway.coordinates) < 2:
            return
        points = [geo_to_screen(c.lat, c.lon) for c in self.taxiway.coordinates]
        pygame.draw.lines(
            surface=screen,
            color=(230, 232, 49),
            closed=False,
            points=points,
            width=1,
        )
