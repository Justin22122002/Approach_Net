from typing import override, Callable

import pygame
from pygame import Surface

from rendering.renderable import Renderable
from src.utils.airport_terminal import AirportTerminal


class AirportTerminal2dRenderer(Renderable):
    def __init__(self, data: AirportTerminal, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terminal: AirportTerminal = data

    @override
    def render(
            self,
            screen: Surface,
            geo_to_screen: Callable[[float, float], tuple[float, float]],
            **kwargs
    ) -> None:
        if len(self.terminal.coordinates) < 3:
            return
        points = [geo_to_screen(c.lat, c.lon) for c in self.terminal.coordinates]

        pygame.draw.polygon(screen, (114, 60, 28), points, 0)
