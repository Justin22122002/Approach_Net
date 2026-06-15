from typing import override

import pygame
from pygame import Surface

from rendering.renderable import Renderable, GeoToScreenFunction
from utils.aircraft import Aircraft


class Aircraft2dRenderer(Renderable):
    def __init__(self, data: Aircraft, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data: Aircraft = data

    @override
    def render(
            self,
            screen: Surface,
            geo_to_screen: GeoToScreenFunction,
            **kwargs
    ) -> None:
        if self.data.latitude is None or self.data.longitude is None:
            return

        x, y = geo_to_screen(self.data.latitude, self.data.longitude)

        color = (0, 255, 0)
        if self.data.onground:
            if self.data.groundspeed is None or self.data.groundspeed == "NA" or self.data.groundspeed == 0:
                color = (128, 128, 128)
            else:
                color = (255, 165, 0)

        pygame.draw.circle(screen, color, (int(x), int(y)), 4)

        if self.data.callsign:
            # todo: move in separate file
            font = pygame.font.SysFont("Arial", 14)

            text = font.render(self.data.callsign.strip(), True, (255, 255, 255))
            screen.blit(text, (x + 5, y - 5))
