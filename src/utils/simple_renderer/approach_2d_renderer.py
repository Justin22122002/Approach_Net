from typing import override

import pygame
from pygame import Surface

from rendering.renderable import Renderable, GeoToScreenFunction
from utils.approch import Approach


class Approach2dRenderer(Renderable):
    def __init__(self, data: Approach, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.approach: Approach = data

    @override
    def render(self, screen: Surface, geo_to_screen: GeoToScreenFunction, **kwargs) -> None:
        color = (255, 0, 0) if self.approach.route_type == "R" else (0, 255, 0) if self.approach.route_type == "A" else (0, 0, 255)

        sorted_fixes = sorted(
            self.approach.fixes,
            key=lambda f: float(f.sequence_number) if isinstance(f.sequence_number, (int, float, str)) and str(f.sequence_number).replace('.', '', 1).isdigit() else 0.0
        )

        font = pygame.font.SysFont("Arial", 12)
        for f in sorted_fixes:
            if f.latitude is not None and f.longitude is not None:
                x, y = geo_to_screen(f.latitude, f.longitude)
                pygame.draw.circle(screen, color, (x, y), 3)

                text_surf = font.render(f.fix_identifier, True, color)
                screen.blit(text_surf, (x + 5, y - 5))
