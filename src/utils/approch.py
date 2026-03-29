from dataclasses import dataclass
from typing import override

import pygame
from pygame import Surface

from src.rendering.renderable import GeoToScreenFunction, Renderable


@dataclass
class FixIdentifier:
    fix_identifier: str
    path_terminator: str # e.g -> IF, TF etc
    latitude: float
    longitude: float
    altitude: float # when receiving from arinc 424 format, this is the height above the runway in feet
    sequence_number: str


@dataclass
class Approach(Renderable):
    procedure_identifier: str
    route_type: str  # R, A, I
    fixes: list[FixIdentifier]

    @override
    def render(self, screen: Surface, geo_to_screen: GeoToScreenFunction, **kwargs) -> None:
        color = (255, 0, 0) if self.route_type == "R" else (0, 255, 0) if self.route_type == "A" else (0, 0, 255)

        sorted_fixes = sorted(
            self.fixes,
            key=lambda f: float(f.sequence_number) if isinstance(f.sequence_number, (int, float, str)) and str(f.sequence_number).replace('.', '', 1).isdigit() else 0.0
        )

        #for i in range(len(sorted_fixes) - 1):
        #    f1 = sorted_fixes[i]
        #    f2 = sorted_fixes[i + 1]

        #    if f1.latitude is not None and f1.longitude is not None and f2.latitude is not None and f2.longitude is not None:
        #        pygame.draw.line(
        #            screen,
        #            color,
        #            geo_to_screen(f1.latitude, f1.longitude),
        #            geo_to_screen(f2.latitude, f2.longitude),
        #            2
        #        )

        font = pygame.font.SysFont("Arial", 12)
        for f in sorted_fixes:
            if f.latitude is not None and f.longitude is not None:
                x, y = geo_to_screen(f.latitude, f.longitude)
                pygame.draw.circle(screen, color, (x, y), 3)

                text_surf = font.render(f.fix_identifier, True, color)
                screen.blit(text_surf, (x + 5, y - 5))