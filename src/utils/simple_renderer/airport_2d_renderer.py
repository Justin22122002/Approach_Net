from typing import override, Callable

import pygame
from pygame import Surface

from rendering.renderable import Renderable
from src.utils.airport import Airport
from src.utils.simple_renderer.airport_terminal_2d_renderer import AirportTerminal2dRenderer
from src.utils.simple_renderer.arinc_runway_2d_renderer import ArincRunway2dRenderer
from src.utils.simple_renderer.open_street_map_runway_2d_renderer import OpenStreetMapRunway2dRenderer
from src.utils.simple_renderer.taxiway_2d_renderer import Taxiway2dRenderer
from src.utils.simple_renderer.terrain_2d_renderer import Terrain2dRenderer


class Airport2dRenderer(Renderable):
    def __init__(self, airport: Airport, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.airport: Airport = airport

    @override
    def render(
            self,
            screen: Surface,
            geo_to_screen: Callable[[float, float], tuple[float, float]],
            **kwargs,
    ) -> None:
        cx, cy = geo_to_screen(self.airport.lat, self.airport.lon)
        # middle point for quering the airport
        pygame.draw.circle(
            surface=screen,
            color=(255, 0, 0),
            center=(cx, cy),
            radius=1,
        )

        Terrain2dRenderer(self.airport.terrain).render(screen, geo_to_screen)

        for rwy in self.airport.open_street_map_runway:
            OpenStreetMapRunway2dRenderer(rwy).render(screen, geo_to_screen)

        for rwy in self.airport.arinc_424_runway:
            ArincRunway2dRenderer(rwy).render(screen, geo_to_screen)

        for twy in self.airport.taxiways:
            Taxiway2dRenderer(twy).render(screen, geo_to_screen)

        for term in self.airport.terminals:
            AirportTerminal2dRenderer(term).render(screen, geo_to_screen)
