import json
from collections.abc import Callable
from pathlib import Path
from typing import override

import pygame
from pygame import Surface

from src.rendering.renderable import Renderable
from src.utils.runway import Coordinate, Runway, Taxiway


class AirportTerminal(Renderable):
    def __init__(
            self,
            id: str,
            coordinates: list[Coordinate],
            name: str | None = None,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.id: str = id
        self.coordinates: list[Coordinate] = coordinates
        self.name: str | None = name

    @override
    def render(
            self,
            screen: Surface,
            geo_to_screen: Callable[[float, float], tuple[float, float]],
            **kwargs
    ) -> None:
        if len(self.coordinates) < 3:
            return
        points = [geo_to_screen(c.lat, c.lon) for c in self.coordinates]

        pygame.draw.polygon(screen, (200, 150, 100), points, 0)

class Airport(Renderable):
    def __init__(
            self,
            icao: str,
            lat: float,
            lon: float,
            runways: list[Runway],
            taxiways: list[Taxiway],
            terminals: list[AirportTerminal],
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.icao: str = icao
        self.lat: float = lat
        self.lon: float = lon
        self.runways: list[Runway] = runways
        self.taxiways: list[Taxiway] = taxiways
        self.terminals : list[AirportTerminal]= terminals

    @override
    def render(
            self,
            screen: Surface,
            geo_to_screen: Callable[[float, float], tuple[float, float]],
            **kwargs,
    ) -> None:
        cx, cy = geo_to_screen(self.lat, self.lon)
        # middle point for quering the airport
        pygame.draw.circle(
            surface=screen,
            color=(255, 0, 0),
            center=(cx, cy),
            radius=1,
        )

        for rwy in self.runways:
            rwy.render(screen, geo_to_screen)

        for twy in self.taxiways:
            twy.render(screen, geo_to_screen)

        for term in self.terminals:
            term.render(screen, geo_to_screen)

    def save_runways_to_json(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)

        runways_data = [rwy.to_dict() for rwy in self.runways]

        filepath.write_text(
            json.dumps(runways_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def load_runways_from_json(self, filepath: Path) -> None:
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist.")

        data = json.loads(filepath.read_text(encoding="utf-8"))
        self.runways = [Runway.from_dict(d) for d in data]


