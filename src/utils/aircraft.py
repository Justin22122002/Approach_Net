from dataclasses import dataclass
from typing import override

import pandas as pd
import pygame
from pygame import Surface

from src.rendering.renderable import GeoToScreenFunction, Renderable


@dataclass
class AircraftData:
    icao24: str | None
    callsign: str | None
    origin_country: str | None
    last_position: pd.Timestamp | None
    timestamp: pd.Timestamp | None
    longitude: float | None
    latitude: float | None
    altitude: float | None
    onground: bool | None
    groundspeed: float | None
    track: float | None
    vertical_rate: float | None
    sensors: list[int] | None # most of the time this is None
    geoaltitude: float | None
    squawk: str | None
    spi: bool | None
    position_source: int | None
    category: int | None


class Aircraft(Renderable):

    def __init__(self, data: AircraftData, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data: AircraftData = data

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
            if self.data.groundspeed is None or self.data.groundspeed == 0:
                color = (128, 128, 128)
            else:
                color = (255, 165, 0)

        pygame.draw.circle(screen, color, (int(x), int(y)), 4)

        if self.data.callsign:
            # todo: move in separate file
            font = pygame.font.SysFont("Arial", 14)
            text = font.render(self.data.callsign.strip(), True, (255, 255, 255))
            screen.blit(text, (x + 5, y - 5))