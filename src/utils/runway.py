from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import pygame
from pygame import Surface
from pyproj import Geod

from src.rendering.renderable import JsonSerializable, Renderable
from src.utils.approach import Approach

geod = Geod(ellps="WGS84")

@dataclass
class Coordinate:
    lat: float
    lon: float


class Runway(Renderable, JsonSerializable):
    def __init__(
            self,
            id: str,
            centerline_coordinates: list[Coordinate],
            approaches: list[Approach],
            polygon_coordinates: list[Coordinate],
            ref: str | None = None,
            name: str | None = None,
            width: float | None = None,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.id: str = id
        self.centerline_coordinates: list[Coordinate] = centerline_coordinates
        self.ref: str | None = ref
        self.name: str | None = name
        self.width: float | None = float(width) if width else None
        self.approaches: list[Approach] = approaches
        self.polygon_coordinates: list[Coordinate]
        if len(polygon_coordinates) == 0:
            self.polygon_coordinates = self._compute_edges()
        else:
            self.polygon_coordinates = polygon_coordinates

    def _compute_edges(self) -> list[Coordinate]:
        if len(self.centerline_coordinates) < 2 or self.width is None:
            return []

        half_width_m = self.width / 2

        start = self.centerline_coordinates[0]
        end = self.centerline_coordinates[-1]

        az12, az21, dist = geod.inv(start.lon, start.lat, end.lon, end.lat)

        left_az = az12 - 90
        right_az = az12 + 90

        left_start_lon, left_start_lat, _ = geod.fwd(start.lon, start.lat, left_az, half_width_m)
        right_start_lon, right_start_lat, _ = geod.fwd(start.lon, start.lat, right_az, half_width_m)
        left_end_lon, left_end_lat, _ = geod.fwd(end.lon, end.lat, left_az, half_width_m)
        right_end_lon, right_end_lat, _ = geod.fwd(end.lon, end.lat, right_az, half_width_m)

        polygon_coords = [
            Coordinate(lat=left_start_lat, lon=left_start_lon),
            Coordinate(lat=left_end_lat, lon=left_end_lon),
            Coordinate(lat=right_end_lat, lon=right_end_lon),
            Coordinate(lat=right_start_lat, lon=right_start_lon),
        ]

        return polygon_coords

    @override
    def render(
            self,
            screen: Surface,
            geo_to_screen: Callable[[float, float], tuple[float, float]],
            **kwargs,
    ) -> None:
        if len(self.polygon_coordinates) < 4 or self.width is None:
            return

        points = [geo_to_screen(c.lat, c.lon) for c in self.polygon_coordinates]

        pygame.draw.polygon(
            surface=screen,
            color=(220, 220, 220),
            points=points,
            width=0,
        )

        points = [geo_to_screen(c.lat, c.lon) for c in self.centerline_coordinates]
        if len(points) > 1:
            pygame.draw.lines(
                surface=screen,
                color=(255, 0, 0),
                closed=False,
                points=points,
                width=1,
            )

        if self.ref:
            font = pygame.font.SysFont("Arial", 14)
            start = self.polygon_coordinates[0]
            end = self.polygon_coordinates[-2]

            sx, sy = geo_to_screen(start.lat, start.lon)
            ex, ey = geo_to_screen(end.lat, end.lon)

            txt_start = font.render(self.ref, True, (255, 255, 0))
            txt_end = font.render(self.ref, True, (255, 255, 0))

            offset = 5

            screen.blit(txt_start, (sx + offset, sy + offset))
            screen.blit(txt_end, (ex + offset, ey + offset))

    @override
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ref": self.ref,
            "name": self.name,
            "width": self.width,
            "polygon_coordinates": [{"lat": c.lat, "lon": c.lon} for c in self.polygon_coordinates]
        }

    @override
    @classmethod
    def from_dict(cls, data: dict) -> "Runway":
        polygon_coordinates = [Coordinate(lat=c["lat"], lon=c["lon"]) for c in data.get("polygon_coordinates", [])]

        return cls(
            id=data["id"],
            centerline_coordinates=[],
            polygon_coordinates=polygon_coordinates,
            approaches=[],
            ref=data.get("ref"),
            name=data.get("name"),
            width=data.get("width"),
        )

class Taxiway(Renderable):
    def __init__(
            self,
            id: str,
            coordinates: list[Coordinate],
            width: float | None = None,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.id: str = id
        self.coordinates: list[Coordinate] = coordinates
        self.width: float = float(width) if width else 1

    @override
    def render(
            self,
            screen: Surface,
            geo_to_screen: Callable[[float, float], tuple[float, float]],
            **kwargs,
    ) -> None:
        if len(self.coordinates) < 2:
            return
        points = [geo_to_screen(c.lat, c.lon) for c in self.coordinates]
        pygame.draw.lines(
            surface=screen,
            color=(120, 120, 120),
            closed=False,
            points=points,
            width=1,
        )

