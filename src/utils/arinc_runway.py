from typing import override

import pygame
from pygame import Surface
from pyproj import Geod

from src.rendering.renderable import JsonSerializable, Renderable, GeoToScreenFunction
from src.utils.approch import Approach, FixIdentifier
from src.utils.open_street_map_runway import Coordinate

def magnetic_variation_to_float(var_str: str) -> float:
    """
    Convert a magnetic variation string like 'W0130' or 'E0450' to decimal degrees.
    W (west) -> negative, E (east) -> positive
    Last two digits are minutes.
    """
    if len(var_str) != 5:
        raise ValueError(f"Invalid variation string: {var_str}")

    direction = var_str[0].upper()
    degrees = int(var_str[1:4])  # 3 digits for degrees
    minutes = int(var_str[4:5]) * 10  # last digit is tenths of minutes

    decimal_degrees = degrees + minutes / 60.0

    if direction == "W":
        decimal_degrees *= -1
    elif direction != "E":
        raise ValueError(f"Unknown direction in variation string: {direction}")

    return decimal_degrees

geod = Geod(ellps="WGS84")

class ArincRunway(Renderable, JsonSerializable):
    def __init__(
        self,
        runway_identifier: str,
        latitude: float,
        longitude: float,
        length: float,
        bearing: float,
        width: float,
        stopway: float,
        displaced_threshold: float,
        approaches: list[Approach] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.runway_identifier: str = runway_identifier
        self.latitude: float = latitude
        self.longitude: float = longitude
        self.length: float = length
        self.bearing: float = bearing
        self.width: float = width
        self.stopway: float = stopway
        self.displaced_threshold: float = displaced_threshold
        self.approaches: list[Approach] = approaches if approaches is not None else []

        self.centerline_coordinates: tuple[Coordinate, Coordinate] = self._compute_centerline()
        self.polygon_coordinates: list[Coordinate] = self._compute_polygon()

    def _compute_centerline(self) -> tuple[Coordinate, Coordinate]:
        bearing = self.bearing / 10.0 + magnetic_variation_to_float("W0130")

        start_lat = self.latitude
        start_lon = self.longitude

        end_lon, end_lat, _ = geod.fwd(
            start_lon,
            start_lat,
            bearing,
            self.ft_to_m(self.length - self.displaced_threshold - self.stopway)
        )

        return (
            Coordinate(lat=start_lat, lon=start_lon),
            Coordinate(lat=end_lat, lon=end_lon),
        )

    def _compute_polygon(self) -> list[Coordinate]:
        start, end = self.centerline_coordinates

        half_width = self.ft_to_m(self.width) / 2

        az12, _, _ = geod.inv(start.lon, start.lat, end.lon, end.lat)

        left_az = az12 - 90
        right_az = az12 + 90

        left_start_lon, left_start_lat, _ = geod.fwd(start.lon, start.lat, left_az, half_width)
        right_start_lon, right_start_lat, _ = geod.fwd(start.lon, start.lat, right_az, half_width)

        left_end_lon, left_end_lat, _ = geod.fwd(end.lon, end.lat, left_az, half_width)
        right_end_lon, right_end_lat, _ = geod.fwd(end.lon, end.lat, right_az, half_width)

        return [
            Coordinate(lat=left_start_lat, lon=left_start_lon),
            Coordinate(lat=left_end_lat, lon=left_end_lon),
            Coordinate(lat=right_end_lat, lon=right_end_lon),
            Coordinate(lat=right_start_lat, lon=right_start_lon),
        ]


    @staticmethod
    def ft_to_m(value: float) -> float:
        return value * 0.3048

    @override
    def render(self, screen: Surface, geo_to_screen: GeoToScreenFunction, **kwargs) -> None:
        #if not self.polygon_coordinates:
        #    return

        #screen_points = [
        #    geo_to_screen(coord.lat, coord.lon)
        #    for coord in self.polygon_coordinates
        #]


        #pygame.draw.polygon(screen, (200, 200, 200), screen_points)

        start, end = self.centerline_coordinates
        #pygame.draw.line(
        #    screen,
        #    (255, 255, 255),
        #    geo_to_screen(start.lat, start.lon),
        #    geo_to_screen(end.lat, end.lon),
        #    2
        #)
        pygame.draw.circle(screen, (0, 0, 255), (geo_to_screen(start.lat, start.lon)), 5)
        pygame.draw.circle(screen, (255, 0, 0), (geo_to_screen(end.lat, end.lon)), 5)

        for approach in self.approaches:
            approach.render(screen, geo_to_screen)

    @override
    @classmethod
    def from_dict(cls: type["ArincRunway"], data: dict) -> "ArincRunway":
        approaches_data = data.get("approaches", [])
        approaches: list[Approach] = []
        for app in approaches_data:
            fixes_data = app.get("fixes", [])
            fixes: list[FixIdentifier] = [
                FixIdentifier(
                    fix_identifier=f.get("fix_identifier"),
                    path_terminator=f.get("path_terminator"),
                    latitude=f.get("latitude"),
                    longitude=f.get("longitude"),
                    altitude=f.get("altitude"),
                    sequence_number=f.get("sequence_number"),
                )
                for f in fixes_data
            ]
            approaches.append(
                Approach(
                    procedure_identifier=app.get("procedure_identifier"),
                    route_type=app.get("route_type"),
                    fixes=fixes,
                )
            )

        return cls(
            runway_identifier=data["runway_identifier"],
            latitude=data["latitude"],
            longitude=data["longitude"],
            length=data["length"],
            bearing=data["bearing"],
            width=data["width"],
            stopway=data.get("stopway", 0.0),
            displaced_threshold=data.get("displaced_threshold", 0.0),
            approaches=approaches,
        )

    @override
    def to_dict(self) -> dict:
        return {
            "runway_identifier": self.runway_identifier,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "length": self.length,
            "bearing": self.bearing,
            "width": self.width,
            "stopway": self.stopway,
            "displaced_threshold": self.displaced_threshold,
            "approaches": [
                {
                    "procedure_identifier": app.procedure_identifier,
                    "route_type": app.route_type,
                    "fixes": [
                        {
                            "fix_identifier": f.fix_identifier,
                            "path_terminator": f.path_terminator,
                            "latitude": f.latitude,
                            "longitude": f.longitude,
                            "altitude": f.altitude,
                            "sequence_number": f.sequence_number,
                        }
                        for f in app.fixes
                    ],
                }
                for app in self.approaches
            ],
        }