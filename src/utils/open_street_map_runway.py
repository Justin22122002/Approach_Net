from typing import override

from pyproj import Geod

from src.rendering.renderable import JsonSerializable
from src.utils.arinc_runway import Approach
from src.utils.geo_types import Coordinate

geod = Geod(ellps="WGS84")


class OpenStreetMapRunway(JsonSerializable):
    def __init__(
            self,
            open_street_map_id: str,
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
        self.open_street_map_id: str = open_street_map_id
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
    def to_dict(self) -> dict:
        return {
            "open_street_map_id": self.open_street_map_id,
            "ref": self.ref,
            "name": self.name,
            "width": self.width,
            "centerline_coordinates": [{"lat": c.lat, "lon": c.lon} for c in self.centerline_coordinates]
        }

    @override
    @classmethod
    def from_dict(cls: type["OpenStreetMapRunway"], data: dict) -> "OpenStreetMapRunway":
        centerline_coordinates = [Coordinate(lat=c["lat"], lon=c["lon"]) for c in data.get("centerline_coordinates", [])]

        return cls(
            open_street_map_id=data["open_street_map_id"],
            centerline_coordinates=centerline_coordinates,
            polygon_coordinates=[],
            approaches=[],
            ref=data.get("ref"),
            name=data.get("name"),
            width=data.get("width"),
        )
