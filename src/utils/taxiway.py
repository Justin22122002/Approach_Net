from typing import override

from rendering.renderable import JsonSerializable
from utils.geo_types import Coordinate


class Taxiway(JsonSerializable):
    def __init__(
            self,
            open_street_map_id: str,
            coordinates: list[Coordinate],
            width: float | None = None,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.open_street_map_id: str = open_street_map_id
        self.coordinates: list[Coordinate] = coordinates
        self.width: float = float(width) if width else 1

    @override
    def to_dict(self) -> dict:
        return {
            "open_street_map_id": self.open_street_map_id,
            "width": self.width,
            "coordinates": [
                {"lat": c.lat, "lon": c.lon}
                for c in self.coordinates
            ],
        }

    @override
    @classmethod
    def from_dict(cls: type["Taxiway"], data: dict) -> "Taxiway":
        coordinates = [
            Coordinate(lat=c["lat"], lon=c["lon"])
            for c in data.get("coordinates", [])
        ]

        return cls(
            open_street_map_id=data["open_street_map_id"],
            coordinates=coordinates,
            width=data.get("width"),
        )
