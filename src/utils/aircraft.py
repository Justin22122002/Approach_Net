from dataclasses import dataclass

import pandas as pd
from pandas import Series


@dataclass
class Aircraft:
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

@dataclass
class AircraftType:
    type: str | None = None
    icao_type: str | None = None
    manufacturer: str | None = None
    mode_s: str | None = None
    registration: str | None = None
    registered_owner: str | None = None
    url_photo_thumbnail: str | None = None

    @classmethod
    def from_row(cls, row: Series) -> "AircraftType":
        return cls(
            type=row.get("type"),
            icao_type=row.get("icao_type"),
            manufacturer=row.get("manufacturer"),
            mode_s=row.get("mode_s"),
            registration=row.get("registration"),
            registered_owner=row.get("registered_owner"),
            url_photo_thumbnail=row.get("url_photo_thumbnail"),
        )
