from dataclasses import dataclass

import pandas as pd


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
