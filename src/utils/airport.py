import json
from pathlib import Path
from typing import override

import numpy as np

from rendering.renderable import JsonSerializable
from utils.arinc_runway import ArincRunway
from utils.open_street_map_runway import OpenStreetMapRunway
from utils.taxiway import Taxiway
from utils.airport_terminal import AirportTerminal
from utils.terrain import Terrain


class Airport(JsonSerializable):
    def __init__(
            self,
            icao: str,
            lat: float,
            lon: float,
            open_street_map_runway: list[OpenStreetMapRunway],
            arinc_424_runway: list[ArincRunway],
            taxiways: list[Taxiway],
            terminals: list[AirportTerminal],
            terrain: Terrain,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.icao: str = icao
        self.lat: float = lat
        self.lon: float = lon
        self.open_street_map_runway: list[OpenStreetMapRunway] = open_street_map_runway
        self.arinc_424_runway: list[ArincRunway] = arinc_424_runway
        self.taxiways: list[Taxiway] = taxiways
        self.terminals : list[AirportTerminal] = terminals
        self.terrain: Terrain = terrain

        # this is used for the JSON file path naming for consistent saving and loading
        self.file_structure: dict[type[JsonSerializable], str] = {
            OpenStreetMapRunway: "open_street_map_runway",
            ArincRunway: "arinc_424_runway",
            Taxiway: "taxiways",
            AirportTerminal: "terminals",
            Terrain: "terrain",
        }

    def save_to_folder(self, base_path: Path) -> None:
        airport_path = base_path / self.icao
        airport_path.mkdir(parents=True, exist_ok=True)

        (airport_path / "airport.json").write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8"
        )

        for cls_type, folder_name in self.file_structure.items():
            items = getattr(self, folder_name)

            if isinstance(items, list):
                data = [item.to_dict() for item in items]
            else:
                data = items.to_dict()

            (airport_path / f"{folder_name}.json").write_text(
                json.dumps(data, indent=2),
                encoding="utf-8"
            )

    @classmethod
    def load_from_folder(cls: type["Airport"], base_path: Path, icao: str) -> "Airport":
        airport_path = base_path / icao

        if not airport_path.exists():
            raise FileNotFoundError(f"{airport_path} not found")

        airport_data = json.loads(
            (airport_path / "airport.json").read_text(encoding="utf-8")
        )

        airport = cls.from_dict(airport_data)

        for cls_type, folder_name in airport.file_structure.items():
            file_path = airport_path / f"{folder_name}.json"

            if not file_path.exists():
                continue

            data = json.loads(file_path.read_text(encoding="utf-8"))

            if isinstance(data, list):
                items = [cls_type.from_dict(d) for d in data]
                setattr(airport, folder_name, items)
            else:
                item = cls_type.from_dict(data)
                setattr(airport, folder_name, item)

        return airport

    @override
    def to_dict(self) -> dict:
        return {
            "icao": self.icao,
            "lat": self.lat,
            "lon": self.lon,
        }

    @override
    @classmethod
    def from_dict(cls: type["Airport"], data: dict) -> "Airport":
        return cls(
            icao=data["icao"],
            lat=data["lat"],
            lon=data["lon"],
            open_street_map_runway=[],
            arinc_424_runway=[],
            taxiways=[],
            terminals=[],
            terrain=Terrain(
                elevation_grid=np.array([]),
                lat_min=0.0,
                lon_min=0.0,
                lat_max=0.0,
                lon_max=0.0,
            ),
        )
