import json
from collections.abc import Callable
from pathlib import Path
from typing import override

import numpy as np
import pygame
from pygame import Surface

from src.rendering.renderable import Renderable, JsonSerializable, GeoToScreenFunction
from src.utils.arinc_runway import ArincRunway
from src.utils.open_street_map_runway import Coordinate, OpenStreetMapRunway, Taxiway


class Terrain(Renderable, JsonSerializable):
    def __init__(
        self,
        elevation_grid: np.ndarray,
        lat_min: float = 0.0,
        lon_min: float = 0.0,
        lat_max: float = 0.0,
        lon_max: float = 0.0,
    ) -> None:
        super().__init__()
        self.elevation_grid: np.ndarray = elevation_grid
        self.lat_min: float = lat_min
        self.lon_min: float = lon_min
        self.lat_max: float = lat_max
        self.lon_max: float = lon_max

    @override
    def render(self, screen: Surface, geo_to_screen: GeoToScreenFunction, **kwargs) -> None:
        if self.elevation_grid is None:
            return

        grid = self.elevation_grid

        # 1. normalize height values 0 - 255
        norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-6)
        img = (norm * 255).astype(np.uint8)

        # 2. grey values → RGB
        rgb = np.stack([img] * 3, axis=-1)

        # 3. numpy → pygame surface
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

        # 4. Geo → Screen Bounding Box
        x_min, y_min = geo_to_screen(self.lat_min, self.lon_min)
        x_max, y_max = geo_to_screen(self.lat_max, self.lon_max)

        top = min(y_min, y_max)
        bottom = max(y_min, y_max)
        left = min(x_min, x_max)
        right = max(x_min, x_max)

        width = int(right - left)
        height = int(bottom - top)

        if width <= 0 or height <= 0:
            return

        surface = pygame.transform.scale(surface, (width, height))

        screen.blit(surface, (left, top))

    @override
    def to_dict(self) -> dict:
        return {
            "lat_min": self.lat_min,
            "lon_min": self.lon_min,
            "lat_max": self.lat_max,
            "lon_max": self.lon_max,
            "elevation_grid": self.elevation_grid.tolist() if self.elevation_grid is not None else None
        }

    @override
    @classmethod
    def from_dict(cls: type["Terrain"], data: dict) -> "Terrain":
        grid = data.get("elevation_grid")

        return cls(
            elevation_grid=np.array(grid) if grid else None,
            lat_min=data.get("lat_min", 0),
            lon_min=data.get("lon_min", 0),
            lat_max=data.get("lat_max", 0),
            lon_max=data.get("lon_max", 0),
        )


class AirportTerminal(Renderable, JsonSerializable):
    def __init__(
            self,
            open_street_map_id: str,
            coordinates: list[Coordinate],
            name: str | None = None,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.open_street_map_id: str = open_street_map_id
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

        pygame.draw.polygon(screen, (114, 60, 28), points, 0)

    @override
    def to_dict(self) -> dict:
        return {
            "open_street_map_id": self.open_street_map_id,
            "name": self.name,
            "coordinates": [
                {"lat": c.lat, "lon": c.lon}
                for c in self.coordinates
            ],
        }

    @override
    @classmethod
    def from_dict(cls: type["AirportTerminal"], data: dict) -> "AirportTerminal":
        coordinates = [
            Coordinate(lat=c["lat"], lon=c["lon"])
            for c in data.get("coordinates", [])
        ]

        return cls(
            open_street_map_id=data["open_street_map_id"],
            coordinates=coordinates,
            name=data.get("name"),
        )

class Airport(Renderable, JsonSerializable):
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

        self.terrain.render(screen, geo_to_screen)

        for rwy in self.open_street_map_runway:
            rwy.render(screen, geo_to_screen)

        for rwy in self.arinc_424_runway:
            rwy.render(screen, geo_to_screen)

        for twy in self.taxiways:
            twy.render(screen, geo_to_screen)

        for term in self.terminals:
            term.render(screen, geo_to_screen)

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
