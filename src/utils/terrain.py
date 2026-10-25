from typing import override

import numpy as np

from rendering.renderable import JsonSerializable


class Terrain(JsonSerializable):
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
