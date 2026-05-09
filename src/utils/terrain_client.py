from typing import Any

import requests
import numpy as np
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt


class CopernicusDEMProvider:
    """
    The Copernicus DEM instances are available in Geographic Coordinates (DGED and DTED format) and
    the corresponding horizontal reference datum is the World Geodetic System 1984 (WGS84-G1150;
    EPSG 4326). In addition, the Copernicus DEM is available in INSPIRE format with projected coordinates
    (Lambert Azimuth Equal Area for continental Europe) with horizontal reference datum ETRS89-GRS80;
    Universal Transverse Mercator for French Overseas Territories (DOMs) with horizontal reference Datum
    WGS84-G1150).
    The vertical reference datum is the Earth Gravitational Model 20081 (EGM2008; EPSG 3855).

    [https://dataspace.copernicus.eu/sites/default/files/media/files/2024-06/geo1988-copernicusdem-spe-002_producthandbook_i5.0.pdf] -> 1.2.1
    """
    TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
    METER_TO_FEET = 3.28084

    def __init__(self, username: str, password: str) -> None:
        self.username: str = username
        self.password: str = password
        self.access_token: str = self._authenticate()

    def _authenticate(self) -> str:
        data: dict[str, str] = {
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password,
            "grant_type": "password"
        }

        r: requests.Response = requests.post(self.TOKEN_URL, data=data)
        r.raise_for_status()

        return r.json()["access_token"]

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

    def get_terrain(
            self,
            lat_min: float,
            lon_min: float,
            lat_max: float,
            lon_max: float,
            width: int = 256,
            height: int = 256,
    ) -> np.ndarray:
        payload: dict[str, Any] = {
            "input": {
                "bounds": {
                    "bbox": [lon_min, lat_min, lon_max, lat_max]
                },
                "data": [
                    {
                        "type": "dem"
                    }
                ]
            },
            "output": {
                "width": width,
                "height": height,
                "responses": [
                    {
                        "identifier": "default",
                        "format": {"type": "image/tiff"}
                    }
                ]
            },
            "evalscript": """
                //VERSION=3
                function setup() {
                    return {
                        input: ["DEM"],
                        output: {
                            bands: 1,
                            sampleType: "FLOAT32"
                        }
                    };
                }

                function evaluatePixel(sample) {
                    return [sample.DEM];
                }
            """
        }

        r: requests.Response = requests.post(self.PROCESS_URL, json=payload, headers=self._headers())

        if r.status_code == 401:
            self.access_token = self._authenticate()
            r = requests.post(self.PROCESS_URL, json=payload, headers=self._headers())

        r.raise_for_status()

        with MemoryFile(r.content) as memfile:
            with memfile.open() as dataset:
                data: np.ndarray = dataset.read(1)

        return data

    def get_height(self, lat: float, lon: float) -> float:
        delta: float = 0.0001

        data: np.ndarray = self.get_terrain(
            lat - delta,
            lon - delta,
            lat + delta,
            lon + delta
        )

        return float(np.mean(data))

    @staticmethod
    def plot_terrain(grid: np.ndarray) -> None:
        plt.figure()
        plt.imshow(grid)
        plt.colorbar(label="Elevation (m)")
        plt.title("Terrain Elevation")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

if __name__ == "__main__":
    password = "Copernicus2026!"
    email = "justin.groh@study.thws.de"

    terrain = CopernicusDEMProvider(email, password)

    grid_terrain = terrain.get_terrain(
        lat_min=50.0,
        lon_min=8.5,
        lat_max=50.1,
        lon_max=8.6
    )

    terrain.plot_terrain(grid_terrain)

    print("Grid shape:", grid_terrain.shape)

    min_h_m = float(np.nanmin(grid_terrain))
    max_h_m = float(np.nanmax(grid_terrain))

    min_h_ft = min_h_m * 3.28084
    max_h_ft = max_h_m * 3.28084

    print(f"Min height: {min_h_m:.2f} m / {min_h_ft:.2f} ft")
    print(f"Max height: {max_h_m:.2f} m / {max_h_ft:.2f} ft")
