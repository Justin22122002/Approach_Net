from collections.abc import Callable
from pathlib import Path
from pyproj import Geod
from typing import Any

import numpy as np
import requests

from src.utils.airac_net_client import AiracNetClient
from src.utils.airport import AirportTerminal, Airport
from utils.terrain import Terrain
from src.utils.arinc_runway import ArincRunway
from src.utils.open_street_map_runway import Coordinate, OpenStreetMapRunway
from utils.taxiway import Taxiway
from src.utils.sqlite_airinc24_database_client import SQLiteArInC24DatabaseClient
from src.utils.terrain_client import CopernicusDEMProvider

# todo: move both methods into helper files
def only_r_approaches(r: list[ArincRunway]) -> list[ArincRunway]:
    for runway in r:
        runway.approaches = [
            app for app in runway.approaches
            if app.route_type == "R"
        ]
    return r

def get_bbox_simple(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    geod = Geod(ellps="WGS84")
    radius_m = radius_km * 1000.0

    _, lat_max, _ = geod.fwd(lon, lat, az=0,   dist=radius_m)
    _, lat_min, _ = geod.fwd(lon, lat, az=180, dist=radius_m)
    lon_max, _, _ = geod.fwd(lon, lat, az=90,  dist=radius_m)
    lon_min, _, _ = geod.fwd(lon, lat, az=270, dist=radius_m)

    return lat_min, lon_min, lat_max, lon_max

class OpenAirportMapClient:
    BASE_URL = "https://openairportmap.org/api/airport"

    def __init__(self, arinc_db_path: Path = Path("FAACIFP18")) -> None:
        self.data: Any | None = None
        self.arinc_client: SQLiteArInC24DatabaseClient = SQLiteArInC24DatabaseClient(arinc_db_path)
        self.terrain_client: CopernicusDEMProvider = CopernicusDEMProvider("justin.groh@study.thws.de", "Copernicus2026!")

    def get_airport(self, icao: str) -> Any:
        r = requests.get(f"{self.BASE_URL}/{icao}")
        r.raise_for_status()
        self.data = r.json()
        return self.data

    def load_new_airport(
            self,
            icao: str,
            lat: float = 0.0,
            lon: float = 0.0,
            runway_approach_filter: Callable[[list[ArincRunway]], list[ArincRunway]] = lambda r: r,
            radius_km: float = 10.0,
            terrain_width: int = 1024,
            terrain_height: int = 1024,
    ) -> Airport:
        self.get_airport(icao)

        arinc_runways: list[ArincRunway] = self.arinc_client.get_all_runways_with_approaches_for_airport(icao)
        filtered_arinc_runways: list[ArincRunway] = runway_approach_filter(arinc_runways)

        if len(arinc_runways) == 0:
            print("Get Runways from AirAC.net")
            client: AiracNetClient = AiracNetClient()
            runways: list[ArincRunway] = client.get_airport_full(icao)
            filtered_arinc_runways = runway_approach_filter(runways)

        # todo: provide new option to inject this from the outside
        lat_min, lon_min, lat_max, lon_max = get_bbox_simple(
            lat, lon, radius_km
        )

        grid: np.ndarray = self.terrain_client.get_terrain(
            lat_min=lat_min,
            lon_min=lon_min,
            lat_max=lat_max,
            lon_max=lon_max,
            width=terrain_width,
            height=terrain_height,
        )

        return Airport(
            icao=icao,
            lat=lat,
            lon=lon,
            open_street_map_runway=self.get_runways(),
            arinc_424_runway=filtered_arinc_runways,
            taxiways=self.get_taxiways(),
            terminals=self.get_terminals(),
            terrain=Terrain(
                elevation_grid=grid,
                lat_min=lat_min,
                lon_min=lon_min,
                lat_max=lat_max,
                lon_max=lon_max,
            ),
        )

    # todo: filter out double runways
    def get_runways(self) -> list[OpenStreetMapRunway]:
        runways = []
        for el in self.data.get("elements", []):
            tags = el.get("tags", {})
            if tags.get("aeroway") != "runway":
                continue

            coords = self._extract_coords(el)
            if not coords:
                continue

            ref = tags.get("ref") or tags.get("name") or f"unknown_{el['id']}"
            name = tags.get("name")
            width = tags.get("width")
            length = tags.get("length")
            surface = tags.get("surface")

            runway = OpenStreetMapRunway(
                open_street_map_id=str(el["id"]),
                centerline_coordinates=coords,
                polygon_coordinates=[],
                ref=ref,
                name=name,
                width=width,
                approaches=[],
            )

            runway.length = length
            runway.surface = surface

            runways.append(runway)

        return runways

    def get_taxiways(self) -> list[Taxiway]:
        taxiways = []
        for el in self.data.get("elements", []):
            tags = el.get("tags", {})
            if tags.get("aeroway") != "taxiway":
                continue

            coords: list[Coordinate] = self._extract_coords(el)
            taxiways.append(Taxiway(
                open_street_map_id=str(el["id"]),
                coordinates=coords,
                width=tags.get("width"),
            ))

        return taxiways


    def get_terminals(self) -> list[AirportTerminal]:
        terminals = []
        for el in self.data.get("elements", []):
            tags = el.get("tags", {})
            if (tags.get("aeroway") != "terminal"
                    and "terminal" not in tags.get("name", "") # removed .tolower()
                    and tags.get("aeroway") != "construction"
            ):
                continue

            coords = self._extract_coords(el)

            terminals.append(AirportTerminal(
                open_street_map_id=str(el["id"]),
                coordinates=coords,
                name=tags.get("name")
            ))

        return terminals

    def _extract_coords(self, el: dict) -> list[Coordinate]:
        coords: list[Coordinate] = []

        if "lat" in el and "lon" in el:
            coords.append(Coordinate(lat=el["lat"], lon=el["lon"]))

        if "nodes" in el:
            node_map = {e["id"]: e for e in self.data.get("elements", []) if e["type"] == "node"}
            for node_id in el["nodes"]:
                node = node_map.get(node_id)
                if node:
                    coords.append(Coordinate(lat=node["lat"], lon=node["lon"]))

        elif el.get("type") == "relation":

            way_map = {e["id"]: e for e in self.data.get("elements", []) if e["type"] == "way"}

            for member in el.get("members", []):
                if member["type"] == "way" and member.get("role") != "inner": # only inner for now
                    ref_id = member.get("ref")
                    if ref_id is None:
                        continue

                    way_el = way_map.get(ref_id)
                    if way_el and "nodes" in way_el:
                        node_map = {e["id"]: e for e in self.data.get("elements", []) if e["type"] == "node"}
                        for node_id in way_el["nodes"]:
                            node = node_map.get(node_id)
                            if node:
                                coords.append(Coordinate(lat=node["lat"], lon=node["lon"]))
        return coords


    def print_json_structure(self, rec_data: Any = None, indent: int = 0):
        if rec_data is None:
            rec_data = self.data

        prefix = "  " * indent

        if isinstance(rec_data, dict):
            for key, value in rec_data.items():
                print(f"{prefix}{key}: {type(value).__name__}")
                self.print_json_structure(value, indent + 1)

        elif isinstance(rec_data, list):
            print(f"{prefix}[list of {len(rec_data)} elements]")
            if len(rec_data) > 0:
                self.print_json_structure(rec_data[0], indent + 1)

if __name__ == "__main__":
    client = OpenAirportMapClient()
    client.get_airport("EDDF")

    print("--- JSON Struktur ---")
    client.print_json_structure()

    print("\n--- Runways ---")
    runways = client.get_runways()
    for rwy in runways:
        print(rwy.ref, len(rwy.centerline_coordinates))

    print("\n--- Taxiways ---")
    taxiways = client.get_taxiways()
    for twy in taxiways:
        print(twy.open_street_map_id, len(twy.coordinates))

    print("\n--- Terminals ---")
    terminals = client.get_terminals()
    for term in terminals:
        print(term.name, len(term.coordinates))