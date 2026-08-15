import os
from pathlib import Path
from typing import Callable

import numpy as np
import requests
from pyproj import Geod

from src.utils.airport import Airport, AirportTerminal
from utils.terrain import Terrain
from src.utils.open_street_map_runway import Coordinate, OpenStreetMapRunway
from utils.taxiway import Taxiway
from src.utils.clients.sqlite_airinc24_database_client import SQLiteArInC24DatabaseClient
from src.utils.clients.airac_net_client import AiracNetClient
from src.utils.arinc_runway import ArincRunway
from src.utils.clients.open_airport_map_client import get_bbox_simple
from src.utils.clients.terrain_client import CopernicusDEMProvider

def only_r_approaches(r: list[ArincRunway]) -> list[ArincRunway]:
    for runway in r:
        runway.approaches = [
            app for app in runway.approaches
            if app.route_type == "R"
        ]
    return r


class OpenStreetMapAirportClient:
    OVERPASS_URL: str = "https://overpass-api.de/api/interpreter"
    OVERPASS_HEADERS = {
        "User-Agent": "MasterThesisTerrainLoader/1.0 (justin.groh@study.thws.de)",
        "Accept": "*/*",
    }

    def __init__(self, arinc_db_path: Path = Path("FAACIFP18")) -> None:
        self.arinc_client = SQLiteArInC24DatabaseClient(arinc_db_path)
        self.terrain_client: CopernicusDEMProvider = CopernicusDEMProvider(
            os.environ["COPERNICUS_USERNAME"],
            os.environ["COPERNICUS_PASSWORD"],
        )

    def load_new_airport(
            self,
            icao: str,
            lat: float,
            lon: float,
            runway_approach_filter: Callable[[list[ArincRunway]], list[ArincRunway]] = lambda r: r,
            radius_km: float = 10.0
    ) -> Airport:
        data = self._load_or_fetch(lat, lon, radius_km, icao)

        runways, taxiways, terminals = self._parse(data)
        arinc_runways: list[ArincRunway] = self.arinc_client.get_all_runways_with_approaches_for_airport(icao)
        filtered_arinc_runways: list[ArincRunway] = runway_approach_filter(arinc_runways)

        if len(arinc_runways) == 0:
            print("Get Runways from AirAC.net")
            # todo: here we can also get lat and lon directly, instead from the outside
            client: AiracNetClient = AiracNetClient()
            runways: list[ArincRunway] = client.get_airport_full(icao)
            filtered_arinc_runways = runway_approach_filter(runways)

        lat_min, lon_min, lat_max, lon_max = get_bbox_simple(
            lat, lon, radius_km
        )

        grid: np.ndarray = self.terrain_client.get_terrain(
            lat_min=lat_min,
            lon_min=lon_min,
            lat_max=lat_max,
            lon_max=lon_max,
        )

        return Airport(
            icao=icao,
            lat=lat,
            lon=lon,
            open_street_map_runway=runways,
            arinc_424_runway=filtered_arinc_runways,
            taxiways=taxiways,
            terminals=terminals,
            terrain=Terrain(
                elevation_grid=grid,
                lat_min=lat_min,
                lon_min=lon_min,
                lat_max=lat_max,
                lon_max=lon_max,
            ),
        )

    def _load_or_fetch(
            self,
            lat: float,
            lon: float,
            radius_km: float,
            icao: str,
    ) -> dict:
        query = self._build_query(lat, lon, radius_km)

        r = requests.post(self.OVERPASS_URL, data=query, timeout=None)
        r.raise_for_status()
        data = r.json()

        return data

    def _build_query(self, lat: float, lon: float, radius_km: float) -> str:
        radius_m = radius_km * 1000
        geod = Geod(ellps="WGS84")

        lon_n, lat_n, _ = geod.fwd(lon, lat, 0, radius_m)  # north
        lon_s, lat_s, _ = geod.fwd(lon, lat, 180, radius_m)  # south
        lon_e, lat_e, _ = geod.fwd(lon, lat, 90, radius_m)  # east
        lon_w, lat_w, _ = geod.fwd(lon, lat, 270, radius_m)  # west

        lamin = min(lat_s, lat_n)
        lamax = max(lat_s, lat_n)
        lomin = min(lon_w, lon_e)
        lomax = max(lon_w, lon_e)

        # way["aeroway"~"apron|gate|hangar|taxilane|terminal|parking_position|holding_position|stopway"]({lamin},{lomin},{lamax},{lomax});
        # relation["aeroway"="terminal"]({lamin},{lomin},{lamax},{lomax});
        # way["building"]({lamin},{lomin},{lamax},{lomax});

        return f"""
        [out:json][timeout:60];
        (
          way["aeroway"="runway"]({lamin},{lomin},{lamax},{lomax});
          way["aeroway"="taxiway"]({lamin},{lomin},{lamax},{lomax});

          way["aeroway"="terminal"]({lamin},{lomin},{lamax},{lomax});

          relation["building"]({lamin},{lomin},{lamax},{lomax});
        );
        out geom;
        """

    def get_buildings_and_roads(
            self,
            lat_min: float,
            lon_min: float,
            lat_max: float,
            lon_max: float,
    ) -> tuple[list[list[Coordinate]], list[list[Coordinate]]]:
        query = f"""
        [out:json][timeout:60];
        (
          way["building"]({lat_min},{lon_min},{lat_max},{lon_max});
          way["highway"]({lat_min},{lon_min},{lat_max},{lon_max});
        );
        out body;
        >;
        out skel qt;
        """
        r = requests.post(
            self.OVERPASS_URL,
            data={"data": query},
            headers=self.OVERPASS_HEADERS,
        )
        r.raise_for_status()
        data = r.json()

        node_map = {el["id"]: el for el in data.get("elements", []) if el["type"] == "node"}

        buildings: list[list[Coordinate]] = []
        roads: list[list[Coordinate]] = []

        for el in data.get("elements", []):
            if el.get("type") != "way":
                continue

            tags = el.get("tags", {})
            coords = [
                Coordinate(lat=node_map[nid]["lat"], lon=node_map[nid]["lon"])
                for nid in el.get("nodes", [])
                if nid in node_map
            ]
            if len(coords) < 2:
                continue

            if "building" in tags:
                if self._is_terminal(tags):
                    continue
                buildings.append(coords)
            elif "highway" in tags:
                roads.append(coords)

        return buildings, roads

    @staticmethod
    def _is_terminal(tags: dict) -> bool:
        return (
                tags.get("aeroway") == "terminal"
                or "terminal" in tags.get("name", "")
                or tags.get("aeroway") == "construction"
        )

    def _parse(self, data: dict) -> tuple[list[OpenStreetMapRunway], list[Taxiway], list[AirportTerminal]]:
        elements = data.get("elements", [])

        runways = self.parse_runways(elements)
        taxiways = self.parse_taxiways(elements)
        terminals = self.parse_terminals(elements)

        return runways, taxiways, terminals

    def parse_runways(self, elements: list[dict]) -> list[OpenStreetMapRunway]:
        best_runways: dict[str, dict] = {}
        for el in elements:
            coords = self._extract_coords(el)
            if not coords:
                continue
            tags = el.get("tags", {})
            aeroway = tags.get("aeroway")
            if aeroway != "runway" or tags.get("area") == "yes":
                continue
            ref = tags.get("ref", f"unknown_{el['id']}")
            if ref not in best_runways or len(coords) > best_runways[ref]["len"]:
                best_runways[ref] = {
                    "coords": coords,
                    "len": len(coords),
                    "id": el["id"],
                    "name": tags.get("name"),
                    "width": tags.get("width"),
                }

        return [
            OpenStreetMapRunway(
                open_street_map_id=str(data["id"]),
                centerline_coordinates=data["coords"],
                polygon_coordinates=[],
                ref=ref,
                name=data["name"],
                width=data["width"],
                approaches=[],
            )
            for ref, data in best_runways.items()
        ]

    def parse_taxiways(self, elements: list[dict]) -> list[Taxiway]:
        taxiways: list[Taxiway] = []
        for el in elements:
            coords = self._extract_coords(el)
            if not coords:
                continue
            tags = el.get("tags", {})
            if tags.get("aeroway") == "taxiway":
                taxiways.append(Taxiway(open_street_map_id=str(el["id"]), coordinates=coords, width=tags.get("width")))
        return taxiways

    def parse_terminals(self, elements: list[dict]) -> list[AirportTerminal]:
        terminals: list[AirportTerminal] = []
        for el in elements:
            coords = self._extract_coords(el)
            if not coords:
                continue
            tags = el.get("tags", {})
            if self._is_terminal(tags):
                terminals.append(
                    AirportTerminal(open_street_map_id=el["id"], coordinates=coords, name=tags.get("name")))
        return terminals

    def _extract_coords(self, el: dict) -> list[Coordinate]:
        coords = []
        if "geometry" in el:
            coords = [Coordinate(lat=g["lat"], lon=g["lon"]) for g in el["geometry"]]
        elif el.get("type") == "relation":
            for member in el.get("members", []):
                if member["type"] == "way" and "geometry" in member:
                    if member.get("role") != "inner":
                        coords.extend([Coordinate(lat=g["lat"], lon=g["lon"]) for g in member["geometry"]])
        return coords