import json
from pathlib import Path

import requests
from pyproj import Geod

from src.utils.airport import Airport, AirportTerminal
from src.utils.runway import Coordinate, Runway, Taxiway


class AirportClient:
    OVERPASS_URL: str = "https://overpass-api.de/api/interpreter"

    def __init__(self, cache_dir: str = ".cache") -> None:
        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_airport(self, icao: str, lat: float, lon: float, radius_km: float = 10) -> Airport:
        data = self._load_or_fetch(lat, lon, radius_km, icao)

        runways, taxiways, terminals = self._parse(data)

        return Airport(
            icao=icao,
            lat=lat,
            lon=lon,
            runways=runways,
            taxiways=taxiways,
            terminals=terminals,
        )

    def _load_or_fetch(
        self,
        lat: float,
        lon: float,
        radius_km: float,
        icao: str,
    ) -> dict:
        cache_file = self.cache_dir / f"{icao}.json"

        # read
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        # fetch
        query = self._build_query(lat, lon, radius_km)

        r = requests.post(self.OVERPASS_URL, data=query, timeout=None)
        r.raise_for_status()
        data = r.json()

        # write
        cache_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return data

    def _build_query(self, lat: float, lon: float, radius_km: float) -> str:
        radius_m = radius_km * 1000
        geod = Geod(ellps="WGS84")

        lon_n, lat_n, _ = geod.fwd(lon, lat, 0, radius_m)    # north
        lon_s, lat_s, _ = geod.fwd(lon, lat, 180, radius_m)  # south
        lon_e, lat_e, _ = geod.fwd(lon, lat, 90, radius_m)   # east
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

    def _parse(self, data: dict) -> tuple[list[Runway], list[Taxiway], list[AirportTerminal]]:
        elements = data.get("elements", [])

        runways = self.parse_runways(elements)
        taxiways = self.parse_taxiways(elements)
        terminals = self.parse_terminals(elements)

        return runways, taxiways, terminals


    def parse_runways(self, elements: list[dict]) -> list[Runway]:
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
            Runway(
                id=str(data["id"]),
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
                taxiways.append(Taxiway(id=str(el["id"]), coordinates=coords, width=tags.get("width")))
        return taxiways


    def parse_terminals(self, elements: list[dict]) -> list[AirportTerminal]:
        terminals: list[AirportTerminal] = []
        for el in elements:
            coords = self._extract_coords(el)
            if not coords:
                continue
            tags = el.get("tags", {})
            aeroway = tags.get("aeroway")
            if aeroway == "terminal" or tags.get("building") == "terminal" or "terminal" in tags.get("name", "").lower():
                terminals.append(AirportTerminal(id=el["id"], coordinates=coords, name=tags.get("name")))
        return terminals


    def _extract_coords(self, el: dict) -> list[Coordinate]:
        coords = []
        if "geometry" in el:
            coords = [Coordinate(lat=g["lat"], lon=g["lon"]) for g in el["geometry"]]
        elif el.get("type") == "relation":
            for member in el.get("members", []):
                if member["type"] == "way" and "geometry" in member:
                    if member.get("role") != "inner":  # nur outer
                        coords.extend([Coordinate(lat=g["lat"], lon=g["lon"]) for g in member["geometry"]])
        return coords