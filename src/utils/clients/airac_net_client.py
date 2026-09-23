import base64
import csv
import io
import zipfile
from typing import Any

import requests
from pyproj import Geod

from utils.approch import Approach
from utils.fix_identifier import FixIdentifier
from utils.arinc_runway import ArincRunway
from utils.geo_types import Coordinate
from utils.clients.sqlite_airinc24_database_client import extract_runway_from_procedure

M_TO_FT = 3.28084


class AiracNetClient:
    RUNWAYS_CSV_URL = (
        "https://raw.githubusercontent.com/davidmegginson/"
        "ourairports-data/main/runways.csv"
    )
    GATEWAY_BASE = "https://gateway.x-plane.com/apiv1"

    _geod = Geod(ellps="WGS84")

    def __init__(self) -> None:
        self.airac_base: str = "https://airac.net/api/v1"

        self.airac_headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "MyApp/1.0 (myemail@example.com)"
        }

        self._runways_csv_rows: list[dict] | None = None

    def get_airport(self, icao: str) -> Any:
        r = requests.get(
            f"{self.airac_base}/airports",
            params={"identifier": icao},
            headers=self.airac_headers
        )
        r.raise_for_status()
        return r.json()

    def get_all_procedures(self, icao: str, proc_type: str = "APP") -> Any:
        all_procedures = []
        seen_identifiers: set[str] = set()
        page = 1
        per_page = 100

        while True:
            r = requests.get(
                f"{self.airac_base}/procedures",
                params={
                    "airport": icao,
                    "type": proc_type,
                    "page": page,
                    "per_page": per_page
                },
                headers=self.airac_headers
            )
            r.raise_for_status()
            data = r.json()

            procedures = data.get("data", [])

            if not procedures:
                break

            for proc in procedures:
                identifier = proc["identifier"]
                if identifier in seen_identifiers:
                    continue
                seen_identifiers.add(identifier)
                all_procedures.append(proc)

            page += 1

        return all_procedures

    def get_procedure_detail(self, icao: str, identifier: str) -> Approach:
        r = requests.get(
            f"{self.airac_base}/procedures/{icao}/{identifier}",
            headers=self.airac_headers
        )

        r.raise_for_status()
        data = r.json()
        proc = data.get("data", [])

        proc_identifier = proc["identifier"]

        all_fixes = []

        common_route: list = proc["common_route"]
        transitions: list = proc["transitions"]

        all_fixes.extend(common_route)

        if isinstance(transitions, dict):
            for transition_name, transition_list in transitions.items():
                all_fixes.extend(transition_list)

        fixes: list[FixIdentifier] = []

        for fix in all_fixes:
            fix_coordinates = fix.get("fix_coordinates", {"lat": 0, "lon": 0})
            lat = fix_coordinates["lat"]
            lon = fix_coordinates["lon"]
            fix_identifier = fix.get("fix_identifier", "no fix name")
            path_terminator = fix["path_terminator"]
            sequence = fix["sequence"]
            altitude = fix.get("altitude_ft", 0)

            course: float | None = fix.get("course", None)
            speed_kts: float | None = fix.get("speed_kts", None)
            turn_direction: str | None = fix.get("turn_direction", None)
            altitude_restriction: str | None = fix.get("altitude_restriction", None)

            fixes.append(
                FixIdentifier(
                    fix_identifier=fix_identifier,
                    path_terminator=path_terminator,
                    latitude=lat,
                    longitude=lon,
                    altitude=altitude,
                    sequence_number=sequence,
                    course=course,
                    speed_kts=speed_kts,
                    turn_direction=turn_direction,
                    altitude_restriction=altitude_restriction,
                    description_code="",
                )
            )

        return Approach(
            procedure_identifier=proc_identifier,
            route_type=proc_identifier[0],
            fixes=fixes,
        )

    def _get_gateway_apt_dat(self, icao: str) -> str:
        r = requests.get(f"{self.GATEWAY_BASE}/airport/{icao}")
        r.raise_for_status()
        airport_data = r.json()["airport"]
        scenery_id = airport_data.get("recommendedSceneryId")

        if not scenery_id:
            raise ValueError(
                f"No recommended scenery pack for {icao} on the X-Plane Gateway "
                f"(recommendedSceneryId={scenery_id})."
            )

        r = requests.get(f"{self.GATEWAY_BASE}/scenery/{scenery_id}")
        r.raise_for_status()
        scenery_data = r.json()["scenery"]

        zip_bytes = base64.b64decode(scenery_data["masterZipBlob"])

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()

            apt_dat_name = next(
                (n for n in names if n.lower() == f"{icao.lower()}.dat"),
                None,
            )
            if apt_dat_name is None:
                apt_dat_name = next(
                    (
                        n for n in names
                        if n.lower().endswith(".dat") and not n.lower().endswith(".zip.dat")
                    ),
                    None,
                )
            if apt_dat_name is None:
                raise FileNotFoundError(
                    f"Could not locate apt.dat in scenery pack {scenery_id} for {icao}. "
                    f"Zip contents: {names}"
                )

            return zf.read(apt_dat_name).decode("utf-8", errors="ignore")

    @staticmethod
    def _parse_gateway_runways(apt_dat_text: str) -> dict[str, dict]:
        geod = AiracNetClient._geod
        runways: dict[str, dict] = {}

        for line in apt_dat_text.splitlines():
            tokens = line.strip().split()
            if not tokens or tokens[0] != "100":
                continue

            width_m = float(tokens[1])

            ends = [
                {
                    "ident": tokens[8],
                    "lat": float(tokens[9]),
                    "lon": float(tokens[10]),
                    "displaced_threshold_m": float(tokens[11]),
                    "stopway_m": float(tokens[12]),
                },
                {
                    "ident": tokens[17],
                    "lat": float(tokens[18]),
                    "lon": float(tokens[19]),
                    "displaced_threshold_m": float(tokens[20]),
                    "stopway_m": float(tokens[21]),
                },
            ]

            fwd_az, back_az, physical_length_m = geod.inv(
                ends[0]["lon"], ends[0]["lat"], ends[1]["lon"], ends[1]["lat"]
            )
            bearings = {0: fwd_az % 360, 1: back_az % 360}

            for i, end in enumerate(ends):
                other = ends[1 - i]

                thr_lon, thr_lat, _ = geod.fwd(
                    end["lon"], end["lat"], bearings[i], end["displaced_threshold_m"]
                )

                runways[end["ident"]] = {
                    "thr_lat": thr_lat,
                    "thr_lon": thr_lon,
                    "end_lat": other["lat"],
                    "end_lon": other["lon"],
                    "other_ident": other["ident"],
                    "physical_length_m": physical_length_m,
                    "landing_length_m": physical_length_m - end["displaced_threshold_m"],
                    "width_m": width_m,
                    "displaced_threshold_m": end["displaced_threshold_m"],
                    "stopway_m": end["stopway_m"],
                    "bearing": bearings[i],
                }

        return runways

    def get_runways(
            self,
            icao: str,
            magnetic_variation: float,
            airport_elevation: float,
            approaches: list[Approach]
    ) -> list[ArincRunway]:
        gateway_runways = self._parse_gateway_runways(self._get_gateway_apt_dat(icao))

        arinc_runways: list[ArincRunway] = []

        for ident, rw in gateway_runways.items():
            lat1, lon1 = rw["thr_lat"], rw["thr_lon"]
            lat2, lon2 = rw["end_lat"], rw["end_lon"]

            rw_approaches = [
                app for app in approaches
                if extract_runway_from_procedure(app.procedure_identifier) == ident
            ]

            for app in rw_approaches:
                for fix in app.fixes:
                    if self._is_runway_related_fix(fix.fix_identifier, ident) and fix.path_terminator == "TF":
                        fix.latitude = lat1
                        fix.longitude = lon1

            runway_elevation = self.get_runway_elevation(icao, ident)
            if runway_elevation is None:
                runway_elevation_ft = airport_elevation
            else:
                runway_elevation_ft = runway_elevation * M_TO_FT

            new_runway = ArincRunway(
                runway_identifier=ident,
                latitude=lat1,
                longitude=lon1,
                length=rw["landing_length_m"] * M_TO_FT,
                bearing=rw["bearing"],
                width=rw["width_m"] * M_TO_FT,
                stopway=rw["stopway_m"] * M_TO_FT,
                displaced_threshold=rw["displaced_threshold_m"] * M_TO_FT,
                magnetic_variation=magnetic_variation,
                runway_altitude=runway_elevation_ft,
                approaches=rw_approaches,
            )
            new_runway.centerline_coordinates = (
                Coordinate(lat=lat1, lon=lon1),
                Coordinate(lat=lat2, lon=lon2),
            )
            new_runway.polygon_coordinates = new_runway._compute_polygon()

            arinc_runways.append(new_runway)

        return arinc_runways

    @staticmethod
    def _is_runway_related_fix(fix_name: str, runway_id: str) -> bool:
        if not fix_name:
            return False

        fix = fix_name.upper().strip()
        rw = runway_id.upper().strip()

        return (
            fix == rw
            or fix == f"RW{rw}"
            or rw in fix
        )

    def get_airport_full(self, icao: str) -> list[ArincRunway]:
        airport = self.get_airport(icao)
        airport_data = airport.get("data", {})
        magnetic_variation = airport_data[0]["magnetic_variation"]
        airport_elevation = airport_data[0]["elevation_ft"]
        procedures = self.get_all_procedures(icao)

        approaches: list[Approach] = []
        for proc in procedures:
            approach = self.get_procedure_detail(icao, proc["identifier"])
            approaches.append(approach)

        runways = self.get_runways(
            icao,
            magnetic_variation,
            airport_elevation,
            approaches
        )

        return runways

    def _load_runways_csv(self) -> list[dict]:
        if self._runways_csv_rows is None:
            response = requests.get(self.RUNWAYS_CSV_URL)
            response.raise_for_status()
            self._runways_csv_rows = list(csv.DictReader(io.StringIO(response.text)))
        return self._runways_csv_rows

    def get_runway_elevation(
            self,
            airport_ident: str,
            runway_ident: str,
    ) -> float | None:
        for row in self._load_runways_csv():
            if row["airport_ident"] != airport_ident:
                continue

            if row["le_ident"] == runway_ident and row["le_elevation_ft"]:
                return float(row["le_elevation_ft"]) * 0.3048

            if row["he_ident"] == runway_ident and row["he_elevation_ft"]:
                return float(row["he_elevation_ft"]) * 0.3048

        return None


if __name__ == "__main__":
    client = AiracNetClient()

    airport_list_icao: list[str] = [
        "CYVR",
        "CYUL",
        "CYYZ",
        "CYYC",
        "CYEG",
    ]

    for airport_icao in airport_list_icao:
        data = client.get_airport_full(airport_icao)

        for rw in data:
            start, end = rw.centerline_coordinates
            print(
                f"{airport_icao} {rw.runway_identifier}: "
                f"thr=({rw.latitude:.6f}, {rw.longitude:.6f}) "
                f"end=({end.lat:.6f}, {end.lon:.6f}) "
                f"len={rw.length:.0f} ft, dthr={rw.displaced_threshold:.0f} ft, "
                f"elev={rw.runway_altitude:.0f} ft"
            )