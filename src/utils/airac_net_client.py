from typing import Any

import requests

from src.utils.approch import FixIdentifier, Approach
from src.utils.arinc_runway import ArincRunway
from src.utils.geo_types import Coordinate
from src.utils.sqlite_airinc24_database_client import extract_runway_from_procedure


class AiracNetClient:
    def __init__(self) -> None:
        self.airac_base: str = "https://airac.net/api/v1"
        self.nav_base: str = "https://api.flightplandatabase.com/nav/airport" # https://flightplandb-py.readthedocs.io/en/v0.8.0/api/main.html -> python client, interesting -> plan -> from to icao

        self.airac_headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "MyApp/1.0 (myemail@example.com)"
        }

        self.nav_headers: dict[str, str] = {
            "Accept": "application/json"
        }

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

            all_procedures.extend(procedures)

            page += 1

        return all_procedures

    def get_procedure_detail(self, icao: str, identifier: str) -> Approach:
        params = {}

        r = requests.get(
            f"{self.airac_base}/procedures/{icao}/{identifier}",
            params=params,
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
            fix_identifier = fix["fix_identifier"]
            path_terminator = fix["path_terminator"]
            sequence = fix["sequence"]
            altitude = fix.get("altitude_ft", 0)

            fixes.append(
                FixIdentifier(
                    fix_identifier=fix_identifier,
                    path_terminator=path_terminator,
                    latitude=lat,
                    longitude=lon,
                    altitude=altitude,
                    sequence_number=sequence,
                )
            )

        #print(proc_identifier)

        return Approach(
            procedure_identifier=proc_identifier,
            route_type=proc_identifier[0],
            fixes=fixes,
        )

    def get_runways(self, icao: str, magnetic_variation: float, airport_elevation: float, approaches: list[Approach]) -> list[ArincRunway]:
        r = requests.get(
            f"{self.nav_base}/{icao}",
            headers=self.nav_headers
        )

        # here we can also get the elevation of the airport -> response.delevation
        r.raise_for_status()
        data = r.json()
        runways = data.get("runways", [])

        arinc_runways: list[ArincRunway] = []

        for rw in runways:
            ident = rw["ident"]
            length = rw["length"]
            width = rw["width"]
            thresholdOffset = rw.get("thresholdOffset", 0)
            overrunLength = rw.get("overrunLength", 0)
            bearing = rw.get("bearing", 0)

            #print(ident)

            ends = rw.get("ends", [])
            end1, end2 = ends[0], ends[1]
            lat1, lon1 = end1["lat"], end1["lon"]
            lat2, lon2 = end2["lat"], end2["lon"]

            rw_approaches = [
                app for app in approaches
                if extract_runway_from_procedure(app.procedure_identifier) == ident
            ]

            for app in rw_approaches:
                for fix in app.fixes:
                    if self._is_runway_related_fix(fix.fix_identifier, ident) and fix.path_terminator == "TF":
                        fix.latitude = lat1
                        fix.longitude = lon1

            new_runway = ArincRunway(
                    runway_identifier=ident,
                    latitude=lat1,
                    longitude=lon1,
                    length=length,
                    bearing=bearing,
                    width=width,
                    stopway=overrunLength,
                    displaced_threshold=thresholdOffset,
                    magnetic_variation=magnetic_variation,
                    runway_altitude=airport_elevation,
                    approaches=rw_approaches,
                )
            new_runway.centerline_coordinates = (
                Coordinate(lat=lat1, lon=lon1),
                Coordinate(lat=lat2, lon=lon2),
            )
            new_runway.polygon_coordinates = new_runway._compute_polygon()

            arinc_runways.append(
                new_runway
            )

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


if __name__ == "__main__":
    client = AiracNetClient()

    icao = "CYVR" # CYVR, EDDF

    data = client.get_airport_full(icao)

    print(data)

