import sqlite3
from collections import defaultdict
from pathlib import Path

from src.utils.arinc_runway import Approach, FixIdentifier, ArincRunway
import re

def extract_runway_from_procedure(proc_id: str) -> str | None:
    """
    R22L   -> 22L
    R22LY  -> 22L
    I04RZ  -> 04R
    RNAV22 -> 22
    """
    proc_id = proc_id.upper()

    match = re.search(r'(\d{2})([LRC]?)', proc_id)
    if not match:
        return None

    number = match.group(1)
    side = match.group(2)

    return f"{number}{side}"

def safe_float(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return default
    try:
        return float(value)
    except ValueError:
        return default

def normalize_runway(runway_id: str) -> str:
    """
    RW22L -> 22L
    RW04  -> 04
    """
    return runway_id.replace("RW", "").upper()

def magnetic_variation_to_float(var_str: str) -> float:
    """
    Convert a magnetic variation string like 'W0130' or 'E0450' to decimal degrees.
    W (west) -> negative, E (east) -> positive
    Last two digits are minutes.
    """
    if len(var_str) != 5:
        raise ValueError(f"Invalid variation string: {var_str}")

    direction = var_str[0].upper()
    degrees = int(var_str[1:4])  # 3 digits for degrees
    minutes = int(var_str[4:5]) * 10  # last digit is tenths of minutes

    decimal_degrees = degrees + minutes / 60.0

    if direction == "W":
        decimal_degrees *= -1
    elif direction != "E":
        raise ValueError(f"Unknown direction in variation string: {direction}")

    return decimal_degrees

class SQLiteArInC24DatabaseClient:

    def __init__(self, db_path: Path) -> None:
        self.db_path: Path = db_path
        self.conn: sqlite3.Connection = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def check_connection(self) -> None:
        try:
            self.conn = sqlite3.connect(database_client.db_path)
            print("Connected to database")
        except sqlite3.Error as e:
            print("Connection failed:", e)

    def get_approaches_by_airport_and_procedure(self, airport: str, procedure: str) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
                       SELECT *
                       FROM approach
                       WHERE Airport_Identifier = ?
                         AND Procedure_Identifier = ?
                       """, (airport, procedure))

        return [dict(row) for row in cursor.fetchall()]

    def get_airport_by_airport_identifier(self, airport_identifier: str) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
                       SELECT *
                       FROM airport
                       WHERE Airport_Identifier = ?
                       """, (airport_identifier,))

        return [dict(row) for row in cursor.fetchall()]

    def get_approaches_by_airport(self, airport: str) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT *
            FROM approach
            WHERE Airport_Identifier = ?
        """, (airport,))

        return [dict(r) for r in cursor.fetchall()]

    def get_waypoints(self, airport: str, waypoint: str) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
                       SELECT *
                       FROM terminal_waypoint
                       WHERE Airport_Identifier = ?
                         AND Waypoint_Identifier = ?
                       """, (airport, waypoint))

        return [dict(row) for row in cursor.fetchall()]

    def get_runways_by_airport(self, airport: str) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
                       SELECT *
                       FROM runway
                       WHERE Airport_Identifier = ?
                       """, (airport,))

        return [dict(row) for row in cursor.fetchall()]

    def get_runways_by_airport_and_runway_identifier(self, airport: str, runway_identifier: str) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
                       SELECT *
                       FROM runway
                       WHERE Airport_Identifier = ?
                           AND Runway_Identifier = ?
                       """, (airport, runway_identifier))

        return [dict(row) for row in cursor.fetchall()]

    def get_all_runways_with_approaches_for_airport(self, airport: str) -> list[ArincRunway]:
        runways_data = self.get_runways_by_airport(airport)
        airport_data = self.get_airport_by_airport_identifier(airport)
        magnetic_variation = 0

        if len(airport_data) == 1:
            variation_string = airport_data[0].get("Magnetic_Variation")

            if variation_string != "" and variation_string is not None:
                magnetic_variation = magnetic_variation_to_float(variation_string)

        runways_dict = {r["Runway_Identifier"]: r for r in runways_data}

        rows = self.get_approaches_by_airport(airport)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["Procedure_Identifier"]].append(row)

        arinc_runways: list[ArincRunway] = []

        for runway_id, runway_info in runways_dict.items():
            runway_lat = self.faa_to_decimal(runway_info.get("Latitude")) if runway_info.get("Latitude") else None
            runway_lon = self.faa_to_decimal(runway_info.get("Longitude")) if runway_info.get("Longitude") else None

            approaches_for_runway: list[Approach] = []
            runway_norm = normalize_runway(runway_id)

            for proc_id, entries in grouped.items():
                proc_runway = extract_runway_from_procedure(proc_id)
                if proc_runway != runway_norm:
                    continue

                grouped_by_procedure = defaultdict(list)
                for e in entries:
                    rtype = e.get("Route_Type") or "Unknown"
                    procedure = e.get("Procedure_Identifier") or "Unknown"
                    grouped_by_procedure[procedure].append(e)

                for procedure, r_entries in grouped_by_procedure.items():
                    fixes: list[FixIdentifier] = []

                    for entry in r_entries:
                        fix_name = entry["Fix_Identifier"]
                        path_term = entry["Path_Terminator"]
                        altitude_1 = entry["Altitude_1"]
                        seq_num = entry.get("Sequence_Number", 0)
                        lat, lon = None, None

                        if fix_name == runway_id and path_term == "TF":
                            lat, lon = runway_lat, runway_lon
                        else:
                            waypoint_data = self.get_waypoints(airport, fix_name)
                            if waypoint_data:
                                lat_raw = waypoint_data[0].get("Latitude")
                                lon_raw = waypoint_data[0].get("Longitude")
                                if lat_raw and lon_raw:
                                    lat = self.faa_to_decimal(lat_raw)
                                    lon = self.faa_to_decimal(lon_raw)

                        fixes.append(FixIdentifier(
                            fix_identifier=fix_name,
                            path_terminator=path_term,
                            latitude=lat,
                            longitude=lon,
                            altitude=float(altitude_1) if altitude_1 != "" else 0,
                            sequence_number=seq_num
                        ))

                    if any(f.fix_identifier == runway_id or f.latitude is not None for f in fixes):
                        approaches_for_runway.append(Approach(
                            procedure_identifier=proc_id,
                            route_type=rtype,
                            fixes=fixes
                        ))

            arinc_runways.append(ArincRunway(
                runway_identifier=runway_id,
                latitude=runway_lat,
                longitude=runway_lon,
                length=float(runway_info.get("Runway_Length", 0.0)),
                bearing=float(runway_info.get("Runway_Bearing", "0").lstrip("0")),
                width=float(runway_info.get("Width", 0.0)),
                stopway=float(safe_float(runway_info.get("Stopway"))),
                displaced_threshold=float(runway_info.get("Displaced_Threshold", 0.0)),
                magnetic_variation=magnetic_variation,
                runway_altitude=runway_info.get("Landing_Threshold_Elevation", 0.0),
                approaches=approaches_for_runway
            ))

        return arinc_runways

    @staticmethod
    def faa_to_decimal(coord: str) -> float:
        """
        Convert FAA coordinate string like 'N40185829' or 'W073452499' into decimal degrees.

        Formula:
            decimal = degrees + (minutes / 60) + (seconds / 3600)

        Direction N/E is positive, S/W is negative.
        """
        direction = coord[0].upper()
        value = coord[1:]

        if direction in ("N", "S"):
            deg = int(value[:2])
            minutes = int(value[2:4])
            seconds = int(value[4:]) / 100
        else:  # E/W
            deg = int(value[:3])
            minutes = int(value[3:5])
            seconds = int(value[5:]) / 100

        decimal = deg + minutes / 60 + seconds / 3600

        if direction in ("S", "W"):
            decimal *= -1

        return decimal

    def close(self):
        self.conn.close()

# N40382374,W073464329
if __name__ == "__main__":
    print(SQLiteArInC24DatabaseClient.faa_to_decimal("N40382374"))
    print(SQLiteArInC24DatabaseClient.faa_to_decimal("W073464329"))

    database_client = SQLiteArInC24DatabaseClient(Path("../FAACIFP18"))

    approaches = database_client.get_approaches_by_airport_and_procedure("KJFK", "R04LY")
    waypoints = database_client.get_waypoints("KJFK", "REPRE")

    arinc_runways = database_client.get_all_runways_with_approaches_for_airport("KJFK")

    for runway in arinc_runways:
        print(f"Runway: {runway.runway_identifier} - Lat: {runway.latitude} - Lon: {runway.longitude} - Magetic Variation: {runway.magnetic_variation} - runway_altitude: {runway.runway_altitude}")

        if not runway.approaches:
            print("  Keine Approaches für diese Runway gefunden.")
            continue

        for approach in runway.approaches:
            print(f"  Procedure: {approach.procedure_identifier}, Route Type: {approach.route_type}")
            for fix in approach.fixes:
                lat_str = f"{fix.latitude:.6f}" if fix.latitude is not None else "None"
                lon_str = f"{fix.longitude:.6f}" if fix.longitude is not None else "None"
                print(f"    {fix.fix_identifier} - {fix.path_terminator} - {lat_str} - {lon_str} - {fix.altitude}")