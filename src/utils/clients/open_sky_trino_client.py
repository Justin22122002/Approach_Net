from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from pyopensky.trino import Trino
from pyproj import Transformer, datadir

from src.utils.clients.open_airport_map_client import get_bbox_simple
from utils.aircraft_history_interpolator import AircraftHistoryInterpolator
from utils.airport import Airport
from utils.clients.aircraft_type_client import AircraftTypeClient


class OpenSkyTrinoClient:
    """
    note: this class converts the wsg84 coordinates to the 3855 projection -> e.g. msl in meter

    when bounds are provided, the lat and lon are ignored
    """
    def __init__(
        self,
        start: datetime,
        stop: datetime,
        arrival_airport: str,
        lat: float,
        lon: float,
        radius_km: float = 10.0,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.trino: Trino = Trino()
        self.start: datetime = start
        self.stop: datetime = stop
        self.bounds: tuple[float, float, float, float] = bounds if bounds is not None else get_bbox_simple(lat, lon, radius_km)
        self.arrival_airport: str = arrival_airport
        self.aircraft_client: AircraftTypeClient = AircraftTypeClient()

        self.transformer: Transformer = Transformer.from_crs(
            "epsg:4979",
            "epsg:4326+3855",
            always_xy=True
        )

    def load_arrivals(self) -> DataFrame:
        landing_icao24s = self._fetch_landing_icao24s()

        if not landing_icao24s:
            raise ValueError(
                f"No arrivals at {self.arrival_airport} found in the given time window."
            )

        df_all = self._fetch_history()
        df_arrivals = df_all[df_all["icao24"].isin(landing_icao24s)].copy()
        return df_arrivals.reset_index(drop=True)

    def save_dataframe(
        self,
        df: DataFrame,
        filename_base: Path,
        formats: list[str] | None = None,
    ) -> None:
        formats = formats or ["csv", "json", "parquet"]

        filename_base.parent.mkdir(parents=True, exist_ok=True)

        for file_format in formats:
            match file_format.lower():
                case "csv":
                    df.to_csv(filename_base.with_suffix(".csv"), index=False)

                case "json":
                    df.to_json(
                        filename_base.with_suffix(".json"),
                        orient="records",
                        indent=2,
                        date_format="iso",
                    )

                case "parquet":
                    df.to_parquet(filename_base.with_suffix(".parquet"))

                case _:
                    raise ValueError(f"Unsupported format: {file_format}")

    def _fetch_history(self) -> DataFrame:
        df: DataFrame | None = self.trino.history(
            start=self.start,
            stop=self.stop,
            bounds=self.bounds,
            selected_columns=(
                "time",
                "icao24",
                "lat",
                "lon",
                "velocity",
                "heading",
                "vertrate",
                "callsign",
                "onground",
                "alert",
                "spi",
                "squawk",
                "baroaltitude",
                "geoaltitude",
                "lastposupdate",
                "lastcontact",
                "serials",
                "hour",
            ),
        )

        if df is None:
            raise ValueError("No data returned from Trino.")

        df["lon"] = pd.to_numeric(df["lon"], errors="coerce").fillna(0.0)
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce").fillna(0.0)
        df["geoaltitude"] = pd.to_numeric(df["geoaltitude"], errors="coerce").fillna(0.0)

        coords = [
            self.transformer.transform(lo, la, al)
            for lo, la, al in zip(
                df["lon"].values,
                df["lat"].values,
                df["geoaltitude"].values
            )
        ]

        lon_new, lat_new, alt_new = map(list, zip(*coords))

        print(f"Alt changed: {any(a1 != a2 for a1, a2 in zip(df['geoaltitude'], alt_new))}")
        print(f"Old alt sample: {list(df['geoaltitude'][:3])}")
        print(f"New alt sample: {alt_new[:3]}")

        df["lon"] = lon_new
        df["lat"] = lat_new
        df["geoaltitude"] = alt_new

        if df is None:
            raise ValueError("No data returned from Trino.")

        df = df.sort_values(["icao24", "time"]).reset_index(drop=True)

        df["time_diff"] = df.groupby("icao24")["time"].diff()
        # filter out the first row (which is NaN)
        df["time_diff"] = df["time_diff"].fillna(pd.Timedelta(0))

        df["flight_id"] = (
            (df["time_diff"] > pd.Timedelta("2h"))
            .groupby(df["icao24"])
            .cumsum()
        )

        return df

    def _fetch_landing_icao24s(self) -> set[str]:
        df: DataFrame | None = self.trino.flightlist(
            start=self.start,
            stop=self.stop,
            arrival_airport=self.arrival_airport,
        )
        if df is None or df.empty:
            return set()

        return set(df["icao24"].dropna().unique())

    def extract_aircraft_table(self, df: DataFrame) -> DataFrame:
        unique_icao24 = df["icao24"].dropna().unique()

        columns = [
            "icao24", "type", "icao_type", "manufacturer", "mode_s",
            "registration", "registered_owner", "url_photo_thumbnail",
            "mass", "C_D0", "area", "cl_max", "max_thrust",
            "max_load_factor", "min_load_factor", "max_roll_angle", "k", "name",
        ]

        rows = []

        for icao24 in unique_icao24:
            row = dict.fromkeys(columns)
            row["icao24"] = icao24

            try:
                aircraft_data, aircraft_dynamics = self.aircraft_client.get_aircraft_and_dynamics(icao24)

                row.update({
                    "type": aircraft_data.type,
                    "icao_type": aircraft_data.icao_type,
                    "manufacturer": aircraft_data.manufacturer,
                    "mode_s": aircraft_data.mode_s,
                    "registration": aircraft_data.registration,
                    "registered_owner": aircraft_data.registered_owner,
                    "url_photo_thumbnail": aircraft_data.url_photo_thumbnail,
                    "mass": aircraft_dynamics.mass,
                    "C_D0": aircraft_dynamics.C_D0,
                    "area": aircraft_dynamics.area,
                    "cl_max": aircraft_dynamics.cl_max,
                    "max_thrust": aircraft_dynamics.max_thrust,
                    "max_load_factor": aircraft_dynamics.max_load_factor,
                    "min_load_factor": aircraft_dynamics.min_load_factor,
                    "max_roll_angle": aircraft_dynamics.max_roll_angle,
                    "k": aircraft_dynamics.k,
                    "name": aircraft_dynamics.name,
                })

            except Exception as e:
                print(f"[WARN] Aircraft fetch failed {icao24}: {e}")

            rows.append(row)

        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _haversine_vectorized(
            lats1: np.ndarray, lons1: np.ndarray, lats2: np.ndarray, lons2: np.ndarray
    ) -> np.ndarray:
        R = 6371000.0
        lats1, lons1, lats2, lons2 = map(np.radians, [lats1, lons1, lats2, lons2])

        dlat = lats2 - lats1
        dlon = lons2 - lons1

        a = (
                np.sin(dlat / 2.0) ** 2
                + np.cos(lats1) * np.cos(lats2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * np.arcsin(np.sqrt(a))
        return R * c

    def assign_runway(self, df: DataFrame, airport: Airport) -> DataFrame:
        unique_fixes = {}
        for runway in airport.arinc_424_runway:
            for ap in runway.approaches:
                _, _, rw_fix = ap.get_final_approach_fixes()
                if rw_fix is not None and rw_fix.fix_identifier:
                    rwy_clean = rw_fix.fix_identifier
                    unique_fixes[rwy_clean] = (rw_fix.latitude, rw_fix.longitude, rw_fix.altitude * 0.3048)

        if not unique_fixes:
            raise ValueError("No valid runway fixes for airport found.")

        rwy_names = list(unique_fixes.keys())
        rwy_coords = np.array(list(unique_fixes.values()))
        rwy_lats = rwy_coords[:, 0]
        rwy_lons = rwy_coords[:, 1]
        rwy_alts = rwy_coords[:, 2]

        df_sorted = df.sort_values(["icao24", "flight_id", "time"])

        results = []

        for (icao24, flight_id), group in df_sorted.groupby(["icao24", "flight_id"]):
            final_phase = group[group["geoaltitude"] < 762]  # only check for landing under 762 meter -> 2500 feet
            if final_phase.empty:
                final_phase = group

            flight_lats = final_phase["lat"].to_numpy()
            flight_lons = final_phase["lon"].to_numpy()
            flight_alts = final_phase["geoaltitude"].to_numpy()

            if len(flight_lats) == 0:
                continue

            f_lats_m = flight_lats[:, np.newaxis]
            f_lons_m = flight_lons[:, np.newaxis]
            r_lats_m = rwy_lats[np.newaxis, :]
            r_lons_m = rwy_lons[np.newaxis, :]

            horizontal_dist_matrix = self._haversine_vectorized(
                f_lats_m, f_lons_m, r_lats_m, r_lons_m
            )

            f_alts_m = flight_alts[:, np.newaxis]
            r_alts_m = rwy_alts[np.newaxis, :]
            vertical_dist_matrix = f_alts_m - r_alts_m

            dist_matrix_3d = np.sqrt(horizontal_dist_matrix ** 2 + vertical_dist_matrix ** 2)

            min_dist_per_rwy = np.min(dist_matrix_3d, axis=0)
            best_rwy_idx = np.argmin(min_dist_per_rwy)

            closest_trackpoint_idx = np.argmin(dist_matrix_3d[:, best_rwy_idx])

            aircraft_landing_alt = flight_alts[closest_trackpoint_idx]
            assigned_rwy_alt = rwy_alts[best_rwy_idx]

            results.append({
                "icao24": icao24,
                "flight_id": flight_id,
                "landing_runway": rwy_names[best_rwy_idx],
                "distance_3d": min_dist_per_rwy[best_rwy_idx],
                "runway_elevation_m": assigned_rwy_alt,
                "aircraft_landing_alt_m": aircraft_landing_alt
            })

        if not results:
            return pd.DataFrame(columns=[
                "icao24",
                "flight_id",
                "landing_runway",
                "distance_3d",
                "runway_elevation_m",
                "aircraft_landing_alt_m"
            ])

        return pd.DataFrame(results)


if __name__ == "__main__":
    path: Path = Path("../../pyproj")
    datadir.append_data_dir(path)

    basePath: Path = Path("../../data")
    airport = Airport.load_from_folder(basePath, "EDDF")

    now = datetime.now(UTC)
    day = now - timedelta(days=20)

    start = day.replace(hour=19, minute=0, second=0, microsecond=0)
    stop = day.replace(hour=19, minute=30, second=0, microsecond=0)

    bounds = (8.0, 49.5, 9.5, 50.8)

    client = OpenSkyTrinoClient(
        start=start,
        stop=stop,
        arrival_airport="EDDF",
        lat=50.0379,
        lon=8.5622,
        radius_km=10.0,
        bounds=bounds
    )

    df = client.load_arrivals()
    print(df.head())
    print(f"Arriving aircraft: {df['icao24'].nunique()}")
    print(f"Total state vectors: {len(df)}")

    suffix = f"frankfurt_{start:%Y%m%d_%H%M}_{stop:%H%M}"
    history_file_path = basePath / f"arrivals_raw_{suffix}.parquet"
    df.to_parquet(history_file_path)

    print("\nStart interpolating with AircraftHistoryInterpolator...")
    interpolator = AircraftHistoryInterpolator(history_file_path, airport_height=airport.arinc_424_runway[0].runway_altitude)
    interpolator.clean_dataframe()
    interpolator.interpolate()
    interpolator.remove_onground()

    df_interpolated = interpolator.dataframe

    print("\nStart Runway-assignment...")
    df_runways = client.assign_runway(df_interpolated, airport)
    print(df_runways.head())

    print("\nLoad aircraft details")
    df_aircraft = client.extract_aircraft_table(df_interpolated)
    print(df_aircraft.head())

    client.save_dataframe(df_interpolated, Path(f"../../data/arrivals_{suffix}"))
    client.save_dataframe(df_aircraft, Path(f"../../data/aircraft_{suffix}"))
    client.save_dataframe(df_runways, Path(f"../../data/runway_assignments_{suffix}"))
    print("\n[INFO] Pipeline finished successfully!")