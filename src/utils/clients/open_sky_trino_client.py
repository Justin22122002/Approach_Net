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

    see: https://trino.opensky-network.org/ui/ for running queries
    """
    def __init__(
        self,
        airport: Airport,
        start: datetime | None = None,
        stop: datetime | None = None,
        radius_km: float = 10.0,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.trino: Trino = Trino()
        self.start: datetime | None = start
        self.stop: datetime | None = stop

        if bounds is not None:
            self.bounds: tuple[float, float, float, float] = bounds
        else:
            lat_min, lon_min, lat_max, lon_max = get_bbox_simple(airport.lat, airport.lon, radius_km)
            self.bounds = (lon_min, lat_min, lon_max, lat_max)  # (west, south, east, north)

        self.airport: Airport = airport
        self.aircraft_client: AircraftTypeClient = AircraftTypeClient()

        self.transformer: Transformer = Transformer.from_crs(
            "epsg:4979",
            "epsg:4326+3855",
            always_xy=True
        )

    def _resolve_range(
        self, start: datetime | None, stop: datetime | None
    ) -> tuple[datetime, datetime]:
        start = start or self.start
        stop = stop or self.stop
        if start is None or stop is None:
            raise ValueError("start/stop need to be set in the constructor or passed by this method.")
        return start, stop

    def _fetch_landing_flights(
            self, start: datetime | None = None, stop: datetime | None = None
    ) -> DataFrame:
        start, stop = self._resolve_range(start, stop)

        df: DataFrame | None = self.trino.flightlist(
            start=start, stop=stop, arrival_airport=self.airport.icao,
        )
        if df is None or df.empty:
            return pd.DataFrame(columns=["icao24", "lastseen"])

        return df[["icao24", "lastseen"]].dropna(subset=["icao24", "lastseen"])

    def load_arrivals(
            self,
            start: datetime | None = None,
            stop: datetime | None = None,
            max_approach_window: timedelta = timedelta(minutes=30),
    ) -> DataFrame:
        start, stop = self._resolve_range(start, stop)

        landing_flights = self._fetch_landing_flights(start, stop)
        if landing_flights.empty:
            raise ValueError(f"No arrivals at {self.airport.icao} found in the given time window.")

        df_all = self._fetch_history(start, stop)
        if df_all.empty:
            raise ValueError(
                f"No history data returned for {self.airport.icao} in window {start} -> {stop}."
            )

        parts: list[DataFrame] = []
        for icao24, lastseen in zip(landing_flights["icao24"], landing_flights["lastseen"]):
            lastseen_ts = pd.Timestamp(lastseen)
            window_start = lastseen_ts - max_approach_window

            mask = (
                    (df_all["icao24"] == icao24)
                    & (df_all["time"] >= window_start)
                    & (df_all["time"] <= lastseen_ts)
            )
            parts.append(df_all.loc[mask])

        if not parts:
            raise ValueError(f"No matching trajectory data found for arrivals at {self.airport.icao}.")

        df_arrivals = pd.concat(parts, ignore_index=True)
        return self._assign_flight_ids(df_arrivals)

    def load_n_arrivals(
            self,
            target_n: int,
            batch: timedelta = timedelta(days=1),
            end: datetime | None = None,
            max_batches: int = 50,
            trino_lag: timedelta = timedelta(days=2),
            max_approach_window: timedelta = timedelta(minutes=30),
    ) -> DataFrame:
        end = end or (datetime.now(UTC) - trino_lag)
        total_landings: int = 0
        batch_bounds: list[tuple[datetime, datetime]] = []
        batch_stop = end

        for i in range(max_batches):
            batch_start = batch_stop - batch
            landing_flights = self._fetch_landing_flights(batch_start, batch_stop)
            n_landings = len(landing_flights)
            total_landings += n_landings
            batch_bounds.append((batch_start, batch_stop))

            print(f"[load_n_arrivals] Batch {i + 1}: {batch_start} -> {batch_stop}, (+{n_landings} arrivals, total {total_landings}/{target_n})")

            if total_landings >= target_n:
                break

            batch_stop = batch_start
        else:
            print(f"[load_n_arrivals] max_batches={max_batches} reached, only {total_landings}/{target_n} arrivals found.")

        history_parts: list[DataFrame] = []
        for batch_start, batch_stop in batch_bounds:
            try:
                history_parts.append(
                    self.load_arrivals(batch_start, batch_stop, max_approach_window=max_approach_window)
                )
            except ValueError:
                continue

        if not history_parts:
            raise ValueError(
                f"No arrivals at {self.airport.icao} found while batching for {target_n} flights."
            )

        df_combined = pd.concat(history_parts, ignore_index=True)
        return self._assign_flight_ids(df_combined)

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

    def _fetch_history(
            self, start: datetime | None = None, stop: datetime | None = None
    ) -> DataFrame:
        start, stop = self._resolve_range(start, stop)

        df: DataFrame | None = self.trino.history(
            start=start,
            stop=stop,
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

        return df.sort_values(["icao24", "time"]).reset_index(drop=True)

    @staticmethod
    def _assign_flight_ids(df: DataFrame, gap: pd.Timedelta = pd.Timedelta("2h")) -> DataFrame:
        df = df.sort_values(["icao24", "time"]).reset_index(drop=True)
        time_diff = df.groupby("icao24")["time"].diff().fillna(pd.Timedelta(0))
        df["flight_id"] = (time_diff > gap).groupby(df["icao24"]).cumsum()
        return df

    # extracts flight vie the callsign and time gap
    #@staticmethod
    #def _assign_flight_ids(
    #        df: DataFrame,
    #        gap: pd.Timedelta = pd.Timedelta("2h")
    #) -> DataFrame:
    #    df = df.sort_values(["icao24", "time"]).reset_index(drop=True)

    #    time_diff = df.groupby("icao24")["time"].diff().fillna(pd.Timedelta(0))
    #    callsign_changed = (
    #        df.groupby("icao24")["callsign"]
    #        .transform(lambda s: s.ne(s.shift()).fillna(False))
    #    )

    #    new_flight = (time_diff > gap) | callsign_changed

    #    df["flight_id"] = new_flight.groupby(df["icao24"]).cumsum()
    #    return df

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
                aircraft_data = self.aircraft_client.get_aircraft_type(icao24)
                row.update({
                    "type": aircraft_data.type,
                    "icao_type": aircraft_data.icao_type,
                    "manufacturer": aircraft_data.manufacturer,
                    "mode_s": aircraft_data.mode_s,
                    "registration": aircraft_data.registration,
                    "registered_owner": aircraft_data.registered_owner,
                    "url_photo_thumbnail": aircraft_data.url_photo_thumbnail,
                })
            except Exception as e:
                print(f"[WARN] adsbdb fetch failed {icao24}: {e}")
                rows.append(row)
                continue

            try:
                aircraft_dynamics = self.aircraft_client.get_aircraft_dynamics(aircraft_data.icao_type)
                row.update({
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
                print(f"[WARN] OpenAP dynamics fetch failed {icao24} ({aircraft_data.icao_type}): {e}")

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

    def assign_runway(
            self,
            df: DataFrame,
            airport: Airport,
            max_heading_dev: float = 10.0, # old: 35
            max_match_distance_m: float = 1000.0, # old: 300.0 # todo: split in separate things -> e.g height and lateral / long etc
            min_touchdown_speed_ms: float = 30.0,
    ) -> DataFrame:
        # note: assign_runway should only be used for interpolated and filtered data
        runway_data = []
        for runway in airport.arinc_424_runway:
            if not runway.approaches:
                continue

            ap = runway.approaches[0]

            entry_fixes, fix_before, rw_fix = ap.get_final_approach_fixes()

            if rw_fix is None or fix_before is None:
                continue

            if rw_fix is not None and rw_fix.fix_identifier:
                runway_data.append({
                    "runway_identifier": runway.runway_identifier,
                    "fix_identifier": rw_fix.fix_identifier,
                    "fix_latitude": rw_fix.latitude,
                    "fix_longitude": rw_fix.longitude,
                    "fix_altitude_m": rw_fix.altitude * 0.3048,
                    "bearing": runway.bearing,
                    "faf_identifier": fix_before.fix_identifier,
                    "faf_latitude": fix_before.latitude,
                    "faf_longitude": fix_before.longitude,
                    "faf_altitude_m": fix_before.altitude * 0.3048 if fix_before.altitude else 0.0,
                })

        if not runway_data:
            raise ValueError("No valid runway fixes for airport found.")

        rwy_lats = np.array([r["fix_latitude"] for r in runway_data])
        rwy_lons = np.array([r["fix_longitude"] for r in runway_data])
        rwy_alts = np.array([r["fix_altitude_m"] for r in runway_data])
        rwy_bearings = np.array([r["bearing"] for r in runway_data])

        df_sorted = df.sort_values(["icao24", "flight_id", "time"])
        results = []

        for (icao24, flight_id), group in df_sorted.groupby(["icao24", "flight_id"]):
            final_phase = group[group["geoaltitude"] < 762]
            if final_phase.empty:
                final_phase = group

            flight_lats = final_phase["lat"].to_numpy()
            flight_lons = final_phase["lon"].to_numpy()
            flight_alts = final_phase["geoaltitude"].to_numpy()
            flight_headings = final_phase["heading"].to_numpy()
            flight_speeds = final_phase["velocity"].to_numpy()

            if len(flight_lats) == 0:
                continue

            horizontal_dist_matrix = self._haversine_vectorized(
                flight_lats[:, np.newaxis], flight_lons[:, np.newaxis],
                rwy_lats[np.newaxis, :], rwy_lons[np.newaxis, :]
            )
            vertical_dist_matrix = flight_alts[:, np.newaxis] - rwy_alts[np.newaxis, :]
            dist_matrix_3d = np.sqrt(horizontal_dist_matrix ** 2 + vertical_dist_matrix ** 2)

            heading_diff = np.abs(flight_headings[:, np.newaxis] - rwy_bearings[np.newaxis, :])
            heading_diff = np.minimum(heading_diff, 360.0 - heading_diff)
            dist_matrix_3d[heading_diff > max_heading_dev] = np.inf

            min_dist_per_rwy = np.min(dist_matrix_3d, axis=0)

            if np.all(min_dist_per_rwy == np.inf):
                continue

            best_rwy_idx = np.argmin(min_dist_per_rwy)
            best_distance = min_dist_per_rwy[best_rwy_idx]

            if best_distance > max_match_distance_m:
                continue

            closest_trackpoint_idx = np.argmin(dist_matrix_3d[:, best_rwy_idx])
            touchdown_speed = flight_speeds[closest_trackpoint_idx]

            if touchdown_speed < min_touchdown_speed_ms:
                continue

            matched_fix = runway_data[best_rwy_idx]
            results.append({
                "icao24": icao24,
                "flight_id": flight_id,

                "landing_runway": matched_fix["runway_identifier"],

                "fix_identifier": matched_fix["fix_identifier"],
                "fix_latitude": matched_fix["fix_latitude"],
                "fix_longitude": matched_fix["fix_longitude"],
                "fix_elevation_m": matched_fix["fix_altitude_m"],

                "faf_identifier": matched_fix["faf_identifier"],
                "faf_latitude": matched_fix["faf_latitude"],
                "faf_longitude": matched_fix["faf_longitude"],
                "faf_elevation_m": matched_fix["faf_altitude_m"],

                "runway_bearing": matched_fix["bearing"],
                "distance_3d": best_distance,
                "aircraft_landing_alt_m": flight_alts[closest_trackpoint_idx],
            })

        if not results:
            return pd.DataFrame(columns=[
                "icao24", "flight_id", "landing_runway", "fix_identifier",
                "fix_latitude", "fix_longitude", "fix_elevation_m",
                "runway_bearing", "distance_3d", "aircraft_landing_alt_m"
            ])

        return pd.DataFrame(results)


def load_data(
    arrival_airport: str,
    target_n_arrivals: int,
    radius_km: float,
    suffix: str,
    folder_name: str,
    end_offset: timedelta = timedelta(0),
) -> None:
    path: Path = Path("../../pyproj")
    datadir.append_data_dir(path)

    basePath: Path = Path("../../data/")
    airport = Airport.load_from_folder(basePath, arrival_airport)

    train_path = basePath / arrival_airport / folder_name
    train_path.mkdir(parents=True, exist_ok=True)

    client = OpenSkyTrinoClient(
        airport=airport,
        radius_km=radius_km,
    )

    trino_lag = timedelta(days=2)
    end = datetime.now(UTC) - trino_lag - end_offset

    df = client.load_n_arrivals(target_n=target_n_arrivals, batch=timedelta(days=1), end=end)

    print(df.head())
    print(f"Arriving unique aircraft: {df['icao24'].nunique()}")
    print(f"Total state vectors: {len(df)}")
    print(f"[load_data:{folder_name}] window end={end} (offset={end_offset})")

    history_file_path = basePath / arrival_airport / folder_name / f"arrivals_raw_{suffix}.parquet"
    df.to_parquet(history_file_path)

    print("\nStart interpolating with AircraftHistoryInterpolator...")
    interpolator = AircraftHistoryInterpolator(
        history_file_path, airport_height=airport.arinc_424_runway[0].runway_altitude
    )
    interpolator.clean_dataframe()
    interpolator.interpolate()
    interpolator.remove_onground()

    df_interpolated = interpolator.dataframe

    print("\nStart Runway-assignment...")
    df_runways = client.assign_runway(
        df_interpolated,
        airport,
    )

    df_interpolated = df_interpolated.merge(
        df_runways[["icao24", "flight_id"]],
        on=["icao24", "flight_id"],
        how="inner",
    )

    print(df_runways.head())

    print("\nLoad aircraft details")
    df_aircraft = client.extract_aircraft_table(df_interpolated)
    print(df_aircraft.head())

    client.save_dataframe(df_interpolated, basePath / arrival_airport / folder_name / f"arrivals_{suffix}", formats=["csv", "parquet"])
    client.save_dataframe(df_aircraft, basePath / arrival_airport / folder_name / f"aircraft_{suffix}", formats=["csv", "parquet"])
    client.save_dataframe(df_runways, basePath / arrival_airport / folder_name / f"runway_assignments_{suffix}", formats=["csv", "parquet"])
    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    airport_list: list[str] = [
        "CYVR",
        "CYUL",
        "CYYZ",
        "CYYC",
        "CYEG",
    ]
    target_n_arrivals_test: int = 500 # 2000, 500
    target_n_arrivals_train: int = 10000 # 2000, 500
    test_offset: timedelta = timedelta(days=90)
    radius_km: float = 30.0
    suffix = datetime.now(UTC).strftime("%Y%m%d_%H%M")

    for airport in airport_list:
        # 20260708_1247 -> old with 2000
        # 20260802_0920 -> new with 10000
        load_data(
            arrival_airport=airport,
            target_n_arrivals=target_n_arrivals_train,
            radius_km=radius_km,
            suffix=suffix,
            folder_name="train",
        )

        load_data(
            arrival_airport=airport,
            target_n_arrivals=target_n_arrivals_test,
            radius_km=radius_km,
            suffix=suffix,
            folder_name="test",
            end_offset=test_offset,
        )