from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from pyopensky.trino import Trino
from pyproj import Transformer, datadir

from src.utils.clients.open_airport_map_client import get_bbox_simple
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

        rows = []

        for icao24 in unique_icao24:
            try:
                aircraft = self.aircraft_client.get_aircraft_type(icao24)

                rows.append({
                    "icao24": icao24,
                    "type": aircraft.type,
                    "icao_type": aircraft.icao_type,
                    "manufacturer": aircraft.manufacturer,
                    "mode_s": aircraft.mode_s,
                    "registration": aircraft.registration,
                    "registered_owner": aircraft.registered_owner,
                    "url_photo_thumbnail": aircraft.url_photo_thumbnail,
                })

            except Exception as e:
                print(f"[WARN] Aircraft fetch failed {icao24}: {e}")

        return pd.DataFrame(rows)


if __name__ == "__main__":
    path: Path = Path("../../pyproj")
    datadir.append_data_dir(path)

    now = datetime.now(UTC)
    day = now - timedelta(days=20)

    start = day.replace(hour=19, minute=0, second=0, microsecond=0)
    stop  = day.replace(hour=19, minute=30, second=0, microsecond=0)

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

    df_aircraft = client.extract_aircraft_table(df)

    print(df_aircraft.head())

    client.save_dataframe(
        df,
        Path(f"../../data/arrivals_frankfurt_{start:%Y%m%d_%H%M}_{stop:%H%M}"),
    )

    client.save_dataframe(
        df_aircraft,
        Path(f"../../data/aircraft_{start:%Y%m%d_%H%M}_{stop:%H%M}")
    )