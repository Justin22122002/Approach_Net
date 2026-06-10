from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from pyopensky.trino import Trino
from pyproj import Transformer, datadir

from utils.open_airport_map_client import get_bbox_simple


class OpenSkyTrinoClient:
    """
    note: this class converts the wsg84 coordinates to the 3855 projection -> e.g. msl in meter
    """
    def __init__(
        self,
        start: datetime,
        stop: datetime,
        arrival_airport: str,
        lat: float,
        lon: float,
        radius_km: float = 10.0,
    ) -> None:
        self.trino: Trino = Trino()
        self.start: datetime = start
        self.stop: datetime = stop
        self.bounds: tuple[float, float, float, float] = get_bbox_simple(lat, lon, radius_km)
        self.arrival_airport: str = arrival_airport

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

        return df.sort_values(["icao24", "time"]).reset_index(drop=True)

    def _fetch_landing_icao24s(self) -> set[str]:
        df: DataFrame | None = self.trino.flightlist(
            start=self.start,
            stop=self.stop,
            arrival_airport=self.arrival_airport,
        )
        if df is None or df.empty:
            return set()

        return set(df["icao24"].dropna().unique())


if __name__ == "__main__":
    path: Path = Path("../pyproj")
    datadir.append_data_dir(path)

    now = datetime.now(UTC)
    day = now - timedelta(days=10)

    start = day.replace(hour=19, minute=0, second=0, microsecond=0)
    stop  = day.replace(hour=19, minute=30, second=0, microsecond=0)

    bounds = (8.0, 49.5, 9.5, 50.8)

    client = OpenSkyTrinoClient(start=start, stop=stop, bounds=bounds, arrival_airport="EDDF")

    df = client.load_arrivals()
    print(df.head())
    print(f"Arriving aircraft: {df['icao24'].nunique()}")
    print(f"Total state vectors: {len(df)}")

    client.save_dataframe(
        df,
        Path(f"../data/arrivals_frankfurt_{start:%Y%m%d_%H%M}_{stop:%H%M}"),
    )