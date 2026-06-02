from datetime import UTC, datetime, timedelta
from pathlib import Path

from pandas import DataFrame
from pyopensky.trino import Trino


class OpenSkyTrinoClient:

    def __init__(self) -> None:
        self.trino: Trino = Trino()

    def load_history(
        self,
        start: datetime,
        stop: datetime,
        bounds: tuple[float, float, float, float],
    ) -> DataFrame:
        df = self.trino.history(
            start=start,
            stop=stop,
            bounds=bounds,
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
            raise ValueError("No data returned from Trino")

        return df.sort_values(["icao24", "time"])

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

if __name__ == "__main__":
    client = OpenSkyTrinoClient()

    now = datetime.now(UTC)
    day_before_yesterday = now - timedelta(days=10)

    start = day_before_yesterday.replace(
        hour=19,
        minute=0,
        second=0,
        microsecond=0,
    )

    stop = day_before_yesterday.replace(
        hour=19,
        minute=30,
        second=0,
        microsecond=0,
    )

    df = client.load_history(
        start=start,
        stop=stop,
        bounds=(8.0, 49.5, 9.5, 50.8),
    )

    print(df.head())
    print(f"data sets: {len(df)}")

    client.save_dataframe(
        df,
        Path(f"../data/flights_frankfurt_{start:%Y%m%d_%H%M}_{stop:%H%M}")
    )




