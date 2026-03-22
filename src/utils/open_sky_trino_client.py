from datetime import UTC, datetime, timedelta
from pathlib import Path

from pandas import DataFrame
from pyopensky.trino import Trino


class OpenSkyTrinoClient:

    def __init__(self) -> None:
        self.trino: Trino = Trino()

if __name__ == "__main__":
    client = OpenSkyTrinoClient()

    now = datetime.now(UTC)
    day_before_yesterday = now - timedelta(days=2)

    start = day_before_yesterday.replace(hour=19, minute=0, second=0, microsecond=0)
    stop  = day_before_yesterday.replace(hour=19, minute=30, second=0, microsecond=0)

    df: DataFrame | None = client.trino.history(
        start=start,
        stop=stop,
        bounds=(8.0, 49.5, 9.5, 50.8),
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
        # limit=20000
    )

    if df is None:
        raise ValueError("No data returned from Trino")

    print(df.head())
    print(f"Datensätze: {len(df)}")

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    filename_base = data_dir / f"flights_frankfurt_{start:%Y%m%d_%H%M}_{stop:%H%M}"

    df = df.sort_values(["icao24", "time"])

    df.to_csv(filename_base.with_suffix(".csv"), index=False)

    df.to_json(filename_base.with_suffix(".json"), orient="records", indent=2)

    df.to_parquet(filename_base.with_suffix(".parquet"))

    print(f"Gespeichert unter:\n{filename_base.with_suffix('.csv')}\n{filename_base.with_suffix('.json')}\n{filename_base.with_suffix('.parquet')}")




