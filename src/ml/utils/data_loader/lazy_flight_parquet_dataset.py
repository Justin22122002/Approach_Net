from pathlib import Path
from matplotlib import pyplot as plt

import pandas as pd
from pandas import DataFrame, Series
from torch.utils.data import Dataset

from engine.physics.aircraft_dynamics import AircraftDynamics
from utils.aircraft import AircraftType
from utils.airport import Airport
from utils.approch import Approach
from utils.arinc_runway import ArincRunway


class LazyFlightParquetDataset(Dataset[DataFrame]):
    """
        ../../data/frankfurt_20260530_1900_1930
        -> folder structure
        arrivals_frankfurt_20260530_1900_1930.parquet
        aircraft_frankfurt_20260530_1900_1930.parquet
        runway_assignments_frankfurt_20260530_1900_1930.parquet

    """
    def __init__(self, base_path: str | Path) -> None:
        base_path: str | Path = Path(base_path)
        self.dataset_path: str | Path = base_path.parent
        self.airport_path: str | Path = base_path.parent.parent
        suffix: str = base_path.name

        self.flight_path: Path   = self.dataset_path / f"arrivals_{suffix}.parquet"
        self.aircraft_path: Path = self.dataset_path / f"aircraft_{suffix}.parquet"
        self.runway_path: Path   = self.dataset_path / f"runway_assignments_{suffix}.parquet"

        self._aircraft_cache: dict[str, tuple[AircraftType, AircraftDynamics]] | None = None
        self._runway_cache: DataFrame | None = None
        self._airport_cache: Airport | None = None

        meta_df: DataFrame = pd.read_parquet(
            self.flight_path, columns=["icao24", "flight_id"]
        )

        meta_df["flight_id"] = meta_df["flight_id"].astype("Int64")
        meta_df["icao24"] = meta_df["icao24"].astype(str)

        meta_df = meta_df.dropna(subset=["icao24", "flight_id"])

        unique_flights: DataFrame = meta_df.drop_duplicates(
            ["icao24", "flight_id"]
        )

        self.flights: list[tuple[str, int]] = [
            (str(row["icao24"]), int(row["flight_id"]))
            for _, row in unique_flights.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.flights)

    def __getitem__(self, idx: int) -> DataFrame:
        icao24: str
        flight_id: int
        icao24, flight_id = self.flights[idx]

        df_flight: DataFrame = pd.read_parquet(
            self.flight_path,
            filters=[
                ("icao24", "==", str(icao24)),
                ("flight_id", "==", int(flight_id)),
            ],
        )
        # sort by time
        if "time" in df_flight.columns:
            df_flight = df_flight.sort_values("time")

        return df_flight

    def get_start_and_end_points(self, df_flight: DataFrame) -> tuple[Series, Series]:
        if df_flight.empty:
            raise ValueError("The provided flight DataFrame is empty.")

        start_point: Series = df_flight.iloc[0]
        end_point: Series = df_flight.iloc[-1]

        return start_point, end_point

    def load_aircraft_table(self) -> dict[str, tuple[AircraftType, AircraftDynamics]]:
        if self._aircraft_cache is None:
            df = pd.read_parquet(self.aircraft_path)
            cache: dict[str, tuple[AircraftType, AircraftDynamics]] = {}
            for _, row in df.iterrows():
                if pd.isna(row.get("mass")):
                    continue
                cache[row["icao24"]] = (AircraftType.from_row(row), AircraftDynamics.from_row(row))
            self._aircraft_cache = cache
        return self._aircraft_cache

    def get_aircraft(self, icao24: str) -> tuple[AircraftType, AircraftDynamics]:
        table = self.load_aircraft_table()
        if icao24 not in table:
            raise KeyError(f"No aircraft data for icao24={icao24!r}.")
        return table[icao24]

    def load_runway_assignments(self) -> DataFrame:
        if self._runway_cache is None:
            self._runway_cache = pd.read_parquet(self.runway_path)
        return self._runway_cache

    def get_runway_assignment(self, icao24: str, flight_id: int) -> Series:
        df = self.load_runway_assignments()
        match = df[(df["icao24"] == icao24) & (df["flight_id"] == flight_id)]
        if match.empty:
            raise KeyError(f"No runway assignment for icao24={icao24!r}, flight_id={flight_id!r}.")
        return match.iloc[0]

    def load_airport(self) -> Airport:
        if self._airport_cache is None:
            icao = self.airport_path.name

            airport_data_path = (
                    Path(__file__).resolve().parents[3] / "data"
            )

            self._airport_cache = Airport.load_from_folder(
                airport_data_path,
                icao,
            )

        return self._airport_cache

    """
    get_procedure("26L") -> self.get_procedure("26L")
    """
    def get_procedure(self, runway_name: str, approach_index: int = 0) -> tuple[Approach, ArincRunway]:
        airport: Airport = self.load_airport()

        searched_runway: ArincRunway = None
        for runway in airport.arinc_424_runway:
            if runway.runway_identifier == runway_name:
                searched_runway = runway
                break

        return searched_runway.approaches[approach_index], searched_runway

if __name__ == "__main__":
    base_path = "../../../data/CYVR/train/20260802_0920"

    dataset = LazyFlightParquetDataset(base_path)
    print(f"Number of available flights in dataset: {len(dataset)}")

    if len(dataset) == 0:
        print("[ERROR] No flights found in the dataset.")
    else:
        df_flight = dataset[0]

        if df_flight.empty:
            print("[ERROR] Filtering failed. The DataFrame is empty!")
        else:
            icao24 = df_flight["icao24"].iloc[0]
            flight_id = df_flight["flight_id"].iloc[0]
            print(f"\n[INFO] Unique flight loaded at index 1: Aircraft {icao24} with Flight ID: {flight_id}")
            print(f"Number of trackpoints: {len(df_flight)}")

            aircraft_type, aircraft_dynamics = dataset.get_aircraft(icao24)
            print(aircraft_type)
            print(aircraft_dynamics)

            runway_row = dataset.get_runway_assignment(icao24, flight_id)
            print(runway_row)

            df_flight = df_flight.sort_values("time")

            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(
                df_flight["lon"], df_flight["lat"],
                c=df_flight["geoaltitude"], cmap="viridis", s=15, edgecolors="none",
            )
            plt.plot(df_flight["lon"].iloc[0], df_flight["lat"].iloc[0], "go", markersize=10, label="Entry Point (Start)")
            plt.plot(df_flight["lon"].iloc[-1], df_flight["lat"].iloc[-1], "ro", markersize=10, label="Landing / Fix (End)")
            plt.title(f"Flight Path for {icao24} (ID: {flight_id}) - Arrival Frankfurt (EDDF)")
            plt.xlabel("Longitude"); plt.ylabel("Latitude")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()

            cbar = plt.colorbar(scatter)
            cbar.set_label("Geometric Altitude (geoaltitude) in Meters")

            plt.show()