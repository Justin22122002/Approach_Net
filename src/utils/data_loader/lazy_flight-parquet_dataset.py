from matplotlib import pyplot as plt

import pandas as pd
from pandas import DataFrame, Series
from torch.utils.data import Dataset


class LazyFlightParquetDataset(Dataset[DataFrame]):
    def __init__(self, file_path: str) -> None:
        self.file_path: str = file_path

        meta_df: DataFrame = pd.read_parquet(
            file_path, columns=["icao24", "flight_id"]
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
            self.file_path,
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

if __name__ == "__main__":
    file_path = (
        "../../data/arrivals_frankfurt_20260530_1900_1930.parquet"
    )

    dataset = LazyFlightParquetDataset(file_path)
    print(f"Number of available flights in dataset: {len(dataset)}")

    if len(dataset) == 0:
        print("[ERROR] No flights found in the dataset.")
    else:
        df_flight = dataset[1]

        if df_flight.empty:
            print(
                "[ERROR] Filtering failed. The DataFrame is empty!"
            )
        else:
            icao24 = df_flight["icao24"].iloc[0]
            flight_id = df_flight["flight_id"].iloc[0]
            print(
                f"\n[INFO] Unique flight loaded at index 1: "
                f"Aircraft {icao24} with specific Flight ID: {flight_id}"
            )
            print(f"Number of trackpoints: {len(df_flight)}")

            df_flight = df_flight.sort_values("time")

            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(
                df_flight["lon"],
                df_flight["lat"],
                c=df_flight["geoaltitude"],
                cmap="viridis",
                s=15,
                edgecolors="none",
            )

            plt.plot(
                df_flight["lon"].iloc[0],
                df_flight["lat"].iloc[0],
                "go",
                markersize=10,
                label="Entry Point (Start)",
            )
            plt.plot(
                df_flight["lon"].iloc[-1],
                df_flight["lat"].iloc[-1],
                "ro",
                markersize=10,
                label="Landing / Fix (End)",
            )

            plt.title(
                f"Flight Path for {icao24} (ID: {flight_id}) - Arrival Frankfurt (EDDF)"
            )
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()

            cbar = plt.colorbar(scatter)
            cbar.set_label("Geometric Altitude (geoaltitude) in Meters")

            plt.show()