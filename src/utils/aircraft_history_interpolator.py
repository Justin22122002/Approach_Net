import math
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame


# todo: add extra data loader for batch load, for bigger datasets
class AircraftHistoryInterpolator:
    """
    this class serves for interpolating and cleaning the aircraft history data
    """
    def __init__(
            self,
            filepath: Path,
            airport_height: float,
    ) -> None:
        self.filepath: Path = filepath
        self.dataframe: DataFrame | None = None
        self.airport_height: float = airport_height * 0.3048

        self._load_file_by_file_path()

    def _load_file_by_file_path(self) -> None:
        self.dataframe = pd.read_parquet(self.filepath)

    def clean_dataframe(self) -> None:
        # fill nan values with empty strings for later usage
        self.dataframe["serials"] = self.dataframe["serials"].apply(AircraftHistoryInterpolator.clean_serial_list)
        self.dataframe["callsign"] = self.dataframe["callsign"].fillna("")
        self.dataframe["velocity"] = self.dataframe["velocity"].ffill()

        cleaned_groups: list[DataFrame] = []
        # remove duplicates with the same latitude and longitude for same aircraft a keep only the first one depending on the timestamp
        for icao, group in self.dataframe.groupby("icao24"):
            # sort by timestamp first
            group = group.sort_values("time")
            duplicates_mask = group.duplicated(subset=["lat", "lon"], keep="first")
            group = group[~duplicates_mask]
            cleaned_groups.append(group)

        self.dataframe = pd.concat(cleaned_groups).reset_index(drop=True)

    def interpolate(self) -> None:
        interpolated_groups: list[DataFrame] = []

        for icao, group in self.dataframe.groupby("icao24"):
            group = group.sort_values("time").copy()

            # clip values to avoid altitudes under the airport
            group["geoaltitude"] = group["geoaltitude"].fillna(0).clip(lower=self.airport_height)

            traj_rows = []

            for i in range(len(group) - 1):
                row1 = group.iloc[i]
                row2 = group.iloc[i + 1]

                t1, t2 = int(row1["time"].timestamp()), int(row2["time"].timestamp())
                dt = t2 - t1
                if dt <= 1:
                    traj_rows.append(row1.to_dict())
                    continue

                lat1, lon1 = row1["lat"], row1["lon"]
                baroaltitude = row1["baroaltitude"] if pd.notna(row1["baroaltitude"]) else 0
                geoaltitude = row1["geoaltitude"] if pd.notna(row1["geoaltitude"]) else 0
                geoaltitude2 = row2["geoaltitude"] if pd.notna(row2["geoaltitude"]) else geoaltitude
                vel = row1["velocity"] if pd.notna(row1["velocity"]) else 0
                vert_rate = row1["vertrate"] if pd.notna(row1["vertrate"]) else 0
                heading = row1["heading"] if pd.notna(row1["heading"]) else 0

                row1_onground = bool(row1.get("onground", False))
                row2_onground = bool(row2.get("onground", False))

                for s in range(dt):
                    fraction = s / dt

                    new_lat = lat1 + fraction * (row2["lat"] - lat1)
                    new_lon = lon1 + fraction * (row2["lon"] - lon1)
                    new_baroaltitude = baroaltitude + fraction * (row2["baroaltitude"] - baroaltitude) if pd.notna(row2["baroaltitude"]) else baroaltitude

                    linear_geoaltitude = geoaltitude + fraction * (geoaltitude2 - geoaltitude)
                    new_geoaltitude = max(linear_geoaltitude, self.airport_height)
                    is_onground = row1_onground if fraction < 0.5 else row2_onground

                    traj_rows.append({
                        "time": pd.Timestamp(t1 + s, unit='s', tz='UTC'),
                        "icao24": icao,
                        "lat": new_lat,
                        "lon": new_lon,
                        "baroaltitude": new_baroaltitude,
                        "geoaltitude": new_geoaltitude,
                        "velocity": vel,
                        "heading": heading,
                        "vertrate": vert_rate,
                        "onground": is_onground,
                        "callsign": row1.get("callsign", ""),
                        "squawk": row1.get("squawk", ""),
                        "spi": row1.get("spi", False),
                        "serials": row1.get("serials", [])
                    })

            traj_rows.append(group.iloc[-1].to_dict())

            interpolated_groups.append(pd.DataFrame(traj_rows))

        self.dataframe = pd.concat(interpolated_groups).reset_index(drop=True)

    def remove_onground(self) -> None:
        if self.dataframe is None or self.dataframe.empty:
            return

        onground = self.dataframe["onground"].fillna(False).astype(bool)
        self.dataframe = self.dataframe[~onground].reset_index(drop=True)

    @staticmethod
    def clean_serial_list(x) -> list[int]:
        if x is None:
            return []
        if isinstance(x, float) and math.isnan(x):
            return []
        if isinstance(x, list):
            return x
        return [x]

def plot_aircraft_trajectory(interpolator: AircraftHistoryInterpolator, icao24: str):
    if interpolator.dataframe is None or interpolator.dataframe.empty:
        print("No data available.")
        return

    df_plane = interpolator.dataframe[interpolator.dataframe["icao24"] == icao24]

    if df_plane.empty:
        print(f"No data for aircraft {icao24}")
        return

    plt.figure(figsize=(10, 8))

    plt.scatter(df_plane["lon"], df_plane["lat"], s=10, color='blue', label="Positions")

    plt.scatter(df_plane["lon"].iloc[0], df_plane["lat"].iloc[0], color='green', s=50, label="Start")
    plt.scatter(df_plane["lon"].iloc[-1], df_plane["lat"].iloc[-1], color='red', s=50, label="End")

    plt.title(f"Trajectory Points of Aircraft {icao24}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.show()

