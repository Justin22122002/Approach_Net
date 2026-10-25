import pandas as pd
from pandas import DataFrame
from pyopensky.api import HasBounds
from pyopensky.config import opensky_config_dir
from pyopensky.rest import REST
from pyproj import Geod


class AdjustedRest(REST):

    _json_columns = (
        "icao24",
        "callsign",
        "origin_country",
        "last_position",
        "timestamp",
        "longitude",
        "latitude",
        "altitude",
        "onground",
        "groundspeed",
        "track",
        "vertical_rate",
        "sensors",
        "geoaltitude",
        "squawk",
        "spi",
        "position_source",
        "category",
    )

    def __init__(self):
        super().__init__()

    def states(
        self,
        own: bool = False,
        bounds: None
        | str
        | HasBounds
        | tuple[float, float, float, float] = None,
        retry: int = 5,
    ) -> pd.DataFrame:
        what = "own" if (own and self.headers != {}) else "all"

        if bounds is not None:
            if hasattr(bounds, "bounds"):
                west, south, east, north = getattr(bounds, "bounds")
            else:
                assert isinstance(bounds, tuple)
                west, south, east, north = bounds

            what += f"?lamin={south}&lamax={north}&lomin={west}&lomax={east}"

        json = self.get(f"https://opensky-network.org/api/states/{what}&extended=1")
        columns: list[str] = list(self._json_columns)

        if len(json["states"]) > 0:
            if len(json["states"][0]) > len(self._json_columns):
                columns.append("_")

        r = pd.DataFrame.from_records(
            json["states"], columns=columns
        ).convert_dtypes(dtype_backend="pyarrow")

        return r.assign(
            timestamp=lambda df: pd.to_datetime(
                df.timestamp, utc=True, unit="s"
            ),
            last_position=lambda df: pd.to_datetime(
                df.last_position, utc=True, unit="s"
            ),
            callsign=lambda df: df.callsign.str.strip(),
        )

class OpenSkyRestClient:
    def __init__(self) -> None:
        self.rest: AdjustedRest = AdjustedRest()

    # todo: provide async option as well
    def get_flights(self, lamin: float, lomin: float, lamax: float, lomax: float) -> DataFrame:
        """
        /api/states/all?lamin=...&lomin=...&lamax=...&lomax=...
        west, south, east, north = getattr(bounds, "bounds")
        what += f"?lamin={south}&lamax={north}&lomin={west}&lomax={east}"

        &extended=1 -> for aircraft category
        """
        df = self.rest.states(
            own=False,
            bounds=(lomin, lamin, lomax, lamax),
            retry=1,
        )
        return df

    def get_flights_in_area(
            self,
            radius_km: float,
            lat_center: float,
            lon_center: float,
            log: bool = False
    ) -> DataFrame:
        geod = Geod(ellps="WGS84")
        half_edge_m = radius_km * 1000 # convert from km to m

        _, lat_north, _ = geod.fwd(lon_center, lat_center, 0, half_edge_m)
        _, lat_south, _ = geod.fwd(lon_center, lat_center, 180, half_edge_m)
        lon_east, _, _ = geod.fwd(lon_center, lat_center, 90, half_edge_m)
        lon_west, _, _ = geod.fwd(lon_center, lat_center, 270, half_edge_m)

        lamin = lat_south
        lamax = lat_north
        lomin = lon_west
        lomax = lon_east

        if log:
            print("Bounding Box in WGS84 (Grad):")
            print(f"lat_min: {lamin}, lat_max: {lamax}")
            print(f"lon_min: {lomin}, lon_max: {lomax}")

        return self.get_flights(lamin, lomin, lamax, lomax)

    def get_arrivals_by_airport(self, icao: str):
        pass

    def get_departures_by_airport(self, icao: str):
        pass

    @staticmethod
    def get_config_info() -> None:
        print(opensky_config_dir)


if __name__ == "__main__":
    from pyopensky.config import opensky_config_dir

    print(opensky_config_dir)

    client = OpenSkyRestClient()

    lat_center = 50.0379
    lon_center = 8.5622
    half_edge_km = 10

    df = client.get_flights_in_area(half_edge_km, lat_center, lon_center)

    if df.empty:
        print("Keine Flugzeuge im 10 km Quadrat gefunden.")
    else:
        print(df.head(len(df)))
        print(df.columns)
        print(df.dtypes)