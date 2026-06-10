from pathlib import Path

from utils.airport import Airport
from src.utils.open_airport_map_client import OpenAirportMapClient, only_r_approaches
from utils.airport_client import OpenStreetMapAirportClient


def main() -> None:
    client = OpenStreetMapAirportClient()
    client2 = OpenAirportMapClient()

    radius_km: float = 20.0

    frankfurt: Airport = client2.load_new_airport(
        icao="EDDF",
        lat=50.0379,
        lon=8.5622,
        runway_approach_filter=only_r_approaches,
        radius_km=radius_km
    )

    toronto: Airport = client2.load_new_airport(
        icao="CYYZ",
        lat=43.6777,
        lon=-79.6248,
        runway_approach_filter=only_r_approaches,
        radius_km=radius_km
    )

    vancouver: Airport = client2.load_new_airport(
        icao="CYVR",
        lat=49.1947,
        lon=-123.1792,
        runway_approach_filter=only_r_approaches,
        radius_km=radius_km
    )

    jfk = client2.load_new_airport(
        icao="KJFK",
        lat=40.63993,
        lon=-73.77869,
        runway_approach_filter=only_r_approaches,
        radius_km=radius_km
    )

    basePath: Path = Path("data")
    frankfurt.save_to_folder(basePath)
    toronto.save_to_folder(basePath)
    jfk.save_to_folder(basePath)
    vancouver.save_to_folder(basePath)

if __name__ == "__main__":
    main()