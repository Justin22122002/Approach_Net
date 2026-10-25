from pathlib import Path

from utils.airport import Airport
from utils.clients.open_airport_map_client import OpenAirportMapClient, only_r_approaches
from utils.clients.airport_client import OpenStreetMapAirportClient
from dotenv import load_dotenv

load_dotenv()

def test_airports() -> None:
    client = OpenStreetMapAirportClient()
    client2 = OpenAirportMapClient()

    radius_km: float = 30.0

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

def data_airports_paper() -> None:
    client = OpenAirportMapClient()

    radius_km: float = 30.0

    airports = [
        ("CYVR", 49.1947, -123.1792),  # Vancouver
        ("CYUL", 45.4706, -73.7408),   # Montreal
        ("CYYZ", 43.6777, -79.6248),   # Toronto Pearson
        ("CYYC", 51.1139, -114.0203),  # Calgary
        ("CYEG", 53.3097, -113.5800),  # Edmonton
    ]

    base_path = Path("data")

    for icao, lat, lon in airports:
        airport: Airport = client.load_new_airport(
            icao=icao,
            lat=lat,
            lon=lon,
            runway_approach_filter=only_r_approaches,
            radius_km=radius_km,
        )

        airport.save_to_folder(base_path)
        print(f"Saved {icao}")

def main() -> None:
    # test_airports()
    data_airports_paper()

if __name__ == "__main__":
    main()