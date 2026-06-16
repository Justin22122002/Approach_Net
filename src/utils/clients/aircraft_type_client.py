import requests

from utils.aircraft import Aircraft, AircraftType


class AircraftTypeClient:
    BASE_URL: str = "https://api.adsbdb.com/v0/aircraft"

    def retrieve_aircraft_data(self, registration_icao24: str) -> dict:
        r = requests.get(f"{self.BASE_URL}/{registration_icao24}")
        data = r.json()
        return data


    def get_aircraft_type(self, value: str | Aircraft) -> AircraftType:
        if isinstance(value, Aircraft):
            if value.icao24 is None:
                raise ValueError("Aircraft must have an ICAO24 code.")
            data = self.retrieve_aircraft_data(value.icao24)

        elif isinstance(value, str):
            data = self.retrieve_aircraft_data(value)

        else:
            raise TypeError("Expected str or Aircraft")

        aircraft_data = data["response"]["aircraft"]

        return AircraftType(
            type=aircraft_data.get("type"),
            icao_type=aircraft_data.get("icao_type"),
            manufacturer=aircraft_data.get("manufacturer"),
            mode_s=aircraft_data.get("mode_s"),
            registration=aircraft_data.get("registration"),
            registered_owner=aircraft_data.get("registered_owner"),
            url_photo_thumbnail=aircraft_data.get("url_photo_thumbnail"),
        )


def main():
    client = AircraftTypeClient()
    data = client.get_aircraft_type("3c0acb")
    print(data)

if __name__ == "__main__":
    main()