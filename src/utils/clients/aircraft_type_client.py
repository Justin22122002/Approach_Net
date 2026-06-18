import openap
import requests

from physics.aircraft_dynamics import AircraftDynamics
from utils.aircraft import Aircraft, AircraftType


class AircraftTypeClient:
    BASE_URL: str = "https://api.adsbdb.com/v0/aircraft"
    # Vref = V_MARGIN * Vs_land (FAA AC 25-7 / 14 CFR 25.125)
    V_MARGIN: float = 1.3

    # CS-25 / 14 CFR 25.345(a)(1): with retracted flaps
    MAX_LOAD_FACTOR_NORMAL_OPS: float = 1.3
    MIN_LOAD_FACTOR_NORMAL_OPS: float = 0.8

    MAX_ROLL_ANGLE_NORMAL_OPS: float = 30.0

    def __init__(
        self,
        max_load_factor: float = MAX_LOAD_FACTOR_NORMAL_OPS,
        min_load_factor: float = MIN_LOAD_FACTOR_NORMAL_OPS,
        max_roll_angle: float = MAX_ROLL_ANGLE_NORMAL_OPS,
    ) -> None:
        self.max_load_factor = max_load_factor
        self.min_load_factor = min_load_factor
        self.max_roll_angle = max_roll_angle
    def retrieve_aircraft_data(self, registration_icao24: str) -> dict:
        r = requests.get(f"{self.BASE_URL}/{registration_icao24}")
        data = r.json()
        return data

    def get_aircraft_and_dynamics(self, value: str | Aircraft) -> tuple[AircraftType, AircraftDynamics]:
        aircraft_data = self.get_aircraft_type(value)
        aircraft_dynamics = self.get_aircraft_dynamics(aircraft_data.icao_type)

        return aircraft_data, aircraft_dynamics

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

    def get_aircraft_dynamics(self, icao_type: str) -> AircraftDynamics:
        ac    = openap.prop.aircraft(icao_type)
        drag  = ac["drag"]

        # Max Thrust
        thrust_model = openap.Thrust(icao_type)
        max_thrust   = thrust_model.eng_max_thrust * ac["engine"]["number"] * 1000 # convert from kn into newton

        wrap    = openap.WRAP(icao_type)
        v_app = wrap.landing_speed()["default"]
        v_stall_landing = v_app / self.V_MARGIN

        rho0    = 1.225
        g       = 9.81
        S       = ac["wing"]["area"]

        # landing weicht --> ac["mlw"], normal weight -> ac["mtow"]
        cl_max = (2 * ac["mlw"] * g) / (rho0 * v_stall_landing**2 * S)

        return AircraftDynamics(
            name            = ac["aircraft"],
            mass            = ac["mlw"],
            area            = S,
            C_D0            = drag["cd0"],
            k               = drag["k"],
            cl_max          = cl_max,
            max_thrust      = max_thrust,
            max_load_factor = self.max_load_factor,
            min_load_factor = self.min_load_factor,
            max_roll_angle  = self.max_roll_angle,
        )


def main() -> None:
    client = AircraftTypeClient()
    aircraft_type, aircraft_dynamics = client.get_aircraft_and_dynamics("3c0acb")
    print(aircraft_type)
    print(aircraft_dynamics)

if __name__ == "__main__":
    main()