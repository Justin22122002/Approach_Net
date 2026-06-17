import numpy as np
from numpy.typing import NDArray

from physics.aircraft_type import AircraftDynamics


class AircraftKinematicsState:
    def __init__(
            self,
            x: float,
            y: float,
            h: float,
            v: float,
            gamma: float,
            chi: float,
            aircraft_type: AircraftDynamics,
            rho0: float = 1.225,
    ) -> None:
        self.x: float = x            # m or lon
        self.y: float = y            # m or lat
        self.h: float = h            # m or feet
        self.v: float = v            # m/s
        self.gamma: float = gamma    # γ flight path angle (Flugbahnwinkel)
        self.chi: float = chi        # 𝒳 Track angle / Course angle (Kursbahnwinkel) from North, clockwise

        self.aircraft_type: AircraftDynamics = aircraft_type

        self.rho0: float = rho0      # air density [kg/m³]
        self.g: float = 9.81         # earth gravity
        self.R: float = 287.05       # gas constant for air [J/(kg·K)]
        self.T: float = 288.15       # temperature in kelvin [K]
        # H = (R · T) / g
        self.H: float = (self.R * self.T) / self.g

    def velocity_vector(self) -> NDArray[np.float64]:
        """
        3-D velocity vector in (East, North, Up):

            ⎡ v·cos(γ)·sin(χ) ⎤   ⎡ ẋ ⎤
        v⃗ = ⎢ v·cos(γ)·cos(χ) ⎥ = ⎢ ẏ ⎥
            ⎣ v·sin(γ)        ⎦   ⎣ ḣ ⎦

        γ = flight path angle
        χ = track angle from True North, clockwise
        """
        return np.array([
            self.v * np.cos(np.radians(self.gamma)) * np.sin(np.radians(self.chi)),
            self.v * np.cos(np.radians(self.gamma)) * np.cos(np.radians(self.chi)),
            self.v * np.sin(np.radians(self.gamma)),
        ])

    def drag_force(self, cl: float,) -> float:
        """
        D = 1/2 · ρ · v² · C_D · A
        """
        rho = self.air_density()
        CD = self.drag_coefficient(cl)
        return 0.5 * rho * self.v ** 2 * CD * self.aircraft_type.area

    def lift_force(self, cl: float) -> float:
        """
        L = 1/2 · ρ · v² · C_L · A
        """
        rho = self.air_density()
        return 0.5 * rho * self.v ** 2 * cl * self.aircraft_type.area

    def air_density(self) -> float:
        """
        H = (R · T) / g
        ρ(h) = ρ₀ · e^−h/H
        note: h is in meter
        H = scale height
        """
        return self.rho0 * np.exp(-self.h / self.H)

    def drag_coefficient(self, cl: float) -> float:
        """
        C_D = C_D0 + k · C_L²
        """
        return self.aircraft_type.C_D0 + self.aircraft_type.k * cl ** 2

    def required_lift_coefficient(
            self,
            load_factor: float,
    ) -> float:
        """
        Required lift coefficient for a desired load factor.

        Load factor:
            n = L / (m·g) -> L = n·m·g

        Lift equation:
            L = 1/2 · ρ · v² · C_L · S

        solving for C_L:
            C_L = (2·L)  / (ρ · v² · S)
            C_L = (2·n·m·g)  / (ρ · v² · S)

        old formula:
        C_L = C_L_alpha · alpha + C_L0
        note: alpha ist always 0, because of our directed flight model, e.g., see the formula:
        https://walter.bislins.ch/aviatik/index.asp?page=Berechnung+des+Auftriebs
        """

        rho = self.air_density()

        return (
                2.0 * load_factor * self.aircraft_type.mass * self.g
                / (rho * self.v ** 2 * self.aircraft_type.area)
        )

    def stall_speed(self) -> float:
        """
        V_stall = sqrt( 2·m·g / (ρ · C_L_max · S) )
        """
        rho = self.air_density()
        return np.sqrt(
            (2.0 * self.aircraft_type.mass * self.g)
            / (rho * self.aircraft_type.cl_max * self.aircraft_type.area)
        )

    def to_array(self) -> NDArray[np.float64]:
        return np.array([
            self.x,
            self.y,
            self.h,
            self.v,
            self.gamma,
            self.chi,
            self.aircraft_type.mass,
            self.aircraft_type.C_D0,
            self.aircraft_type.area
        ])

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "AircraftKinematicsState":
        return cls(
            x=arr[0],
            y=arr[1],
            h=arr[2],
            v=arr[3],
            gamma=arr[4],
            chi=arr[5],
            mass=arr[6],
            C_D0=arr[7],
            area=arr[8],
        )