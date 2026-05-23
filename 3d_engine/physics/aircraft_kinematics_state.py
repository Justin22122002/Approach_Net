import numpy as np
from numpy.typing import NDArray


class AircraftKinematicsState:
    def __init__(
            self,
            x: float,
            y: float,
            h: float,
            v: float,
            gamma: float,
            chi: float,
            mass: float,
            C_D0: float,
            C_L0: float,
            area: float,
            rho0: float = 1.225,
            k: float = 0.2,
            H: float = 8500.0,
    ) -> None:
        self.x: float = x         # m or lon
        self.y: float = y         # m or lat
        self.h: float = h         # m or feet
        self.v: float = v         # m/s
        self.gamma: float = gamma # γ flight path angle (Flugbahnwinkel)
        self.chi: float = chi     # 𝒳 Track angle / Course angle (Kursbahnwinkel) from North, clockwise

        self.mass: float = mass      # [kg]
        self.C_D0: float = C_D0        # drag coefficient
        self.C_L0: float = C_L0        # lift coefficient
        self.area: float = area      # wing reference area [m²]
        self.rho0: float = rho0        # air density [kg/m³]
        self.k: float = k
        self.H: float = H

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

    def drag_force(self) -> float:
        """
        D = 1/2 · ρ · v² · C_D · A
        """
        rho = self.air_density()
        CD = self.drag_coefficient()
        return 0.5 * rho * self.v ** 2 * CD * self.area

    def lift_force(self) -> float:
        """
        L = 1/2 · ρ · v² · C_L · A
        """
        rho = self.air_density()
        CL = self.lift_coefficient()
        base_lift = 0.5 * rho * self.v ** 2 * CL * self.area
        return base_lift

    def air_density(self) -> float:
        """
        ρ(h) = ρ₀ · e^−h/H
        note: h is in meter
        """
        return self.rho0 * np.exp(-self.h / self.H)

    def lift_coefficient(self) -> float:
        """
        C_L = C_L_alpha · alpha + C_L0
        note: alpha ist always 0, because of our directed flight model, e.g., see the formula:
        https://walter.bislins.ch/aviatik/index.asp?page=Berechnung+des+Auftriebs
        """
        return self.C_L0

    def drag_coefficient(self) -> float:
        """
        C_D = C_D0 + k · C_L²
        """
        CL = self.lift_coefficient()
        return self.C_D0 + self.k * CL ** 2

    def to_array(self) -> NDArray[np.float64]:
        return np.array([
            self.x,
            self.y,
            self.h,
            self.v,
            self.gamma,
            self.chi,
            self.mass,
            self.C_D0,
            self.C_L0,
            self.area
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
            C_L0=arr[8],
            area=arr[9],
        )