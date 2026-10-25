import casadi as ca

from engine.physics.aircraft_dynamics import AircraftDynamics


class AircraftKinematicsStateCasadi:
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

    def velocity_vector(self) -> ca.MX:
        """
        3-D velocity vector in (East, North, Up):

            ⎡ v·cos(γ)·sin(χ) ⎤   ⎡ ẋ ⎤
        v⃗ = ⎢ v·cos(γ)·cos(χ) ⎥ = ⎢ ẏ ⎥
            ⎣ v·sin(γ)        ⎦   ⎣ ḣ ⎦

        γ = flight path angle
        χ = track angle from True North, clockwise
        """
        return ca.vertcat(
            self.v * ca.cos(self.gamma * (ca.pi / 180.0)) * ca.sin(self.chi * (ca.pi / 180.0)),
            self.v * ca.cos(self.gamma * (ca.pi / 180.0)) * ca.cos(self.chi * (ca.pi / 180.0)),
            self.v * ca.sin(self.gamma * (ca.pi / 180.0)),
        )

    def drag_force(self, cl: ca.MX) -> ca.MX:
        """
        D = 1/2 · ρ · v² · C_D · A
        """
        rho = self.air_density()
        CD = self.drag_coefficient(cl)
        return 0.5 * rho * self.v ** 2 * CD * self.aircraft_type.area

    def lift_force(self, cl: ca.MX) -> ca.MX:
        """
        L = 1/2 · ρ · v² · C_L · A
        """
        rho = self.air_density()
        return 0.5 * rho * self.v ** 2 * cl * self.aircraft_type.area

    def air_density(self) -> ca.MX:
        """
        H = (R · T) / g
        ρ(h) = ρ₀ · e^−h/H
        note: h is in meter
        H = scale height
        """
        return self.rho0 * ca.exp(-self.h / self.H)

    def drag_coefficient(self, cl: ca.MX) -> ca.MX:
        """
        C_D = C_D0 + k · C_L²
        """
        return self.aircraft_type.C_D0 + self.aircraft_type.k * cl ** 2

    def required_lift_coefficient(
            self,
            load_factor: ca.MX,
    ) -> ca.MX:
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

    def stall_speed(self, h: ca.MX | float | None) -> ca.MX:
        """
        V_stall = sqrt( 2·m·g / (ρ · C_L_max · S) )
        """
        if h is not None:
            rho = self.rho0 * ca.exp(-h / self.H)
        else:
            rho = self.air_density()

        return ca.sqrt(
            (2.0 * self.aircraft_type.mass * self.g)
            / (rho * self.aircraft_type.cl_max * self.aircraft_type.area)
        )

    def to_array(self) -> ca.MX:
        return ca.vertcat(
            self.x,
            self.y,
            self.h,
            self.v,
            self.gamma,
            self.chi,
            self.aircraft_type.mass,
            self.aircraft_type.C_D0,
            self.aircraft_type.area,
        )

    @classmethod
    def from_array(cls, arr: ca.MX) -> "AircraftKinematicsStateCasadi":
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