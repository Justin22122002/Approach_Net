
class AircraftDynamics:
    """
    those are the maximum values for the aircraft kinematics, these are not dependent
    """
    def __init__(
            self,
            mass: float,
            C_D0: float,
            area: float,
            cl_max: float,
            max_thrust: float,
            max_load_factor: float,
            min_load_factor: float,
            max_roll_angle: float,
            k: float  = 0.2,
            name: str = "default",
    ) -> None:
        self.cl_max: float = cl_max
        self.mass: float = mass # [kg] -> this is the landing weight !!!
        self.C_D0: float = C_D0 # drag coefficient
        self.area: float = area # wing reference area [m²]
        self.k: float = k

        self.max_thrust: float = max_thrust
        self.max_load_factor: float = max_load_factor
        self.min_load_factor: float = min_load_factor
        self.max_roll_angle: float = max_roll_angle

        self.name: str = name

    def __repr__(self) -> str:
        w = 28
        return (
            f"  AircraftType: {self.name}\n"
            f"  {'mass':<{w}} {self.mass:>10.0f} kg\n"
            f"  {'area':<{w}} {self.area:>10.1f} m²\n"
            f"  {'C_D0':<{w}} {self.C_D0:>10.4f}\n"
            f"  {'k':<{w}} {self.k:>10.4f}\n"
            f"  {'cl_max':<{w}} {self.cl_max:>10.3f}\n"
            f"  {'max_thrust':<{w}} {self.max_thrust / 1000:>10.1f} kN\n"
            f"  {'max_load_factor':<{w}} {self.max_load_factor:>10.1f} g\n"
            f"  {'min_load_factor':<{w}} {self.min_load_factor:>10.1f} g\n"
            f"  {'max_roll_angle':<{w}} {self.max_roll_angle:>10.1f} °\n"
        )
