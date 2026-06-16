
class AircraftType:
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
    ) -> None:
        self.cl_max: float = cl_max
        self.mass: float = mass # [kg]
        self.C_D0: float = C_D0 # drag coefficient
        self.area: float = area # wing reference area [m²]
        self.k: float = k

        self.max_thrust: float = max_thrust
        self.max_load_factor: float = max_load_factor
        self.min_load_factor: float = min_load_factor
        self.max_roll_angle: float = max_roll_angle

        # todo: max thrust, min and max load factor, min and max roll angle
