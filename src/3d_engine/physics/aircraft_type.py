
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
            k: float  = 0.2,
    ) -> None:
        self.cl_max: float = cl_max
        self.mass: float = mass # [kg]
        self.C_D0: float = C_D0 # drag coefficient
        self.area: float = area # wing reference area [m²]
        self.k: float = k
