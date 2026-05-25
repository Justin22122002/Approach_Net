
class AircraftKinematicsMaxConstrains:
    """
    those are the maximum values for the aircraft kinematics, these are not dependent
    """
    def __init__(
            self,
            cl_max: float,
    ) -> None:
        self.cl_max: float = cl_max
