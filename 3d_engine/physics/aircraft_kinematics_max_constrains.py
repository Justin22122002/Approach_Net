
class AircraftKinematicsMaxConstrains:
    """
    those are the maximum values for the aircraft kinematics, these are not dependent
    """
    def __init__(
            self,
            v_min: float,
            v_max: float,
            roc_max: float,
            rod_max: float,
            gamma_dot_max: float,
            chi_dot_max: float,
    ) -> None:
        self.v_min: float = v_min                   # v_min        minimum speed (stall)            [m/s]
        self.v_max: float = v_max                   # v_max        maximum speed                    [m/s]
        self.roc_max: float = roc_max               # ROC_max      max rate of climb   ḣ ≤  ROC_max [m/s]
        self.rod_max: float = rod_max               # ROD_max      max rate of descent ḣ ≥ -ROD_max [m/s]
        self.gamma_dot_max: float = gamma_dot_max   # γ_max        max |Δγ/Δt|                      [°/s]
        self.chi_dot_max: float = chi_dot_max       # χ_max        max |Δχ/Δt|                      [°/s]