import numpy as np
from numpy.typing import NDArray


class AircraftGoal:
    def __init__(self, x: float, y: float, h: float) -> None:
        self.x: float = x  # East  [m]
        self.y: float = y  # North [m]
        self.h: float = h  # Altitude [m]

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.h])