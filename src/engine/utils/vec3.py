import math

import numpy as np
from numpy.typing import NDArray


class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(
            self,
            x: float = 0.0,
            y: float = 0.0,
            z: float = 0.0,
    ) -> None:
        self.x: float = float(x)
        self.y: float = float(y)
        self.z: float = float(z)

    def __add__(self, o: 'Vec3') -> 'Vec3': return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)
    def __sub__(self, o: 'Vec3') -> 'Vec3': return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)
    def __mul__(self, s: float) -> 'Vec3':  return Vec3(self.x * s,   self.y * s,   self.z * s)
    def __rmul__(self, s: float) -> 'Vec3': return self.__mul__(s)
    def __neg__(self) -> 'Vec3': return Vec3(-self.x, -self.y, -self.z)
    def __repr__(self) -> str: return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def dot(self, o: 'Vec3') -> float: return self.x * o.x + self.y * o.y + self.z * o.z
    def length(self) -> float: return math.sqrt(self.dot(self))
    def normalized(self) -> 'Vec3':
        l = self.length()
        return Vec3(self.x / l, self.y / l, self.z / l) if l > 1e-9 else Vec3(0,0,0)

    def cross(self, o) -> 'Vec3':
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x
        )

    def to_list(self) -> list[float]: return [self.x, self.y, self.z]
    def to_np(self) -> NDArray[np.float32]: return np.array([self.x, self.y, self.z], dtype=np.float32)

    @staticmethod
    def zero() -> 'Vec3': return Vec3(0,0,0)
    @staticmethod
    def one() -> 'Vec3': return Vec3(1,1,1)
    @staticmethod
    def up() -> 'Vec3': return Vec3(0,1,0)
    @staticmethod
    def forward() -> 'Vec3': return Vec3(0,0,-1)
    @staticmethod
    def right() -> 'Vec3': return Vec3(1,0,0)
