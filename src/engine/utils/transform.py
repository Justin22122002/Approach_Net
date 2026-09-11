import math
from engine.utils.helpers import (
    _mat4_rotate_x,
    _mat4_rotate_y,
    _mat4_rotate_z,
    _mat4_translate,
    _mat4_scale
)
from engine.utils.vec3 import Vec3
import numpy as np


class Transform:
    """Position, rotation (Euler degrees), and scale for a GameObject."""
    def __init__(self) -> None:
        self.position: Vec3 = Vec3(0, 0, 0)
        self.rotation: Vec3 = Vec3(0, 0, 0)   # Euler angles in degrees
        self.scale: Vec3    = Vec3(1, 1, 1)

    def matrix(self) -> np.ndarray:
        rx = math.radians(self.rotation.x)
        ry = math.radians(self.rotation.y)
        rz = math.radians(self.rotation.z)
        T  = _mat4_translate(self.position)
        Rx = _mat4_rotate_x(rx)
        Ry = _mat4_rotate_y(ry)
        Rz = _mat4_rotate_z(rz)
        S  = _mat4_scale(self.scale)
        return T @ Ry @ Rx @ Rz @ S