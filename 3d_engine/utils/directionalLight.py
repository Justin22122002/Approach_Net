from dataclasses import dataclass, field

from utils.vec3 import Vec3
from utils.color import Color


@dataclass
class DirectionalLight:
    direction: Vec3  = field(default_factory=lambda: Vec3(-1,-1,-1))
    color:     Color = field(default_factory=Color.white)
    intensity: float = 1.0