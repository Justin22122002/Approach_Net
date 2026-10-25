import math
from dataclasses import dataclass, field

from engine.utils.vec3 import Vec3
from engine.utils.color import Color


@dataclass
class DirectionalLight:
    direction: Vec3  = field(default_factory=lambda: Vec3(-1, -1, -1))
    color:     Color = field(default_factory=Color.white)
    intensity: float = 1.0


@dataclass
class PointLight:
    position: Vec3 = field(default_factory=Vec3.zero)
    color: Color = field(default_factory=Color.white)
    intensity: float = 1.0
    a: float = 0.07
    b: float = 0.14


@dataclass
class SpotLight:
    position: Vec3 = field(default_factory=Vec3.zero)
    direction: Vec3 = field(default_factory=Vec3.forward)
    color: Color = field(default_factory=Color.white)
    intensity: float = 1.0
    inner_deg: float = 15.0
    outer_deg: float = 25.0

    @property
    def inner_cos(self) -> float:
        return math.cos(math.radians(self.inner_deg))

    @property
    def outer_cos(self) -> float:
        return math.cos(math.radians(self.outer_deg))