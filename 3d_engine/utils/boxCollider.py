from dataclasses import dataclass, field
from utils.component import Component
from utils.vec3 import Vec3


@dataclass
class BoxCollider(Component):
    """Axis-aligned bounding box collider in local space."""
    half_extents: Vec3 = field(default_factory=lambda: Vec3(0.5, 0.5, 0.5))

    def world_min(self) -> Vec3:
        p = self.owner.transform.position
        h = self.half_extents
        return Vec3(p.x-h.x, p.y-h.y, p.z-h.z)

    def world_max(self) -> Vec3:
        p = self.owner.transform.position
        h = self.half_extents
        return Vec3(p.x+h.x, p.y+h.y, p.z+h.z)

    def overlaps(self, other: "BoxCollider") -> bool:
        a0, a1 = self.world_min(), self.world_max()
        b0, b1 = other.world_min(), other.world_max()
        return (a0.x <= b1.x and a1.x >= b0.x and
                a0.y <= b1.y and a1.y >= b0.y and
                a0.z <= b1.z and a1.z >= b0.z)