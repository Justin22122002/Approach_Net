from typing import override

import numpy as np
from pyproj import Transformer

from utils.approch import Approach
from utils.color import Color
from utils.gameObject import GameObject
from utils.lineRenderer import LineRenderer
from utils.mesh import Mesh
from utils.scene import Scene
from utils.vec3 import Vec3
from utils_project.geo_transformer import GeoTransformer
from utils_project.rnav_lpv_funnel_geometry import LpvFunnelGeometry
from utils_project.project_mesh import ProjectMesh


def _to_vec3(geo: GeoTransformer, x: float, y: float, h: float) -> Vec3:
    lx, lz = geo.to_local_in_m(x, y)
    return Vec3(lx, geo.altitude_to_local(h), lz)


class RnavLpvFunnelProjectMesh(ProjectMesh):
    def __init__(
        self,
        approach: Approach,
        transformer: Transformer,
        horizontal_scale: float,
        vertical_scale: float,
        glideslope_deg: float = 3.0,
        n_rings: int = 12,
        rail_color: Color = Color(1.0, 0.4, 0.1),
        ring_color: Color = Color(1.0, 0.6, 0.3),
    ) -> None:
        self.approach: Approach = approach
        self.glideslope_deg: float = glideslope_deg
        self.n_rings: float = n_rings
        self.rail_color: Color = rail_color
        self.ring_color: Color = ring_color

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=0.0,
            lon=0.0,
            altitude=0.0,
            transformer=transformer,
        )

    @override
    def spawn(self, scene: Scene, geo: GeoTransformer) -> list[GameObject]:
        _entry_fixes, faf_fix, rw_fix = self.approach.get_final_approach_fixes()
        if faf_fix is None or rw_fix is None:
            print(f"[funnel] WARN: no FAF/RW-Fix for '{self.approach.procedure_identifier}' - skip.")
            return []

        faf_x, faf_y = self.transformer.transform(faf_fix.longitude, faf_fix.latitude)
        rw_x, rw_y = self.transformer.transform(rw_fix.longitude, rw_fix.latitude)
        rw_h = rw_fix.altitude * 0.3048  # inkl. TCH

        funnel = LpvFunnelGeometry(
            thr_xy=(rw_x, rw_y), thr_h=rw_h, faf_xy=(faf_x, faf_y),
            glideslope_deg=self.glideslope_deg,
        )

        s_vals = [funnel.s_max * i / 39 for i in range(40)]
        rails = [[_to_vec3(geo, *funnel.boundary_point(s, sy, sz)) for s in s_vals]
                 for sy, sz in [(-1, 1), (1, 1), (-1, -1), (1, -1)]]

        ring_s = [funnel.s_max * i / (self.n_rings - 1) for i in range(self.n_rings)]
        rings = [[_to_vec3(geo, *p) for p in funnel.ring(s)] for s in ring_s]

        objects: list[GameObject] = []
        for i, rail in enumerate(rails):
            go = scene.spawn(f"RnavLpvFunnelRail_{i}")
            go.add_component(LineRenderer.strip(rail, color=self.rail_color))
            objects.append(go)
        for i, ring in enumerate(rings):
            go = scene.spawn(f"RnavLpvFunnelRing_{i}")
            go.add_component(LineRenderer.strip(ring, color=self.ring_color))
            objects.append(go)

        print(f"[funnel] {self.approach.procedure_identifier}: FAF={faf_fix.fix_identifier} "
              f"-> RW={rw_fix.fix_identifier}  s_max={funnel.s_max:.0f}m")
        return objects

    @override
    def generate_mesh(self) -> Mesh:
        return Mesh(np.empty(0, dtype=np.float32), np.empty(0, dtype=np.uint32))