from typing import override

import numpy as np
from pyproj import Transformer

from utils.approch import Approach
from utils.arinc_runway import ArincRunway

from engine.utils.color import Color
from engine.utils.gameObject import GameObject
from engine.utils.lineRenderer import LineRenderer
from engine.utils.mesh import Mesh
from engine.utils.meshRenderer import MeshRenderer
from engine.utils.scene import Scene
from engine.utils.simplematerial import SimpleMaterial
from engine.utils.vec3 import Vec3
from engine.utils_project.geo_transformer import GeoTransformer
from engine.utils_project.rnav_lpv_funnel_geometry import LpvFunnelGeometry
from engine.utils_project.project_mesh import ProjectMesh


class RnavLpvFunnelProjectMesh(ProjectMesh):
    CORNERS: list[tuple[int, int]] = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    FACE_NAMES: dict[str, int] = {"bottom": 0, "right": 1, "top": 2, "left": 3}

    def __init__(
        self,
        approach: Approach,
        transformer: Transformer,
        horizontal_scale: float,
        vertical_scale: float,
        runway: ArincRunway,
        glideslope_deg: float = 3.0,
        n_rings: int = 12,
        rail_color: Color = Color(1.0, 0.4, 0.1),
        ring_color: Color = Color(1.0, 0.6, 0.3),
        fill: bool = True,
        fill_color: Color = Color(1.0, 0.5, 0.1),
        fill_opacity: float = 0.25,
        fill_faces: tuple[str, ...] = ("bottom", "right", "top", "left"),
        n_fill_segments: int = 40,
    ) -> None:
        self.approach: Approach = approach
        self.glideslope_deg: float = glideslope_deg
        self.n_rings: int = n_rings
        self.rail_color: Color = rail_color
        self.ring_color: Color = ring_color
        self.runway: ArincRunway = runway

        self.fill: bool = fill
        self.fill_color: Color = fill_color
        self.fill_opacity: float = fill_opacity
        self.fill_faces: tuple[str, ...] = fill_faces
        self.n_fill_segments: int = n_fill_segments

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=0.0,
            lon=0.0,
            altitude=0.0,
            transformer=transformer,
        )

    @staticmethod
    def _to_vec3(geo: GeoTransformer, x: float, y: float, h: float) -> Vec3:
        lx, lz = geo.to_local_in_m(x, y)
        return Vec3(lx, geo.altitude_to_local(h), lz)

    @staticmethod
    def _vertex(p: Vec3, n: np.ndarray) -> list[float]:
        """interleaved vertex: position (3), normal (3), uv (2)"""
        return [p.x, p.y, p.z, float(n[0]), float(n[1]), float(n[2]), 0.0, 0.0]

    @staticmethod
    def _tri_normal(a: Vec3, b: Vec3, c: Vec3) -> np.ndarray:
        u = np.array([b.x - a.x, b.y - a.y, b.z - a.z])
        v = np.array([c.x - a.x, c.y - a.y, c.z - a.z])
        n = np.cross(u, v)
        length = np.linalg.norm(n)
        return n / length if length > 1e-12 else np.array([0.0, 1.0, 0.0])

    def _fill_mesh(self, funnel: LpvFunnelGeometry, geo: GeoTransformer) -> Mesh:
        """transparent hull: quads between consecutive cross-sections, both windings (no backface culling issues)"""
        s_vals = [funnel.s_max * i / self.n_fill_segments for i in range(self.n_fill_segments + 1)]
        sections = [[self._to_vec3(geo, *funnel.boundary_point(s, sy, sz)) for sy, sz in self.CORNERS]
                    for s in s_vals]
        faces = [self.FACE_NAMES[f] for f in self.fill_faces]

        verts: list[float] = []
        for k in range(len(sections) - 1):
            a_sec, b_sec = sections[k], sections[k + 1]
            for f in faces:
                c0, c1 = f, (f + 1) % len(self.CORNERS)
                quad = (a_sec[c0], a_sec[c1], b_sec[c1], b_sec[c0])
                for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                    n = self._tri_normal(*tri)
                    for p in tri:                     # front side
                        verts += self._vertex(p, n)
                    for p in reversed(tri):           # back side
                        verts += self._vertex(p, -n)

        vertices = np.asarray(verts, dtype=np.float32)
        indices = np.arange(len(verts) // 8, dtype=np.uint32)
        return Mesh(vertices, indices)

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
            runway_length_m=self.runway.length * 0.3048
        )

        s_vals = [funnel.s_max * i / 39 for i in range(40)]
        rails = [[self._to_vec3(geo, *funnel.boundary_point(s, sy, sz)) for s in s_vals]
                 for sy, sz in [(-1, 1), (1, 1), (-1, -1), (1, -1)]]

        ring_s = [funnel.s_max * i / (self.n_rings - 1) for i in range(self.n_rings)]
        rings = [[self._to_vec3(geo, *p) for p in funnel.ring(s)] for s in ring_s]

        objects: list[GameObject] = []

        if self.fill:
            go = scene.spawn("RnavLpvFunnelFill")
            material = SimpleMaterial(color=self.fill_color, opacity=self.fill_opacity)
            go.add_component(MeshRenderer(self._fill_mesh(funnel, geo), material=material, cast_shadow=False))
            objects.append(go)

        for i, rail in enumerate(rails):
            go = scene.spawn(f"RnavLpvFunnelRail_{i}")
            go.add_component(LineRenderer.strip(rail, color=self.rail_color))
            objects.append(go)
        for i, ring in enumerate(rings):
            go = scene.spawn(f"RnavLpvFunnelRing_{i}")
            go.add_component(LineRenderer.strip(ring, color=self.ring_color))
            objects.append(go)

        print(f"[funnel] {self.approach.procedure_identifier}: FAF={faf_fix.fix_identifier} "
              f"-> RW={rw_fix.fix_identifier}  s_max={funnel.s_max:.0f}ft")
        return objects

    @override
    def generate_mesh(self) -> Mesh:
        return Mesh(np.empty(0, dtype=np.float32), np.empty(0, dtype=np.uint32))