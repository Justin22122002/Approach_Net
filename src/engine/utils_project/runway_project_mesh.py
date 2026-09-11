from typing import override

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer, Geod

from utils.arinc_runway import ArincRunway

from engine.utils.gameObject import GameObject
from engine.utils.simplematerial import SimpleMaterial
from engine.utils.mesh import Mesh
from engine.utils.meshRenderer import MeshRenderer
from engine.utils.scene import Scene
from engine.utils.texture import Texture
from engine.utils.vec3 import Vec3
from engine.utils_project.geo_transformer import GeoTransformer
from engine.utils_project.project_mesh import ProjectMesh


class RunwayProjectMesh(ProjectMesh):
    def __init__(
        self,
        runway: ArincRunway,
        horizontal_scale: float,
        vertical_scale: float,
        transformer: Transformer,
        px_per_m: float = 4.0,
        asphalt_color: tuple[int, int, int] = (179, 179, 179),
        marking_color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self.geod = Geod(ellps="WGS84")
        self.runway: ArincRunway = runway
        self.length_m: float = runway.length * self.FT_TO_M
        self.width_m: float = runway.width * self.FT_TO_M
        self.displaced_threshold_m: float = runway.displaced_threshold * self.FT_TO_M
        self.bearing: float = runway.bearing

        self._texture: Texture
        self.px_per_m: float = px_per_m
        self.asphalt_color: tuple[int, int, int] = asphalt_color
        self.marking_color: tuple[int, int, int] = marking_color

        super().__init__(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            lat=runway.latitude,
            lon=runway.longitude,
            altitude=runway.runway_altitude * self.FT_TO_M,
            transformer=transformer
        )

    def _compute_endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        start_lat = self.lat
        start_lon = self.lon

        start_lon, start_lat, _ = self.geod.fwd(
            start_lon,
            start_lat,
            self.bearing,
            self.displaced_threshold_m
        )

        end_lon, end_lat, _ = self.geod.fwd(
            start_lon,
            start_lat,
            self.bearing,
            self.length_m
        )

        return (start_lat, start_lon), (end_lat, end_lon)

    @staticmethod
    def _reciprocal_identifier(identifier: str) -> str:
        import re
        m = re.match(r"^(\d{1,2})([LRC]?)$", identifier.upper())
        if not m:
            return identifier
        n, suf = int(m.group(1)), m.group(2)
        n_recip = n + 18 if n <= 18 else n - 18
        suf_recip = {"L": "R", "R": "L", "C": "C", "": ""}[suf]
        return f"{n_recip:02d}{suf_recip}"

    def _render_runway_texture_rgb(self) -> np.ndarray:
        identifier = self.runway.runway_identifier
        reciprocal = self._reciprocal_identifier(identifier)
        length_run_m = self.length_m - self.displaced_threshold_m

        w_px = max(16, int(round(self.width_m * self.px_per_m)))
        h_px = max(16, int(round(length_run_m * self.px_per_m)))

        img = Image.new("RGB", (w_px, h_px), self.asphalt_color)
        draw = ImageDraw.Draw(img)

        dash_len_px = int(round(15 * self.px_per_m))
        gap_px = int(round(10 * self.px_per_m))
        dash_w_px = max(1, int(round(0.9 * self.px_per_m)))
        cx = w_px // 2
        y = int(round(30 * self.px_per_m))
        while y < h_px - int(round(30 * self.px_per_m)):
            draw.rectangle(
                [cx - dash_w_px // 2, y, cx + dash_w_px // 2, min(h_px, y + dash_len_px)],
                fill=self.marking_color,
            )
            y += dash_len_px + gap_px

        font_size = 80
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default(size=font_size)

        def draw_label(text: str, v_center_px: int, upside_down: bool) -> None:
            label_img = Image.new("RGBA", (w_px, font_size * 2), (0, 0, 0, 0))
            label_draw = ImageDraw.Draw(label_img)
            bbox = label_draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            label_draw.text(
                (w_px / 2 - tw / 2 - bbox[0], font_size - th / 2 - bbox[1]),
                text, fill=self.marking_color, font=font,
            )
            if upside_down:
                label_img = label_img.rotate(180)
            img.paste(label_img, (0, v_center_px - font_size), label_img)

        margin_px = int(round(25 * self.px_per_m))
        draw_label(identifier, margin_px, upside_down=True)
        draw_label(reciprocal, h_px - margin_px, upside_down=False)

        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        return np.array(img, dtype=np.uint8)

    def _build_texture(self) -> Texture:
        return Texture.from_array(self._render_runway_texture_rgb())

    @override
    def spawn(self, scene: Scene, geo: GeoTransformer) -> list[GameObject]:
        cx, cz = geo.to_local(self.lat, self.lon)
        cy = geo.altitude_to_local(self.altitude)

        go = scene.spawn("Runway")
        go.transform.position = Vec3(cx, cy, cz)
        go.add_component(MeshRenderer(self.generate_mesh(), SimpleMaterial(texture=self._build_texture())))
        return [go]

    @override
    def generate_mesh(self) -> Mesh:
        (lat0, lon0), (lat1, lon1) = self._compute_endpoints()

        az, _, _ = self.geod.inv(lon0, lat0, lon1, lat1)

        half_width = self.width_m / 2.0

        left_az = az - 90
        right_az = az + 90

        p0_lon, p0_lat, _ = self.geod.fwd(lon0, lat0, left_az, half_width)
        p1_lon, p1_lat, _ = self.geod.fwd(lon0, lat0, right_az, half_width)

        p2_lon, p2_lat, _ = self.geod.fwd(lon1, lat1, right_az, half_width)
        p3_lon, p3_lat, _ = self.geod.fwd(lon1, lat1, left_az, half_width)

        def proj(lat, lon):
            x, y = self.transformer.transform(lon, lat)

            x0, y0 = self.transformer.transform(self.lon, self.lat)

            dx = (x - x0) * self.horizontal_scale
            dz = (y - y0) * self.horizontal_scale

            return Vec3(dx, 0.0, -dz)

        v0 = proj(p0_lat, p0_lon)
        v1 = proj(p1_lat, p1_lon)
        v2 = proj(p2_lat, p2_lon)
        v3 = proj(p3_lat, p3_lon)

        verts = []
        idx = [0, 1, 2, 0, 2, 3]

        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

        for v, (u, vv) in zip([v0, v1, v2, v3], uvs):
            verts += [
                v.x, 0.0, v.z,
                0.0, 1.0, 0.0,
                u, vv
            ]

        return Mesh(
            np.array(verts, dtype=np.float32),
            np.array(idx, dtype=np.uint32)
        )
