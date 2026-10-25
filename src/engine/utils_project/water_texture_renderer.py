import math

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

from utils.clients.airport_client import OpenStreetMapAirportClient
from utils.clients.osm_water_client import WaterKind, WaterFeatures, OpenStreetMapWaterClient


class WaterTextureRenderer:

    DEFAULT_COLORS: dict[WaterKind, tuple[int, int, int, int]] = {
        WaterKind.OCEAN: (30, 55, 92, 255),
        WaterKind.WATER: (44, 76, 116, 255),
        WaterKind.RIVER: (52, 86, 126, 255),
        WaterKind.CANAL: (56, 90, 128, 255),
        WaterKind.STREAM: (62, 96, 132, 255),
        WaterKind.TIDAL_CHANNEL: (50, 84, 122, 255),
        WaterKind.DRAIN: (62, 96, 132, 255),
        WaterKind.DITCH: (62, 96, 132, 255),
        WaterKind.WETLAND: (70, 104, 82, 115),
    }

    DEFAULT_WIDTH_M: dict[WaterKind, float] = {
        WaterKind.RIVER: 25.0,
        WaterKind.CANAL: 12.0,
        WaterKind.STREAM: 4.0,
        WaterKind.TIDAL_CHANNEL: 8.0,
        WaterKind.DRAIN: 2.0,
        WaterKind.DITCH: 1.5,
    }

    def __init__(
            self,
            lat_min: float,
            lon_min: float,
            lat_max: float,
            lon_max: float,
            width: int,
            height: int,
            supersample: int = 2,
            colors: dict[WaterKind, tuple[int, int, int, int]] | None = None,
            default_widths_m: dict[WaterKind, float] | None = None,
    ) -> None:
        self.lat_min: float = lat_min
        self.lon_min: float = lon_min
        self.lat_max: float = lat_max
        self.lon_max: float = lon_max

        self.width: int = width
        self.height: int = height
        self.ss: int = max(1, supersample)
        self.w_ss: int = width * self.ss
        self.h_ss: int = height * self.ss

        self.bbox: Polygon = box(lon_min, lat_min, lon_max, lat_max)
        self.colors = {**self.DEFAULT_COLORS, **(colors or {})}
        self.default_widths_m = {**self.DEFAULT_WIDTH_M, **(default_widths_m or {})}

        lat_mid = 0.5 * (lat_min + lat_max)
        m_x = (lon_max - lon_min) * 111_320.0 * math.cos(math.radians(lat_mid))
        m_y = (lat_max - lat_min) * 111_320.0
        self.m_per_px: float = 0.5 * (m_x / self.w_ss + m_y / self.h_ss)

    def composite(
            self,
            features: WaterFeatures,
            base_rgb: np.ndarray | None = None,
            land_color: tuple[int, int, int] = (102, 140, 77),
    ) -> np.ndarray:
        if base_rgb is not None and base_rgb.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"base_rgb shape {base_rgb.shape[:2]} != renderer size {(self.height, self.width)}"
            )

        wetland_layer, water_layer = self._render_layers(features)

        if base_rgb is None:
            base = Image.new("RGBA", (self.width, self.height), (*land_color, 255))
        else:
            base = Image.fromarray(base_rgb.astype(np.uint8)).convert("RGBA")

        out = Image.alpha_composite(base, wetland_layer)
        out = Image.alpha_composite(out, water_layer)

        return np.asarray(out.convert("RGB"))

    def water_mask(self, features: WaterFeatures) -> np.ndarray:
        _, water_layer = self._render_layers(features)
        return np.asarray(water_layer.getchannel("A"), dtype=np.float32) / 255.0


    def _render_layers(self, features: WaterFeatures) -> tuple[Image.Image, Image.Image]:
        wetland = Image.new("RGBA", (self.w_ss, self.h_ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(wetland)
        self._draw_areas(d, features, kinds={WaterKind.WETLAND})

        water = Image.new("RGBA", (self.w_ss, self.h_ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(water)
        for poly in self._ocean_polygons(features.coastlines):
            self._draw_polygon(d, poly, self.colors[WaterKind.OCEAN])
        self._draw_areas(d, features, kinds={WaterKind.WATER})
        self._draw_lines(d, features)

        resample = Image.Resampling.LANCZOS
        return (
            wetland.resize((self.width, self.height), resample),
            water.resize((self.width, self.height), resample),
        )

    def _draw_areas(self, draw: ImageDraw.ImageDraw, features: WaterFeatures, kinds: set[WaterKind]) -> None:
        for area in features.areas:
            if area.kind not in kinds:
                continue
            color = self.colors[area.kind]

            for outer in area.outers:
                for poly in self._clip_ring(outer):
                    self._draw_polygon(draw, poly, color)

            for inner in area.inners:
                for poly in self._clip_ring(inner):
                    self._draw_polygon(draw, poly, (0, 0, 0, 0))

    def _draw_lines(self, draw: ImageDraw.ImageDraw, features: WaterFeatures) -> None:
        for line in features.lines:
            if len(line.coords) < 2:
                continue

            width_m = line.width_m or self.default_widths_m.get(line.kind, 3.0)
            width_px = max(1, round(width_m / self.m_per_px))
            color = self.colors[line.kind]

            clipped = LineString(line.coords).intersection(self.bbox)
            for ls in self._iter_lines(clipped):
                pts = [self._to_px(x, y) for x, y in ls.coords]
                if len(pts) < 2:
                    continue
                draw.line(pts, fill=color, width=width_px, joint="curve")

                if width_px >= 3:
                    r = width_px / 2.0
                    for px, py in (pts[0], pts[-1]):
                        draw.ellipse([px - r, py - r, px + r, py + r], fill=color)

    def _draw_polygon(self, draw: ImageDraw.ImageDraw, poly: Polygon, color: tuple[int, int, int, int]) -> None:
        if poly.is_empty:
            return
        ext = [self._to_px(x, y) for x, y in poly.exterior.coords]
        if len(ext) >= 3:
            draw.polygon(ext, fill=color)
        for ring in poly.interiors:
            pts = [self._to_px(x, y) for x, y in ring.coords]
            if len(pts) >= 3:
                draw.polygon(pts, fill=(0, 0, 0, 0))


    def _ocean_polygons(self, coastlines: list[list[tuple[float, float]]]) -> list[Polygon]:
        lines = [LineString(c) for c in coastlines if len(c) >= 2]
        if not lines:
            return []

        clipped = unary_union(lines).intersection(self.bbox)
        if clipped.is_empty:
            return []

        noded = unary_union([self.bbox.boundary, clipped])
        faces = list(polygonize(noded))

        return [f for f in faces if not f.is_empty and self._is_water_side(f.representative_point(), lines)]

    @staticmethod
    def _is_water_side(pt: Point, coast_lines: list[LineString], eps: float = 1e-5) -> bool:
        best_d, best = math.inf, None
        for ln in coast_lines:
            d = ln.distance(pt)
            if d < best_d:
                best_d, best = d, ln
        if best is None:
            return False

        t = best.project(pt)
        p0 = best.interpolate(max(0.0, t - eps))
        p1 = best.interpolate(min(best.length, t + eps))

        cross = (p1.x - p0.x) * (pt.y - p0.y) - (p1.y - p0.y) * (pt.x - p0.x)
        return cross < 0.0

    def _clip_ring(self, ring: list[tuple[float, float]]):
        if len(ring) < 3:
            return
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        yield from self._iter_polys(poly.intersection(self.bbox))

    def _to_px(self, lon: float, lat: float) -> tuple[float, float]:
        x = (lon - self.lon_min) / (self.lon_max - self.lon_min) * (self.w_ss - 1)
        y = (self.lat_max - lat) / (self.lat_max - self.lat_min) * (self.h_ss - 1)
        return x, y

    @staticmethod
    def _iter_polys(geom: BaseGeometry):
        if isinstance(geom, Polygon):
            yield geom
        elif hasattr(geom, "geoms"):
            for g in geom.geoms:
                yield from WaterTextureRenderer._iter_polys(g)

    @staticmethod
    def _iter_lines(geom: BaseGeometry):
        if isinstance(geom, LineString):
            yield geom
        elif hasattr(geom, "geoms"):
            for g in geom.geoms:
                yield from WaterTextureRenderer._iter_lines(g)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lat_min, lon_min, lat_max, lon_max = 40.55, -73.90, 40.70, -73.70

    features = OpenStreetMapWaterClient.get_water_features(lat_min, lon_min, lat_max, lon_max)
    print(features)

    renderer = WaterTextureRenderer(lat_min, lon_min, lat_max, lon_max, width=1024, height=1024)
    rgb = renderer.composite(features)

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title("OSM Water Overlay (JFK)")
    plt.show()