import re
from dataclasses import dataclass, field
from enum import Enum

import requests
from shapely.geometry import LineString
from shapely.ops import linemerge


class WaterKind(str, Enum):
    OCEAN = "ocean"
    WATER = "water"
    WETLAND = "wetland"
    RIVER = "river"
    CANAL = "canal"
    STREAM = "stream"
    TIDAL_CHANNEL = "tidal_channel"
    DRAIN = "drain"
    DITCH = "ditch"


_LINEAR_WATERWAYS: dict[str, WaterKind] = {
    "river": WaterKind.RIVER,
    "canal": WaterKind.CANAL,
    "stream": WaterKind.STREAM,
    "tidal_channel": WaterKind.TIDAL_CHANNEL,
    "drain": WaterKind.DRAIN,
    "ditch": WaterKind.DITCH,
}


@dataclass
class WaterArea:
    outers: list[list[tuple[float, float]]]        # [(lon, lat), ...]
    inners: list[list[tuple[float, float]]] = field(default_factory=list)
    kind: WaterKind = WaterKind.WATER


@dataclass
class WaterLine:
    coords: list[tuple[float, float]]              # [(lon, lat), ...]
    kind: WaterKind = WaterKind.RIVER
    width_m: float | None = None


@dataclass
class WaterFeatures:
    areas: list[WaterArea] = field(default_factory=list)
    lines: list[WaterLine] = field(default_factory=list)
    coastlines: list[list[tuple[float, float]]] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"WaterFeatures(areas={len(self.areas)}, lines={len(self.lines)}, "
                f"coastlines={len(self.coastlines)})")


class OpenStreetMapWaterClient:

    OVERPASS_URL: str = "https://overpass-api.de/api/interpreter"
    OVERPASS_HEADERS: dict[str, str] = {
        "User-Agent": "MasterThesisTerrainLoader/1.0 (justin.groh@study.thws.de)",
        "Accept": "*/*",
    }

    def get_water_features(
            self,
            lat_min: float,
            lon_min: float,
            lat_max: float,
            lon_max: float,
    ) -> WaterFeatures:
        query = self._build_query(lat_min, lon_min, lat_max, lon_max)

        r = requests.post(
            self.OVERPASS_URL,
            data={"data": query},
            headers=self.OVERPASS_HEADERS,
            timeout=300,
        )
        r.raise_for_status()

        return self._parse(r.json())

    @staticmethod
    def _build_query(lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> str:
        b = f"{lat_min},{lon_min},{lat_max},{lon_max}"
        return f"""
        [out:json][timeout:180];
        (
          way["natural"="water"]({b});
          relation["natural"="water"]({b});

          way["waterway"~"^(riverbank|dock)$"]({b});
          relation["waterway"="riverbank"]({b});

          way["landuse"~"^(reservoir|basin)$"]({b});
          relation["landuse"~"^(reservoir|basin)$"]({b});

          way["natural"="wetland"]({b});
          relation["natural"="wetland"]({b});

          way["waterway"~"^(river|stream|canal|drain|ditch|tidal_channel)$"]({b});

          way["natural"="coastline"]({b});
        );
        out geom;
        """

    def _parse(self, data: dict) -> WaterFeatures:
        features = WaterFeatures()

        for el in data.get("elements", []):
            tags: dict = el.get("tags", {}) or {}

            if el.get("type") == "way":
                coords = [(g["lon"], g["lat"]) for g in el.get("geometry", [])]
                if len(coords) < 2:
                    continue

                if tags.get("natural") == "coastline":
                    features.coastlines.append(coords)
                    continue

                area_kind = self._area_kind(tags)
                if area_kind is not None:
                    features.areas.append(WaterArea(outers=[coords], kind=area_kind))
                    continue

                line_kind = _LINEAR_WATERWAYS.get(tags.get("waterway", ""))
                if line_kind is not None:
                    features.lines.append(
                        WaterLine(coords=coords, kind=line_kind, width_m=self._parse_width(tags))
                    )

            elif el.get("type") == "relation":
                area_kind = self._area_kind(tags)
                if area_kind is None:
                    continue

                outer_geoms: list[list[tuple[float, float]]] = []
                inner_geoms: list[list[tuple[float, float]]] = []

                for m in el.get("members", []):
                    geom = [(g["lon"], g["lat"]) for g in m.get("geometry", []) or []]
                    if len(geom) < 2:
                        continue
                    if m.get("role") == "inner":
                        inner_geoms.append(geom)
                    else:
                        outer_geoms.append(geom)

                outers = self._assemble_rings(outer_geoms)
                inners = self._assemble_rings(inner_geoms)

                if outers:
                    features.areas.append(WaterArea(outers=outers, inners=inners, kind=area_kind))

        return features

    @staticmethod
    def _area_kind(tags: dict) -> WaterKind | None:
        if tags.get("natural") == "water":
            return WaterKind.WATER
        if tags.get("waterway") in ("riverbank", "dock"):
            return WaterKind.WATER
        if tags.get("landuse") in ("reservoir", "basin"):
            return WaterKind.WATER
        if tags.get("natural") == "wetland":
            return WaterKind.WETLAND
        return None

    @staticmethod
    def _assemble_rings(geoms: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
        lines = [LineString(g) for g in geoms if len(g) >= 2]
        if not lines:
            return []

        merged = linemerge(lines) if len(lines) > 1 else lines[0]
        parts = [merged] if isinstance(merged, LineString) else list(merged.geoms)

        return [list(p.coords) for p in parts if len(p.coords) >= 3]

    @staticmethod
    def _parse_width(tags: dict) -> float | None:
        raw = tags.get("width")
        if not raw:
            return None
        m = re.search(r"(\d+(?:[.,]\d+)?)", str(raw))
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", "."))
        except ValueError:
            return None