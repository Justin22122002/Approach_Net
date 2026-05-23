from pyproj import Transformer


class GeoTransformer:
    def __init__(
            self,
            lat: float,
            lon: float,
            horizontal_scale: float,
            vertical_scale: float,
            transformer: Transformer,
    ) -> None:
        self.transformer: Transformer = transformer
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale

        self.origin_x, self.origin_y = self.latlon_to_utm(lat, lon)

    def latlon_to_utm(self, lat: float, lon: float) -> tuple[float, float]:
        return self.transformer.transform(lon, lat)

    def to_local(self, lat: float, lon: float) -> tuple[float, float]:
        x, y = self.latlon_to_utm(lat, lon)

        dx = (x - self.origin_x) * self.horizontal_scale
        dz = (self.origin_y - y) * self.horizontal_scale

        return dx, dz

    def to_local_in_m(self, x: float, y: float) -> tuple[float, float]:
        dx = (x - self.origin_x) * self.horizontal_scale
        dz = (self.origin_y - y) * self.horizontal_scale

        return dx, dz

    def altitude_to_local(self, alt_m: float) -> float:
        return alt_m * self.vertical_scale

    @staticmethod
    def utm_epsg(lat: float, lon: float) -> str:
        zone = int((lon + 180) // 6) + 1
        return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"