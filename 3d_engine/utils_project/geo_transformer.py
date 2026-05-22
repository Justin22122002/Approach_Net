from pyproj import Transformer


class GeoTransformer:
    def __init__(
            self,
            lat: float,
            lon: float,
            horizontal_scale: float,
            vertical_scale: float,
    ) -> None:
        self.transformer = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:32632",
            always_xy=True
        )

        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale

        self.origin_x, self.origin_y = self.latlon_to_utm(lat, lon)

    def latlon_to_utm(self, lat: float, lon: float):
        return self.transformer.transform(lon, lat)

    def to_local(self, lat: float, lon: float):
        x, y = self.latlon_to_utm(lat, lon)

        dx = (x - self.origin_x) * self.horizontal_scale
        dz = (self.origin_y - y) * self.horizontal_scale

        return dx, dz

    def altitude_to_local(self, alt_m: float):
        return alt_m * self.vertical_scale