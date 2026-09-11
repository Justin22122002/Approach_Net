import math

NM_TO_M = 1852.0
FT_TO_M = 0.3048


class SeparationGeometry:
    """
    Hard separation minima (FAA JO 7110.65BB).
      - 4-5-1: vertical separation 1000 ft below FL290
      - 5-5-4: radar separation 2.5 NM on final approach within 10 NM of the runway (behind each aircraft)
      - 5-9-6: diagonal minimum between successive aircraft on adjacent parallel (next to each aircraft)
         final approach courses, as a function of runway centerline spacing:
           2500-3600 ft -> 1.0 NM
           3600-8300 ft -> 1.5 NM
           8300-9000 ft -> 2.0 NM

    Superellipsoid:  ((d_h/D_h)^2)^(n/2) + ((d_v/D_v)^2)^(n/2) >= 1
        n=2      -> plain ellipsoid
        n -> inf -> exact cylinder (horizontal OR vertical minimum)

    contain_cylinder=True inflates both semi-axes by k=2^(1/n) so the
    ellipsoid fully contains the FAA cylinder (radius D_h, half-height D_v).
    Derivation: the cylinder corner (D_h, D_v) is the binding point -
    plugging the scaled axes (k*D_h, k*D_v) into the constraint at that
    point gives 2/k^n = 1, hence k = 2^(1/n). All other cylinder points
    then lie strictly inside. contain_cylinder=False (default) uses the
    raw FAA minima as ellipsoid semi-axes directly (undersized vs. the
    exact cylinder at the corners, see the widget from earlier).

    Box:        cylinder of radius D_h and half-height D_v (both minima checked
                independently, as ICAO/FAA define them) - always uses the raw
                minima, unaffected by contain_cylinder.
    """

    def __init__(
        self,
        d_horizontal_m: float = 2.5 * NM_TO_M,
        d_vertical_m: float = 1000.0 * FT_TO_M,
        n: float = 4.0, # 4.0, 2.0, 8.0
        contain_cylinder: bool = True,
    ) -> None:
        self.d_h = d_horizontal_m
        self.d_v = d_vertical_m
        self.n = n
        self.contain_cylinder = contain_cylinder

        k = 2.0 ** (1.0 / n) if contain_cylinder else 1.0
        self.d_h_eff = k * d_horizontal_m
        self.d_v_eff = k * d_vertical_m

    @staticmethod
    def _lame(val: float, exp: float) -> float:
        """|val|^exp with val's sign restored (Lame/superellipse helper)."""
        return math.copysign(abs(val) ** exp, val) if val != 0 else 0.0

    def normalized_radius(self, dx: float, dy: float, dz: float) -> float:
        u = ((dx * dx + dy * dy) / self.d_h_eff ** 2) ** (self.n / 2)
        v = (dz * dz / self.d_v_eff ** 2) ** (self.n / 2)
        return (u + v) ** (1.0 / self.n)

    def ellipsoid_constraint(self, dx: float, dy: float, dz: float) -> float:
        """>= 0 when separated. Smooth, differentiable (for IPOPT/FATROP). n=2 -> plain ellipsoid."""
        u = ((dx * dx + dy * dy) / self.d_h_eff ** 2) ** (self.n / 2)
        v = (dz * dz / self.d_v_eff ** 2) ** (self.n / 2)
        return u + v - 1.0

    def is_separated_cylindrical(self, dx: float, dy: float, dz: float) -> bool:
        """Classical rule: horizontal OR vertical minimum satisfied (non-smooth). Uses raw minima."""
        return math.hypot(dx, dy) >= self.d_h or abs(dz) >= self.d_v

    def ellipsoid_rings(
        self, center: tuple[float, float, float],
        n_points: int = 48, n_slices: int = 5,
    ) -> list[list[tuple[float, float, float]]]:
        cx, cy, ch = center
        a, c, n = self.d_h_eff, self.d_v_eff, self.n
        exp = 2.0 / n
        rings = []

        for i in range(1, n_slices + 1):
            phi = -math.pi / 2 + math.pi * i / (n_slices + 1)
            r_xy = a * self._lame(math.cos(phi), exp)
            z = c * self._lame(math.sin(phi), exp)
            rings.append([
                (cx + r_xy * math.cos(2 * math.pi * j / n_points),
                 cy + r_xy * math.sin(2 * math.pi * j / n_points),
                 ch + z)
                for j in range(n_points + 1)
            ])

        for theta in (0.0, math.pi / 2):
            rings.append([
                (cx + a * self._lame(math.cos(2 * math.pi * j / n_points), exp) * math.cos(theta),
                 cy + a * self._lame(math.cos(2 * math.pi * j / n_points), exp) * math.sin(theta),
                 ch + c * self._lame(math.sin(2 * math.pi * j / n_points), exp))
                for j in range(n_points + 1)
            ])
        return rings

    def cylinder_edges(
        self, center: tuple[float, float, float], n_points: int = 48,
    ) -> list[list[tuple[float, float, float]]]:
        """Radar minimum (radius D_h) x vertical minimum (+-D_v), checked independently. Raw minima."""
        cx, cy, ch = center
        a, c = self.d_h, self.d_v

        def circle(z: float) -> list[tuple[float, float, float]]:
            return [
                (cx + a * math.cos(2 * math.pi * j / n_points),
                 cy + a * math.sin(2 * math.pi * j / n_points),
                 z)
                for j in range(n_points + 1)
            ]

        edges = [circle(ch + c), circle(ch - c), circle(ch)]
        for j in range(4):
            ang = 2 * math.pi * j / 4
            px, py = cx + a * math.cos(ang), cy + a * math.sin(ang)
            edges.append([(px, py, ch - c), (px, py, ch + c)])
        return edges