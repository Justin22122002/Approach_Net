import math

"""
## X. RNAV (GPS) LPV Final Approach Funnel

Geometry of the LPV final approach protection volume between the Final Approach Fix (PFAF)
and the Landing Threshold Point (LTP), as defined by
FAA Order 8260.58D, *United States Standard for Performance Based Navigation (PBN) -> TP 308/GPH 209 (Transport Canada)
Instrument Procedure Design*, and RTCA DO-229 (WAAS MOPS).

### X.1 Reference Frame

Let $s$ be the along-track distance from the LTP towards the PFAF:

$$
s = 0 \;\Rightarrow\; \text{LTP (threshold)}, \qquad
s = s_{max} \;\Rightarrow\; \text{PFAF}
$$

$$
s_{max} = \left\lVert \mathbf{p}_{PFAF} - \mathbf{p}_{LTP} \right\rVert, \qquad
\hat{a} = \frac{\mathbf{p}_{PFAF} - \mathbf{p}_{LTP}}{s_{max}}, \qquad
\hat{l} = \begin{bmatrix} -\hat{a}_y \\ \hat{a}_x \end{bmatrix}
$$

with $\hat{a}$ the along-track and $\hat{l}$ the lateral unit vector.

### X.2 Vertical Centerline (Glidepath)

*Order 8260.58D, Ch. 1, Formula 1-3-4 "Altitude on Baro Glidepath", p. 1-29*

$$
Z_{Baro} = e^{\frac{d \cdot \tan(\theta)}{r}} \cdot (r + alt_b) - r
$$

where $r = 20\,890\,537\;\text{ft}$ is the mean earth radius
(*Order 8260.58D, para. 1-2-2, p. 1-2*), $\theta$ the glidepath angle (GPA) and
$alt_b = \text{LTP}_{elev} + \text{TCH}$.

For final segment lengths of a few NM the spherical term is negligible, so the
implementation uses the flat-earth form:

$$
h_c(s) = h_{LTP} + \text{TCH} + s \cdot \tan(\theta)
$$

**Threshold Crossing Height (TCH)**
*Order 8260.58D, para. 1-3-1.f(2)(b), p. 1-27 and Table 1-3-1, p. 1-28*

$$
\text{WCH} = \text{TCH} - h_{glidepath \rightarrow wheel}, \qquad
20\;\text{ft} \le \text{WCH} \le 50\;\text{ft}
$$

| Height Group | Representative Aircraft | Recommended TCH |
|---|---|---|
| 1 | General aviation, corporate jets | 40 ft |
| 2 | B737, DC-9 | 45 ft |
| 3 | B757, C-17 | 50 ft |
| 4 | B747/777, A300 | 55 ft |

Maximum GPA is $3.5°$ for procedures with HAT below 250 ft; maximum TCH is 60 ft
(*Order 8260.58D, para. 3-4-1.a, p. 3-23*).

### X.3 Lateral Half Width (FAS Course Width)

The lateral guidance sector originates at the **GARP**, not at the threshold.

**Flight Path Alignment Point (FPAP)**
*Order 8260.58D, para. 3-1-5.c(1), p. 3-7*

$$
d_{FPAP} = \max\left(L_{runway},\; 9023\;\text{ft}\right)
$$

**GNSS Azimuth Reference Point (GARP)**
*Order 8260.58D, para. 3-1-5.c(3)(a), p. 3-7*

$$
d_{GARP} = d_{FPAP} + 1000\;\text{ft}
$$

**Course Width at Threshold**
*Order 8260.58D, Formula 3-1-1, p. 3-8*

$$
w_{thr} = \max\left(350\;\text{ft},\; \tan(1.5°) \cdot d_{GARP}\right)
$$

**Lateral Guidance Sector Angle (LGSA)**
*Order 8260.58D, para. 3-1-5.c(3)(b), p. 3-7*

The LGSA is bounded by $1.5° \le \text{LGSA} \le 2.0°$ and follows from the
threshold course width:

$$
\text{LGSA}_{eff} = \arctan\left(\frac{w_{thr}}{d_{GARP}}\right)
$$

Since the sector is angular about the GARP, the half width at any $s$ is:

$$
w_{lat}(s) = \tan\left(\text{LGSA}_{eff}\right) \cdot \left(s + d_{GARP}\right)
$$

> **Note:** the $350$ ft floor is a one-sided value ($\pm 350$ ft full-scale
> deflection), identical in concept to the ILS localizer. For the minimum
> $d_{GARP} = 10023$ ft the floor yields exactly $\text{LGSA}_{eff} = 2.0°$;
> for runways longer than $\approx 12\,365$ ft the angular term dominates and
> $\text{LGSA}_{eff} \rightarrow 1.5°$.

### X.4 Vertical Half Width (Full Scale Deflection)

*RTCA DO-229 (WAAS MOPS) — angular, referenced to the FPCP above the LTP at TCH* !!! ICAO Annex 10

$$
w_{vert}(s) = \mathrm{clamp}\left(\tan\left(0.25 \cdot \theta\right) \cdot s,\;
45\;\text{m},\; 150\;\text{m}\right)
$$

Unlike the lateral sector, the vertical deflection does **not** narrow all the way
to the threshold: it is clamped at $\pm 45$ m (reached at $\approx 1.7$ NM for a
$3°$ glidepath) and at $\pm 150$ m at the far end ($\approx 6$ NM).

### X.5 Protection Volume

The funnel boundary for signs $\sigma_y, \sigma_z \in \{-1, +1\}$:

$$
\mathbf{b}(s, \sigma_y, \sigma_z) =
\begin{bmatrix}
x_{LTP} + \hat{a}_x s + \hat{l}_x \, \sigma_y w_{lat}(s) \\
y_{LTP} + \hat{a}_y s + \hat{l}_y \, \sigma_y w_{lat}(s) \\
h_c(s) + \sigma_z w_{vert}(s)
\end{bmatrix}
$$

with an additional implementation-side floor (not a TERPS requirement) preventing
the lower boundary from dropping below runway level:

$$
\sigma_z = -1 \;\Rightarrow\;
\sigma_z w_{vert}(s) \leftarrow \max\left(-w_{vert}(s),\; h_{LTP} - m_{ground} - h_c(s)\right)
$$

### X.6 Reference Summary

| Quantity | Source | Location |
|---|---|---|
| Glidepath altitude | Order 8260.58D, Formula 1-3-4 | p. 1-29 |
| TCH / WCH | Order 8260.58D, para. 1-3-1.f(2)(b), Table 1-3-1 | p. 1-27 / 1-28 |
| FPAP location | Order 8260.58D, para. 3-1-5.c(1) | p. 3-7 |
| GARP location | Order 8260.58D, para. 3-1-5.c(3)(a) | p. 3-7 |
| LGSA range | Order 8260.58D, para. 3-1-5.c(3)(b) | p. 3-7 |
| Course width at threshold | Order 8260.58D, Formula 3-1-1 | p. 3-8 |
| Max GPA / TCH limits | Order 8260.58D, para. 3-4-1.a | p. 3-23 |
| Vertical FSD, clamps | RTCA DO-229 (WAAS MOPS) | — |

### X.7 Implementation
"""


class LpvFunnelGeometry:
    """
    RNAV (GPS) LPV final approach funnel between PFAF and LTP.
    Geometry per FAA Order 8260.58D (FAS data block) and RTCA DO-229 (FSD).
    """

    def __init__(
        self,
        thr_xy: tuple[float, float],
        thr_h: float,
        faf_xy: tuple[float, float],
        glideslope_deg: float = 3.0,
        thr_half_width_lat_m: float = 106.68,   # 350 ft floor, Formula 3-1-1
        fsd_v_ratio: float = 0.25,              # DO-229: vertical FSD = 0.25 * GPA
        fsd_v_min_m: float = 45.0,
        fsd_v_max_m: float = 150.0,
        ground_margin_m: float = 15.0,          # implementation floor, not TERPS
        runway_length_m: float | None = None,   # else 9023 ft FPAP floor
        lgsa_deg: float | None = None,          # override; else derived from Formula 3-1-1
    ) -> None:
        self.thr_x, self.thr_y = thr_xy
        self.thr_h = thr_h
        self.gpa = math.radians(glideslope_deg)
        self.fsd_v_ratio = fsd_v_ratio
        self.fsd_v_min = fsd_v_min_m
        self.fsd_v_max = fsd_v_max_m
        self.ground_margin = ground_margin_m

        # para. 3-1-5.c(1) + c(3)(a): FPAP = max(runway length, 9023 ft), GARP = FPAP + 1000 ft
        fpap_distance_ft = max((runway_length_m or 0.0) / 0.3048, 9023.0)
        self.d_garp = (fpap_distance_ft + 1000.0) * 0.3048

        if lgsa_deg is not None:
            self.lgsa = math.radians(lgsa_deg)
        else:
            # Formula 3-1-1: course width at threshold -> effective LGSA (1.5 deg .. 2.0 deg)
            course_width_thr = max(thr_half_width_lat_m,
                                   math.tan(math.radians(1.5)) * self.d_garp)
            self.lgsa = math.atan(course_width_thr / self.d_garp)

        dx, dy = faf_xy[0] - self.thr_x, faf_xy[1] - self.thr_y
        self.s_max = math.hypot(dx, dy)
        self.along = (dx / self.s_max, dy / self.s_max)
        self.lateral = (-self.along[1], self.along[0])

    def centerline_height(self, s: float) -> float:
        return self.thr_h + s * math.tan(self.gpa)

    def half_width_lateral(self, s: float) -> float:
        # angular sector about the GARP, located behind the threshold
        return math.tan(self.lgsa) * (s + self.d_garp)

    def half_width_vertical(self, s: float) -> float:
        angular = math.tan(self.fsd_v_ratio * self.gpa) * s
        return min(max(angular, self.fsd_v_min), self.fsd_v_max)

    def centerline_point(self, s: float) -> tuple[float, float, float]:
        x = self.thr_x + self.along[0] * s
        y = self.thr_y + self.along[1] * s
        return x, y, self.centerline_height(s)

    def boundary_point(self, s: float, sign_y: float, sign_z: float) -> tuple[float, float, float]:
        cx, cy, ch = self.centerline_point(s)
        wy = sign_y * self.half_width_lateral(s)
        wz = sign_z * self.half_width_vertical(s)

        if sign_z < 0:
            wz = max(wz, self.thr_h - self.ground_margin - ch)

        return cx + self.lateral[0] * wy, cy + self.lateral[1] * wy, ch + wz

    def ring(self, s: float) -> list[tuple[float, float, float]]:
        corners = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
        pts = [self.boundary_point(s, sy, sz) for sy, sz in corners]
        pts.append(pts[0])
        return pts