import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from OpenGL.GL import glDeleteVertexArrays, glDeleteBuffers
from pyproj import Transformer

from engine.utils.camera import FlyCamera
from engine.utils.color import Color
from engine.utils.directionalLight import DirectionalLight
from engine.utils.engine import Engine
from engine.utils.lineRenderer import LineRenderer
from engine.utils.pointRenderer import PointRenderer
from engine.utils.vec3 import Vec3
from engine.utils_project.approach_project_mesh import ApproachProjectMesh
from engine.utils_project.geo_transformer import GeoTransformer
from engine.utils_project.rnav_lpv_funnel_project_mesh import RnavLpvFunnelProjectMesh
from engine.utils_project.runway_project_mesh import RunwayProjectMesh
from engine.utils_project.terrain_project_mesh import TerrainProjectMesh
from utils.airport import Airport

ICAO = "CYVR"
DATA_ROOT = Path("./data")
OUTPUT_FOLDER = "multiple_airports_ft_transformer_gnn_rollout_constraint_training"
RUN_ID = "2026-10-11_17-13-54"
CONSTRAINED = True
TAG = f"gnn{'_constrained' if CONSTRAINED else ''}_rollout_{RUN_ID}"

SIM_SPEED: float = 200.0
TRAIL_S: float = 90.0
HEADING_LEN_S: float = 30.0
SKIP_EMPTY: bool = True
HUD_MAX_ROWS: int = 14

NM_TO_M, FT_TO_M = 1852.0, 0.3048
SEP_H_M, SEP_V_M = 2.5 * NM_TO_M, 1000.0 * FT_TO_M
HIDDEN = Vec3(0.0, -1.0e5, 0.0)

COLOR_LANDED = Color(0.2, 0.9, 0.3)
COLOR_FAILED = Color(1.0, 0.25, 0.2)
COLOR_SEP = Color(1.0, 0.6, 0.0)
COLOR_HEADING = Color(0.95, 0.95, 0.95)


@dataclass
class ReplayFlight:
    icao24: str
    flight_id: int
    landed: bool
    t: np.ndarray        # (n,) scene clock [unix s]
    x: np.ndarray        # (n,) UTM [m], absolute
    y: np.ndarray
    h: np.ndarray        # (n,) [m], absolute
    v: np.ndarray
    chi: np.ndarray      # (n,) [rad]
    gamma: np.ndarray    # (n,) [rad]
    sep: np.ndarray      # (n,) bool, separation QP active
    min_dist: np.ndarray # (n,) horizontal distance to the nearest aircraft [m]
    dh: np.ndarray       # (n,) height difference to it [m]
    point_go: object = None
    trail_lr: LineRenderer | None = None
    head_lr: LineRenderer | None = None

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    def index(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1))

    def position(self, t: float) -> tuple[float, float, float]:
        return float(np.interp(t, self.t, self.x)), float(np.interp(t, self.t, self.y)), float(np.interp(t, self.t, self.h))


def load_flights(path: Path, status_path: Path) -> list[ReplayFlight]:
    df = pd.read_parquet(path)
    if "t_sim" not in df.columns:
        raise RuntimeError("no t_sim column -> rerun the rollout with the current multi_rollout_evaluator.py")
    status = [json.loads(line) for line in open(status_path)]
    landed = {(str(s["icao24"]), int(s["flight_id"])): bool(s["success"]) for s in status}

    flights: list[ReplayFlight] = []
    for (icao24, flight_id), g in df.groupby(["icao24", "flight_id"], sort=False):
        g = g.sort_values("step")
        t = g["t_sim"].to_numpy(dtype=float).copy()
        dt = np.nanmedian(np.diff(t)) if np.isfinite(t).sum() > 1 else 1.0
        for i in range(1, len(t)):
            if not np.isfinite(t[i]):
                t[i] = t[i - 1] + dt
        if not np.isfinite(t).all():
            continue
        col = lambda name, fill=0.0: g[name].fillna(fill).to_numpy() if name in g else np.full(len(g), fill)
        flights.append(ReplayFlight(
            icao24=str(icao24), flight_id=int(flight_id), landed=landed.get((str(icao24), int(flight_id)), False),
            t=t, x=col("x"), y=col("y"), h=col("h"), v=col("v"), chi=col("chi"), gamma=col("gamma"),
            sep=col("sep_active", False).astype(bool),
            min_dist=col("min_dist_h_m", np.nan), dh=col("dh_nearest_m", np.nan),
        ))
    flights.sort(key=lambda f: f.t_start)
    print(f"[replay] {len(flights)} flights from {path.name}, "
          f"{(flights[-1].t_end - flights[0].t_start) / 60:.0f} min scene time")
    return flights


def set_lines(lr: LineRenderer, points: list[Vec3], colors: list[Color]) -> None:
    if len(points) < 2:
        points, colors = [HIDDEN, HIDDEN], [colors[0], colors[0]] if colors else [COLOR_HEADING] * 2
    seg_pts: list[Vec3] = []
    seg_cols: list[Color] = []
    for i in range(len(points) - 1):
        seg_pts += [points[i], points[i + 1]]
        seg_cols += [colors[i], colors[i + 1]]
    lr.points, lr.colors = seg_pts, seg_cols
    if lr._vao is not None:
        glDeleteVertexArrays(1, [lr._vao])
        glDeleteBuffers(1, [lr._vbo])
        lr._vao = None
        lr._vbo = None


def main() -> None:
    airport = Airport.load_from_folder(DATA_ROOT, ICAO)
    transformer = Transformer.from_crs("EPSG:4326", GeoTransformer.utm_epsg(airport.lat, airport.lon), always_xy=True)
    horizontal_scale = vertical_scale = 0.01
    geo = GeoTransformer(
        lat=airport.lat,
        lon=airport.lon,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        transformer=transformer
    )

    engine = Engine(1280, 720, f"Rollout replay - {ICAO} - {TAG}")
    scene = engine.scene
    terrain = TerrainProjectMesh(
        grid=airport.terrain.elevation_grid,
        lat_min=airport.terrain.lat_min,
        lon_min=airport.terrain.lon_min,
        lat_max=airport.terrain.lat_max,
        lon_max=airport.terrain.lon_max,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        transformer=transformer,
    )
    terrain.spawn(scene=scene, geo=geo)

    rw = None
    for runway in airport.arinc_424_runway:
        rw = RunwayProjectMesh(
            runway=runway,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            transformer=transformer
        ).spawn(scene, geo)[0].transform.position

        for approach in runway.approaches:
            ApproachProjectMesh(
                approach=approach,
                horizontal_scale=horizontal_scale,
                vertical_scale=vertical_scale,
                transformer=transformer
            ).spawn(scene, geo)

            RnavLpvFunnelProjectMesh(
                approach=approach,
                transformer=transformer,
                horizontal_scale=horizontal_scale,
                vertical_scale=vertical_scale,
                runway=runway
            ).spawn(scene, geo)

    scene.ambient_light = Color(0.25, 0.25, 0.3)
    scene.directional_light = DirectionalLight(
        direction=Vec3(-1, -2, -1),
        color=Color(1.0, 0.95, 0.85),
        intensity=1.0
    )
    engine.camera = FlyCamera(speed=100.0)
    rw = rw or Vec3(0.0, 0.0, 0.0)
    engine.camera.position = Vec3(rw.x, geo.altitude_to_local(terrain.altitude) + 50, rw.z)

    folder = DATA_ROOT / ICAO / OUTPUT_FOLDER
    flights = load_flights(folder / f"{TAG}.parquet", folder / f"{TAG}_status.jsonl")
    if not flights:
        raise RuntimeError("nothing to replay")
    t_first, t_last = flights[0].t_start, max(f.t_end for f in flights)

    def to_local(x: float, y: float, h: float) -> Vec3:
        xg, zg = geo.to_local_in_m(x, y)
        return Vec3(xg, geo.altitude_to_local(h), zg)

    state = {"t": t_first, "next": 0, "active": [], "finished": 0, "loops": 0}

    def spawn(f: ReplayFlight) -> None:
        if f.point_go is None:
            f.point_go = scene.spawn(f"ac_{f.icao24}_{f.flight_id}")
            f.point_go.add_component(PointRenderer(color=COLOR_LANDED if f.landed else COLOR_FAILED, size=120.0))
            f.trail_lr = LineRenderer([HIDDEN, HIDDEN], color=Color(0.3, 0.75, 1.0))
            scene.spawn(f"trail_{f.icao24}_{f.flight_id}").add_component(f.trail_lr)
            f.head_lr = LineRenderer([HIDDEN, HIDDEN], color=COLOR_HEADING)
            scene.spawn(f"heading_{f.icao24}_{f.flight_id}").add_component(f.head_lr)
        state["active"].append(f)

    def park(f: ReplayFlight) -> None:
        f.point_go.transform.position = HIDDEN
        set_lines(f.trail_lr, [], [COLOR_HEADING])
        set_lines(f.head_lr, [], [COLOR_HEADING])

    def restart() -> None:
        for f in state["active"]:
            park(f)
        state.update(t=t_first, next=0, active=[], finished=0, loops=state["loops"] + 1)

    def on_update(dt: float) -> None:
        state["t"] += dt * SIM_SPEED
        if state["t"] > t_last:
            restart()

        while state["next"] < len(flights) and flights[state["next"]].t_start <= state["t"]:
            spawn(flights[state["next"]])
            state["next"] += 1
        still = []
        for f in state["active"]:
            if state["t"] > f.t_end:
                park(f)
                state["finished"] += 1
            else:
                still.append(f)
        state["active"] = still
        if SKIP_EMPTY and not still and state["next"] < len(flights):
            state["t"] = flights[state["next"]].t_start
            return

        t = state["t"]
        for f in still:
            x, y, h = f.position(t)
            pos = to_local(x, y, h)
            f.point_go.transform.position = pos

            lo = f.index(t - TRAIL_S)
            hi = f.index(t)
            pts = [to_local(f.x[i], f.y[i], f.h[i]) for i in range(lo, hi + 1)] + [pos]
            n = len(pts)
            cols = [Color(0.1 * b, 0.6 * b, 1.0 * b) for b in (0.15 + 0.85 * i / max(n - 1, 1) for i in range(n))]
            set_lines(f.trail_lr, pts, cols)

            i = f.index(t)
            d = f.v[i] * HEADING_LEN_S
            end = to_local(x + d * math.cos(f.gamma[i]) * math.sin(f.chi[i]),
                           y + d * math.cos(f.gamma[i]) * math.cos(f.chi[i]),
                           h + d * math.sin(f.gamma[i]))
            color = COLOR_SEP if f.sep[i] else COLOR_HEADING
            set_lines(f.head_lr, [pos, end], [color, color])

    def on_gui() -> None:
        hud = engine.hud
        t = state["t"]
        active = state["active"]
        clock = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        rows = active[:HUD_MAX_ROWS]

        hud.panel(10, 10, 560, 110 + 18 * max(len(rows), 1), border_radius=8)
        hud.text(f"{TAG}", 20, 16, size=13, color=(60, 210, 255), shadow=True)
        hud.text(f"{clock}   x{SIM_SPEED:.0f}   airborne {len(active)}   finished {state['finished']}   "
                 f"pending {len(flights) - state['next']}", 20, 36, size=12, color=(200, 200, 210))
        hud.line(20, 56, 560, 56, color=(80, 80, 100, 180))

        y_px = 62
        for f in rows:
            i = f.index(t)
            _, _, h = f.position(t)
            los = f.min_dist[i] < SEP_H_M and f.dh[i] < SEP_V_M
            tags = ("  SEP" if f.sep[i] else "") + ("  LoS" if los else "")
            dist = f"{f.min_dist[i] / NM_TO_M:5.1f} NM" if np.isfinite(f.min_dist[i]) else "   -    "
            hud.text(f"{f.icao24:8s} h={h:5.0f} m  v={f.v[i]:4.0f} m/s  nearest {dist}{tags}",
                     20, y_px, size=12,
                     color=(255, 80, 60) if los else ((255, 170, 60) if tags else (220, 220, 230)))
            y_px += 18
        if len(active) > HUD_MAX_ROWS:
            hud.text(f"... +{len(active) - HUD_MAX_ROWS} more", 20, y_px, size=12, color=(160, 160, 180))
        hud.crosshair()

    engine.on_update = on_update
    engine.on_gui = on_gui
    engine.run()


if __name__ == "__main__":
    main()