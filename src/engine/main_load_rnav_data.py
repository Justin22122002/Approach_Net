import os
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset

from utils.airport import Airport
from utils.arinc_runway import ArincRunway
from utils.clients.terrain_client import CopernicusDEMProvider

from engine.utils.engine import Engine
from engine.utils.color import Color
from engine.utils.directionalLight import DirectionalLight
from engine.utils.camera import OrbitCamera
from engine.utils.pointRenderer import PointRenderer
from engine.utils.lineRenderer import LineRenderer
from engine.utils.vec3 import Vec3
from engine.utils_project.geo_transformer import GeoTransformer
from engine.utils_project.runway_project_mesh import RunwayProjectMesh
from engine.utils_project.terrain_project_mesh import TerrainProjectMesh

from dotenv import load_dotenv

load_dotenv()

icao = {"icao": "CYVR"}

DATASET_BASE_PATH = "../data/CYVR/20260625_1508"

TRAJECTORY_PARQUET = Path("../data/CYVR/rnav_optimized/rnav_trajectories.parquet")

ICAO24: str | None = None
FLIGHT_ID: int | None = None


def load_optimized_trajectory(path: Path, icao24: str | None, flight_id: int | None) -> tuple[pd.DataFrame, str, int]:
    df_all = pd.read_parquet(path)

    if icao24 is None or flight_id is None:
        icao24 = str(df_all["icao24"].iloc[0])
        flight_id = int(df_all["flight_id"].iloc[0])
        print(f"[INFO] Kein Flight gewählt -> nehme ersten Eintrag: {icao24} / {flight_id}")

    mask = (df_all["icao24"] == icao24) & (df_all["flight_id"] == flight_id)
    df_traj = df_all.loc[mask].sort_values("step").reset_index(drop=True)

    if df_traj.empty:
        raise KeyError(f"No saved trajectory for icao24={icao24!r}, flight_id={flight_id!r} in {path}")

    return df_traj, icao24, flight_id


def to_seconds_since_start(time_series: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(time_series):
        t0 = time_series.iloc[0]
        return (time_series - t0).dt.total_seconds().to_numpy()
    t = time_series.to_numpy(dtype=float)
    return t - t[0]


def to_world_points(geo: GeoTransformer, xs: np.ndarray, ys: np.ndarray, hs: np.ndarray) -> list[Vec3]:
    pts = []
    for x, y, h in zip(xs, ys, hs):
        px, pz = geo.to_local_in_m(float(x), float(y))
        py = geo.altitude_to_local(float(h))
        pts.append(Vec3(px, py, pz))
    return pts


# load optimized trajectory entry
df_opt, ICAO24, FLIGHT_ID = load_optimized_trajectory(TRAJECTORY_PARQUET, ICAO24, FLIGHT_ID)

opt_times = df_opt["time_s"].to_numpy()
opt_x = df_opt["x"].to_numpy()
opt_y = df_opt["y"].to_numpy()
opt_h = df_opt["h"].to_numpy()
T_opt = float(opt_times[-1])

print(f"[INFO] Optimized trajectory geladen: {ICAO24} / {FLIGHT_ID}  ({len(df_opt)} steps, T={T_opt:.1f}s)")

# airport + terrain + scene
basePath: Path = Path("../data")
airport = Airport.load_from_folder(basePath, icao["icao"])

crs_to = GeoTransformer.utm_epsg(airport.lat, airport.lon)
transformer = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)

runways: list[ArincRunway] = airport.arinc_424_runway

provider = CopernicusDEMProvider(
   os.environ["COPERNICUS_USERNAME"],
   os.environ["COPERNICUS_PASSWORD"],
)
delta = 0.2
grid = provider.get_terrain(
    airport.lat - delta, airport.lon - delta,
    airport.lat + delta, airport.lon + delta,
    1024, 1024,
)

horizontal_scale = 0.01
vertical_scale = 0.01
geo = GeoTransformer(airport.lat, airport.lon, horizontal_scale, vertical_scale, transformer)

engine = Engine(1280, 720, "Trajectory Replay")
scene = engine.scene
TerrainProjectMesh(
    grid=grid,
    lat_min=airport.lat - delta, lon_min=airport.lon - delta,
    lat_max=airport.lat + delta, lon_max=airport.lon + delta,
    horizontal_scale=horizontal_scale, vertical_scale=vertical_scale,
    transformer=transformer,
).spawn(scene, geo)

scene.ambient_light = Color(0.25, 0.25, 0.3)
scene.directional_light = DirectionalLight(Vec3(-1, -2, -1), Color(1, 0.95, 0.85), 1)

cam = OrbitCamera(distance=6)
engine.camera = cam

for runway in runways:
    RunwayProjectMesh(
        runway=runway,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        transformer=transformer,
    ).spawn(scene, geo)

dataset = LazyFlightParquetDataset(DATASET_BASE_PATH)
runway_row = dataset.get_runway_assignment(ICAO24, FLIGHT_ID)

goal_x, goal_y = transformer.transform(
    float(runway_row["fix_longitude"]), float(runway_row["fix_latitude"])
)
goal_h = float(runway_row["fix_elevation_m"])

faf_x, faf_y = transformer.transform(
    float(runway_row["faf_longitude"]), float(runway_row["faf_latitude"])
)
faf_h = float(runway_row["faf_elevation_m"])

goal_go = scene.spawn("Goal")
goal_go.add_component(PointRenderer(color=Color(1, 0, 0), size=140))
gx, gz = geo.to_local_in_m(goal_x, goal_y)
goal_go.transform.position = Vec3(gx, geo.altitude_to_local(goal_h), gz)

via_go = scene.spawn("ViaPoint")
via_go.add_component(PointRenderer(color=Color(0, 1, 0), size=120))
vx, vz = geo.to_local_in_m(faf_x, faf_y)
via_go.transform.position = Vec3(vx, geo.altitude_to_local(faf_h), vz)

flight_idx = dataset.flights.index((ICAO24, FLIGHT_ID))
df_real = dataset[flight_idx].sort_values("time").reset_index(drop=True)

real_x, real_y = transformer.transform(df_real["lon"].to_numpy(), df_real["lat"].to_numpy())
real_h = df_real["geoaltitude"].to_numpy()
real_t = to_seconds_since_start(df_real["time"])

print(f"[INFO] Real trajectory geladen: {len(df_real)} trackpoints, "
      f"Dauer={real_t[-1]:.1f}s")

opt_line_go = scene.spawn("OptimizedPath")
opt_line_go.add_component(LineRenderer(to_world_points(geo, opt_x, opt_y, opt_h), color=Color(1, 1, 0)))

real_line_go = scene.spawn("RealPath")
real_line_go.add_component(LineRenderer(to_world_points(geo, real_x, real_y, real_h), color=Color(0, 0.6, 1)))

opt_go = scene.spawn("Aircraft_Optimized")
opt_go.add_component(PointRenderer(color=Color(1, 0.85, 0), size=120))

real_go = scene.spawn("Aircraft_Real")
real_go.add_component(PointRenderer(color=Color(0, 0.6, 1), size=120))

cam.target = Vec3(0, 0, 0)

_t = 0.0
_opt_pos = np.zeros(3)
_real_pos = np.zeros(3)


def interp_xyz(t: float, times: np.ndarray, xs: np.ndarray, ys: np.ndarray, hs: np.ndarray) -> tuple[float, float, float]:
    x = float(np.interp(t, times, xs))
    y = float(np.interp(t, times, ys))
    h = float(np.interp(t, times, hs))
    return x, y, h


def on_update(dt_frame: float) -> None:
    global _t, _opt_pos, _real_pos

    _t += dt_frame
    if _t > T_opt:
        _t = 0.0  # loop

    ox, oy, oh = interp_xyz(_t, opt_times, opt_x, opt_y, opt_h)
    px, pz = geo.to_local_in_m(ox, oy)
    py = geo.altitude_to_local(oh)
    opt_go.transform.position = Vec3(px, py, pz)
    _opt_pos = np.array([ox, oy, oh])

    real_frac = _t / T_opt
    real_t_scaled = real_frac * real_t[-1]
    rx, ry, rh = interp_xyz(real_t_scaled, real_t, real_x, real_y, real_h)
    rpx, rpz = geo.to_local_in_m(rx, ry)
    rpy = geo.altitude_to_local(rh)
    real_go.transform.position = Vec3(rpx, rpy, rpz)
    _real_pos = np.array([rx, ry, rh])

    cam.target = opt_go.transform.position


def on_gui() -> None:
    hud = engine.hud
    fps = engine._clock.get_fps() if engine._clock else 0.0
    prog = min(_t / T_opt, 1.0)

    delta = _opt_pos - _real_pos
    dist_xy = float(np.linalg.norm(delta[:2]))
    dist_z = float(delta[2])

    hud.panel(10, 10, 320, 220, border_radius=8)
    hud.text("Trajectory Replay", 20, 16, size=17, color=(255, 210, 60), shadow=True)
    hud.line(20, 38, 320, 38, color=(80, 80, 100, 180))
    hud.text(f"icao24   {ICAO24}  flight_id {FLIGHT_ID}", 20, 44, size=14)
    hud.text(f"t        {_t:6.1f} s / {T_opt:.1f} s", 20, 64, size=14)
    hud.text(f"Progress {prog*100:4.0f}%", 20, 84, size=14)
    hud.bar(20, 100, 280, 10, value=prog, fg_color=(100, 220, 120, 220))
    hud.line(20, 118, 320, 118, color=(80, 80, 100, 180))
    hud.text(f"Δ lateral {dist_xy:6.1f} m", 20, 124, size=14, color=(255, 150, 150))
    hud.text(f"Δ height  {dist_z:+6.1f} m", 20, 144, size=14, color=(255, 150, 150))
    hud.line(20, 164, 320, 164, color=(80, 80, 100, 180))
    hud.text("yellow = optimized   blue = real", 20, 170, size=13, color=(200, 200, 210))
    hud.text(f"FPS      {fps:5.0f}", 20, 192, size=13, color=(160, 160, 180))


engine.on_update = on_update
engine.on_gui = on_gui
engine.run()