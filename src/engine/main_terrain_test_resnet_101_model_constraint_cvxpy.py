import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from pyproj import Transformer

from ml.train_resnet_101_all_airports import LandingActionResNet101
from ml.utils.data_loader.landing_action_dataset import LandingActionDataset
from ml.utils.data_loader.lazy_flight_parquet_dataset import LazyFlightParquetDataset
from ml.utils.models.constraint_layer_cvxpy import (
    AircraftEnvelopeQPLayer, AircraftFunnelQPLayer, AircraftCombinedQPLayer,
)

from engine.physics.aircraft_environment import AircraftEnvironment
from engine.physics.aircraft_kinematics_state import AircraftKinematicsState

from utils.airport import Airport
from utils.arinc_runway import ArincRunway
from engine.utils.engine import Engine
from utils.clients.terrain_client import CopernicusDEMProvider

from engine.utils.lineRenderer import LineRenderer
from engine.utils.color import Color
from engine.utils.directionalLight import DirectionalLight
from engine.utils.camera import OrbitCamera
from engine.utils.pointRenderer import PointRenderer
from engine.utils.vec3 import Vec3
from engine.utils_project.geo_transformer import GeoTransformer
from engine.utils_project.runway_project_mesh import RunwayProjectMesh
from engine.utils_project.terrain_project_mesh import TerrainProjectMesh

from dotenv import load_dotenv

load_dotenv()

ICAO = "CYVR"
FLIGHT_DATA_PATH = "../data/CYVR/test/20260709_1207"  # "../data/CYVR/test/20260709_1207" "../data/CYVR/train/20260802_0920"
FLIGHT_IDX = 105

TIME = "2026-08-03_15-43-07"  # results/multiple_airports_landing_action_resnet101_torchvision/<TIME>

MODEL_PATH = f"../ml/results/multiple_airports_landing_action_resnet101_torchvision/{TIME}/best_model.pth"
NORM_X_PATH = f"../ml/results/multiple_airports_landing_action_resnet101_torchvision/{TIME}/normalization_x.csv"
NORM_Y_PATH = f"../ml/results/multiple_airports_landing_action_resnet101_torchvision/{TIME}/normalization_y.csv"

PHYS_DT = 1.0

STALL_MARGIN = 1.15
ALPHA_STALL = 1.0

# "envelope"
# "funnel"
# "combined"
LAYER_MODE = "combined"


def load_norm(path: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[0].astype(np.float32), arr[1].astype(np.float32)


basePath = Path("../data")
airport = Airport.load_from_folder(basePath, ICAO)

crs_to = GeoTransformer.utm_epsg(airport.lat, airport.lon)
to_utm = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)
to_latlon = Transformer.from_crs(crs_to, "EPSG:4326", always_xy=True)

airport_cx, airport_cy = to_utm.transform(airport.lon, airport.lat)

airport_elev = airport.get_mean_runway_elevation_ft() * 0.3048

runways: list[ArincRunway] = airport.arinc_424_runway

delta = 0.2
LAT_MIN, LAT_MAX = airport.lat - delta, airport.lat + delta
LON_MIN, LON_MAX = airport.lon - delta, airport.lon + delta

provider = CopernicusDEMProvider(
   os.environ["COPERNICUS_USERNAME"],
   os.environ["COPERNICUS_PASSWORD"],
)

grid = provider.get_terrain(
    lat_min=LAT_MIN,
    lon_min=LON_MIN,
    lat_max=LAT_MAX,
    lon_max=LON_MAX,
    width=1024,
    height=1024,
)

horizontal_scale = 0.01
vertical_scale = 0.01

geo = GeoTransformer(
    lat=airport.lat,
    lon=airport.lon,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    transformer=to_utm,
)

terrain_project_mesh = TerrainProjectMesh(
    grid=grid,
    lat_min=LAT_MIN,
    lon_min=LON_MIN,
    lat_max=LAT_MAX,
    lon_max=LON_MAX,
    horizontal_scale=horizontal_scale,
    vertical_scale=vertical_scale,
    transformer=to_utm,
)

engine = Engine(1280, 720, f"ResNet101-Policy ({LAYER_MODE}) - {ICAO}")
scene = engine.scene

terrain_project_mesh.spawn(scene=scene, geo=geo)

for runway in runways:
    runway_mesh_builder = RunwayProjectMesh(
        runway=runway,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        transformer=to_utm,
    )
    runway_mesh_builder.spawn(scene, geo)

scene.ambient_light = Color(0.25, 0.25, 0.3)
scene.directional_light = DirectionalLight(
    direction=Vec3(-1, -2, -1),
    color=Color(1.0, 0.95, 0.85),
    intensity=1.0,
)

cam = OrbitCamera(distance=5.0)
cam.pitch = 15.0
engine.camera = cam

flight_dataset = LazyFlightParquetDataset(FLIGHT_DATA_PATH)
icao24, flight_id = flight_dataset.flights[FLIGHT_IDX]
df_flight = flight_dataset[FLIGHT_IDX]

start_point, _ = flight_dataset.get_start_and_end_points(df_flight)
_, aircraft_dynamics = flight_dataset.get_aircraft(icao24)
runway_row = flight_dataset.get_runway_assignment(icao24, flight_id)
runway_elevation_m = float(runway_row["fix_elevation_m"])
runway_bearing = float(runway_row["runway_bearing"])

x0, y0 = to_utm.transform(float(start_point["lon"]), float(start_point["lat"]))
v0 = float(start_point["velocity"])
h0 = float(start_point["geoaltitude"])
heading0 = float(start_point["heading"]) if not np.isnan(start_point["heading"]) else 0.0
vertrate0 = float(start_point["vertrate"]) if not np.isnan(start_point["vertrate"]) else 0.0
gamma0 = float(np.degrees(np.arctan2(vertrate0, max(v0, 1e-3))))

aircraft_state = AircraftKinematicsState(
    x=x0, y=y0, h=h0, v=v0, gamma=gamma0, chi=heading0,
    aircraft_type=aircraft_dynamics,
)
env = AircraftEnvironment(state=aircraft_state, goal=None, dt=PHYS_DT)

dyn_feat = np.array(
    [getattr(aircraft_dynamics, k) for k in LandingActionDataset.DYNAMICS_KEYS],
    dtype=np.float32,
)

fix_x, fix_y = to_utm.transform(
    float(runway_row["fix_longitude"]), float(runway_row["fix_latitude"])
)
fix_x_c = fix_x - airport_cx
fix_y_c = fix_y - airport_cy
rwy_feat = np.array([
    fix_x_c, fix_y_c, runway_elevation_m - airport_elev, runway_bearing,
], dtype=np.float32)

print(runway_row["fix_identifier"])

x_mean, x_std = load_norm(NORM_X_PATH)
y_mean, y_std = load_norm(NORM_Y_PATH)

model = LandingActionResNet101(input_dim=len(x_mean), output_dim=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

_layer_kwargs = dict(base_model=model, y_mean=y_mean, y_std=y_std, debug=False)

if LAYER_MODE == "envelope":
    guided_model = AircraftEnvelopeQPLayer(
        **_layer_kwargs, stall_margin=STALL_MARGIN, alpha_stall=ALPHA_STALL, solver=None
    )
elif LAYER_MODE == "funnel":
    guided_model = AircraftFunnelQPLayer(
        **_layer_kwargs, h_thr=runway_elevation_m, solver=None
    )
elif LAYER_MODE == "combined":
    guided_model = AircraftCombinedQPLayer(
        **_layer_kwargs, stall_margin=STALL_MARGIN, alpha_stall=ALPHA_STALL, h_thr=runway_elevation_m, solver=None
    )
else:
    raise ValueError(f"unknown LAYER_MODE: {LAYER_MODE!r}")

landed = False
thrust, load_factor, roll = 0.0, 1.0, 0.0


def predict_action() -> tuple[float, float, float]:
    x_ac_c = float(aircraft_state.x) - airport_cx
    y_ac_c = float(aircraft_state.y) - airport_cy

    state_feat = np.array(
        [x_ac_c, y_ac_c, aircraft_state.h - airport_elev, aircraft_state.v, aircraft_state.gamma, aircraft_state.chi],
        dtype=np.float32,
    )
    x = np.concatenate([dyn_feat, state_feat, rwy_feat])
    assert x.shape[0] == x_mean.shape[0], f"feature dim {x.shape[0]} != normalization dim {x_mean.shape[0]}"
    x_norm = torch.from_numpy((x - x_mean) / x_std).unsqueeze(0)

    aircraft_state_t = torch.tensor(
        [[aircraft_state.h, aircraft_state.v, aircraft_state.gamma]], dtype=torch.float32
    )
    aircraft_dyn_t = torch.from_numpy(dyn_feat).unsqueeze(0)
    aircraft_limits_t = torch.tensor([[
        aircraft_dynamics.max_thrust, aircraft_dynamics.min_load_factor,
        aircraft_dynamics.max_load_factor, aircraft_dynamics.max_roll_angle,
    ]], dtype=torch.float32)

    if LAYER_MODE == "envelope":
        out = guided_model(x_norm, aircraft_state_t, aircraft_dyn_t, aircraft_limits_t)
    else:
        aircraft_pos_t = torch.tensor([[x_ac_c, y_ac_c, aircraft_state.chi]], dtype=torch.float32)
        runway_pos_t = torch.tensor([[fix_x_c, fix_y_c, runway_bearing]], dtype=torch.float32)
        out = guided_model(
            x_norm, aircraft_state_t, aircraft_dyn_t, aircraft_limits_t,
            aircraft_pos=aircraft_pos_t, runway_pos=runway_pos_t,
        )

    t, n, phi = out.squeeze(0).tolist()

    t = float(np.clip(t, 0.0, aircraft_dynamics.max_thrust))
    n = float(np.clip(n, aircraft_dynamics.min_load_factor, aircraft_dynamics.max_load_factor))
    phi = float(np.clip(phi, -aircraft_dynamics.max_roll_angle, aircraft_dynamics.max_roll_angle))

    return t, n, phi


trail_pts: list[Vec3] = []
_phys_acc = 0.0

aircraft_go = scene.spawn("Aircraft")
xg, zg = geo.to_local_in_m(aircraft_state.x, aircraft_state.y)
yg = geo.altitude_to_local(aircraft_state.h)
aircraft_go.transform.position = Vec3(xg, yg, zg)
aircraft_go.add_component(PointRenderer(color=Color(1.0, 0.85, 0.0), size=120.0))

vel_go = scene.spawn("VelocityVector")
vel_lr = LineRenderer([Vec3.zero(), Vec3.zero()], color=Color(0.1, 1.0, 0.3))
vel_go.add_component(vel_lr)

trail_go = scene.spawn("Trail")
trail_lr = LineRenderer([Vec3.zero(), Vec3.zero()], color=Color(0.3, 0.75, 1.0))
trail_go.add_component(trail_lr)

def on_update(dt: float) -> None:
    global thrust, load_factor, roll, _phys_acc, landed

    if not landed:
        _phys_acc += 1.0
        while _phys_acc >= PHYS_DT:
            thrust, load_factor, roll = predict_action()

            env.apply_action(thrust=thrust, load_factor=load_factor, roll=roll)
            _phys_acc -= PHYS_DT

            if aircraft_state.h <= runway_elevation_m:
                landed = True
                print(f"[INFO] gelandet bei h={float(aircraft_state.h):.0f} m")
                break

    xg, zg = geo.to_local_in_m(aircraft_state.x, aircraft_state.y)
    yg = geo.altitude_to_local(aircraft_state.h)
    wp = Vec3(xg, yg, zg)

    aircraft_go.transform.position = wp
    cam.target = wp

    trail_pts.append(Vec3(wp.x, wp.y, wp.z))
    if len(trail_pts) >= 2:
        trail_lr.set_points_strip(trail_pts)

    v_raw = aircraft_state.velocity_vector()
    vel_tip = Vec3(
        wp.x + v_raw[0] * horizontal_scale,
        wp.y + v_raw[2] * vertical_scale,
        wp.z - v_raw[1] * horizontal_scale,
    )
    vel_lr.set_points_strip([wp, vel_tip])


def on_gui() -> None:
    hud = engine.hud
    fps = engine._clock.get_fps() if engine._clock else 0.0

    hud.panel(10, 10, 285, 170, border_radius=8)
    hud.text(f"ResNet101 policy + {LAYER_MODE}-QP", 20, 16, size=14, color=(60, 210, 255), shadow=True)
    hud.line(20, 38, 285, 38, color=(80, 80, 100, 180))

    v_kts = aircraft_state.v * 1.944
    h_ft = aircraft_state.h * 3.281

    hud.text(f"Velocity   {aircraft_state.v:6.1f} m/s  ({v_kts:5.0f} kts)", 20, 44, size=15)
    hud.text(f"Height     {aircraft_state.h:6.0f} m    ({h_ft:6.0f} ft)", 20, 64, size=15)
    hud.text(f"chi        {aircraft_state.chi % 360:5.1f}°", 20, 84, size=15)
    hud.text(f"gamma      {aircraft_state.gamma:+5.1f}°", 20, 104, size=15)
    hud.line(20, 145, 285, 145, color=(80, 80, 100, 180))
    hud.text(f"FPS        {fps:5.0f}", 20, 150, size=13, color=(160, 160, 180))

    px = hud.width - 295
    hud.panel(px, 10, 285, 140, border_radius=8)
    hud.text("model output", px + 10, 16, size=17, color=(60, 210, 255), shadow=True)
    hud.line(px + 10, 38, px + 275, 38, color=(80, 80, 100, 180))

    hud.text(f"Thrust        {thrust / 1000:5.1f} kN", px + 10, 44, size=15)
    hud.bar(px + 10, 62, 265, 10, value=thrust / max(aircraft_dynamics.max_thrust, 1.0), fg_color=(255, 160, 30, 220))

    hud.text(f"Roll          {roll:+5.1f}°", px + 10, 78, size=15)
    hud.bar(px + 10, 96, 265, 10, value=(roll + 60) / 120, fg_color=(80, 180, 255, 220))

    hud.text(f"Load-Factor   {load_factor:4.2f}", px + 10, 112, size=15)
    hud.bar(px + 10, 130, 265, 10, value=load_factor / 3.0, fg_color=(100, 220, 120, 220))

    if landed:
        hud.text("GELANDET", hud.width // 2 - 50, hud.height - 40, size=20, color=(255, 80, 80))


engine.on_gui = on_gui
engine.on_update = on_update
engine.run()