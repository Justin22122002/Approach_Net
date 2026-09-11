import time
from pathlib import Path

import pandas as pd
import pygame
from pyproj import Transformer

from rendering.visualizer import Visualizer
from utils.aircraft import Aircraft
from utils.aircraft_history_interpolator import AircraftHistoryInterpolator, plot_aircraft_trajectory

from utils.airport import Airport
from utils.simple_renderer.aircraft_2d_renderer import Aircraft2dRenderer
from utils.simple_renderer.airport_2d_renderer import Airport2dRenderer

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def main():
    basePath: Path = Path("data")

    frankfurt = Airport.load_from_folder(basePath, "EDDF")
    toronto = Airport.load_from_folder(basePath, "CYYZ")
    jfk = Airport.load_from_folder(basePath, "KJFK")

    airport = toronto

    history_file_path: Path = basePath / Path(f"{airport.icao}/test/arrivals_raw_20260709_1207.parquet")
    interpolator: AircraftHistoryInterpolator = AircraftHistoryInterpolator(history_file_path, airport_height=airport.arinc_424_runway[0].runway_altitude)
    interpolator.clean_dataframe()
    interpolator.interpolate()

    plot_aircraft_trajectory(interpolator, icao24="3c6583")

    print("Runways:", len(airport.open_street_map_runway))
    print("Taxiways:", len(airport.taxiways))
    for idx, runway in enumerate(airport.open_street_map_runway):
        print(f"index: {idx + 1}, coordinates amount: {len(runway.centerline_coordinates)}, name: {runway.name}, ref: {runway.ref}")

    visualizer = Visualizer(width=1080, height=1080)

    all_lats = [c.lat for rwy in airport.open_street_map_runway for c in rwy.centerline_coordinates] + \
               [c.lat for twy in airport.taxiways for c in twy.coordinates]
    all_lons = [c.lon for rwy in airport.open_street_map_runway for c in rwy.centerline_coordinates] + \
               [c.lon for twy in airport.taxiways for c in twy.coordinates]

    lamin = min(all_lats)
    lamax = max(all_lats)
    lomin = min(all_lons)
    lomax = max(all_lons)

    mercator_x, mercator_y = transformer.transform(airport.lon, airport.lat)
    xmin, ymin = transformer.transform(lomin, lamin)
    xmax, ymax = transformer.transform(lomax, lamax)
    scale = min(visualizer.width / (xmax - xmin), visualizer.height / (ymax - ymin))
    scale *= visualizer.zoom

    visualizer.offset_x = visualizer.width / 2 - (mercator_x - xmin) * scale
    visualizer.offset_y = visualizer.height / 2 + (mercator_y - ymin) * scale

    running = True
    mouse_pos = None
    dragging = False
    drag_start = (0, 0)

    df = interpolator.dataframe
    df["time"] = pd.to_datetime(df["time"], unit='s', errors='coerce')
    current_sim_time = df["time"].min()
    window = pd.Timedelta(seconds=1)

    while running:

        time.sleep(0.1)

        flight_df = df[
            (df["time"] >= current_sim_time) &
            (df["time"] < current_sim_time + window)
            ]

        current_sim_time += window

        aircraft_objects = []
        if flight_df is not None and not flight_df.empty:
            for idx, row in flight_df.iterrows():
                icao24 = row.get("icao24")
                callsign = row.get("callsign")
                origin_country = None
                last_position = row.get("lastposupdate")
                timestamp = row.get("time")
                longitude = row.get("lon")
                latitude = row.get("lat")
                altitude = row.get("baroaltitude")
                onground = row.get("onground")
                groundspeed = row.get("velocity")
                track = row.get("heading")
                vertical_rate = row.get("vertrate")
                sensors = row.get("serials")
                geoaltitude = row.get("geoaltitude")
                squawk = row.get("squawk")
                spi = row.get("spi")
                position_source = None
                category = None

                data = Aircraft(
                    icao24=icao24,
                    callsign=callsign,
                    origin_country=origin_country,
                    last_position=last_position,
                    timestamp=timestamp,
                    longitude=longitude,
                    latitude=latitude,
                    altitude=altitude,
                    onground=onground,
                    groundspeed=groundspeed,
                    track=track,
                    vertical_rate=vertical_rate,
                    sensors=sensors,
                    geoaltitude=geoaltitude,
                    squawk=squawk,
                    spi=spi,
                    position_source=position_source,
                    category=category,
                )
                aircraft_objects.append(Aircraft2dRenderer(data))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
                if dragging:
                    dx = event.pos[0] - drag_start[0]
                    dy = event.pos[1] - drag_start[1]
                    visualizer.offset_x += dx
                    visualizer.offset_y += dy
                    drag_start = event.pos

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drag_start = event.pos
                    dragging = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False

            elif event.type == pygame.MOUSEWHEEL and mouse_pos:
                mx, my = mouse_pos
                old_zoom = visualizer.zoom

                if event.y > 0:
                    visualizer.zoom *= 1.1
                else:
                    visualizer.zoom /= 1.1

                visualizer.offset_x = mx - (mx - visualizer.offset_x) * (visualizer.zoom / old_zoom)
                visualizer.offset_y = my - (my - visualizer.offset_y) * (visualizer.zoom / old_zoom)

        objects: list = [Airport2dRenderer(airport)] + aircraft_objects
        visualizer.render_objects(objects, lamin, lamax, lomin, lomax)

if __name__ == "__main__":
    main()