import time
from pathlib import Path
from typing import TYPE_CHECKING

import pygame
from pyproj import Transformer

from src.rendering.visualizer import Visualizer
from src.utils.aircraft import Aircraft, AircraftData
from src.utils.airport_client import AirportClient
from src.utils.open_sky_rest_client import OpenSkyRestClient

if TYPE_CHECKING:
    from src.utils.airport import Airport

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def main():
    client = AirportClient()
    frankfurt: Airport = client.get_airport(
        icao="EDDF",
        lat=50.0379,
        lon=8.5622,
        radius_km=5
    )

    toronto: Airport = client.get_airport(
        icao="CYYZ",
        lat=43.6777,
        lon=-79.6248,
        radius_km=5
    )

    jfk = client.get_airport(
        icao="KJFK",
        lat=40.63993,
        lon=-73.77869,
        radius_km=5
    )

    basePath: Path = Path("data")
    frankfurt.save_runways_to_json(basePath / Path("frankfurt_runways.json"))
    frankfurt.load_runways_from_json(basePath / Path("frankfurt_runways.json"))

    toronto.save_runways_to_json(basePath / Path("toronto_runways.json"))
    toronto.load_runways_from_json(basePath / Path("toronto_runways.json"))

    jfk.save_runways_to_json(basePath / Path("jfk_runways.json"))
    jfk.load_runways_from_json(basePath / Path("jfk_runways.json"))

    airport = jfk

    opensky_client = OpenSkyRestClient()

    half_edge_km = 10

    print("Runways:", len(airport.runways))
    print("Taxiways:", len(airport.taxiways))
    for idx, runway in enumerate(airport.runways):
        print(f"index: {idx + 1}, coordinates amount: {len(runway.centerline_coordinates)}, name: {runway.name}, ref: {runway.ref}")

    visualizer = Visualizer(width=1080, height=1080)

    all_lats = [c.lat for rwy in airport.runways for c in rwy.centerline_coordinates] + \
               [c.lat for twy in airport.taxiways for c in twy.coordinates]
    all_lons = [c.lon for rwy in airport.runways for c in rwy.centerline_coordinates] + \
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

    last_update_time = 0
    aircraft_objects = []
    get_live_aircrafts = True

    running = True
    mouse_pos = None
    dragging = False
    drag_start = (0, 0)


    while running:
        current_time = time.time()
        if (current_time - last_update_time >= 5) and get_live_aircrafts:
            flight_df = opensky_client.get_flights_in_area(half_edge_km, airport.lat, airport.lon)
            last_update_time = current_time
            print(f"Flüge aktualisiert: {len(flight_df) if flight_df is not None else 0}")

            aircraft_objects = []
            if flight_df is not None and not flight_df.empty:
                for idx, row in flight_df.iterrows():
                    icao24 = row.get("icao24")
                    callsign = row.get("callsign")
                    origin_country = row.get("origin_country")
                    last_position = row.get("last_position")
                    timestamp = row.get("timestamp")
                    longitude = row.get("longitude")
                    latitude = row.get("latitude")
                    altitude = row.get("altitude")
                    onground = row.get("onground")
                    groundspeed = row.get("groundspeed")
                    track = row.get("track")
                    vertical_rate = row.get("vertical_rate")
                    sensors = row.get("sensors")
                    geoaltitude = row.get("geoaltitude")
                    squawk = row.get("squawk")
                    spi = row.get("spi")
                    position_source = row.get("position_source")
                    category = row.get("category")

                    data = AircraftData(
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
                    aircraft_objects.append(Aircraft(data))

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

        objects: list = [airport] + aircraft_objects
        visualizer.render_objects(objects, lamin, lamax, lomin, lomax)

if __name__ == "__main__":
    main()