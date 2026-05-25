
## Data sources and their projections
### OpenStreetMap

api: https://overpass-api.de/api/interpreter

coordinate standard / sphere projection: https://osmdata.openstreetmap.de/info/projections.html -> WGS84 (EPSG 4326)

### OpenSky

api: https://opensky-network.org/api/states

coordinate standard / sphere projection: https://openskynetwork.github.io/opensky-api/rest.html -> WGS84

### copernicus

api: https://sh.dataspace.copernicus.eu/api/v1/process 

coordinate standard / sphere projection: https://dataspace.copernicus.eu/sites/default/files/media/files/2024-06/geo1988-copernicusdem-spe-002_producthandbook_i5.0.pdf -> 1.2.1 
-> (WGS84-G1150; EPSG 4326)
-> The vertical reference datum is the Earth Gravitational Model 20081 (EGM2008; EPSG 3855).

### flightplandatabase

api: https://api.flightplandatabase.com/nav/airport -> https://flightplandatabase.com/dev/api

coordinate standard / sphere projection: -> WGS84

### airac.net

api: https://airac.net/api/v1

coordinate standard / sphere projection: rnav approch plates -> WGS84

## General comments:
- performance targets: can we flexibly balance between conservative routing (high safety) and high throughput (lower safety)?
- visualization: mention 2.5D or 3D visualization.
- key pitch: guaranteed forward-enforcement of constraints.
- stress score: how close is the system to collapse (= no feasible solutions).
- assistance: scalar forward-risk score similar to chess engines.

## geoid correction

see: https://en.wikipedia.org/wiki/Geoid
Model	    Year	 Resolution
EGM96	    1996	 (~30 km)
EGM2008	    2008	 (~10 km)
EGM08_25	aus 2008 (~5 km)