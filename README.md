# ApproachNet: Physics-Grounded Safety Layers for Data-Driven Aircraft Approach Control

<table>
  <tr>
    <td><img src="images/screenshot_1.png" width="400"/></td>
    <td><img src="images/screenshot_2.png" width="400"/></td>
    <td><img src="images/screenshot_3.png" width="400"/></td>
    <td><img src="images/screenshot_4.png" width="400"/></td>
  </tr>
</table>

Automated guidance of aircraft through the approach phase typically relies on either 
optimal control or neural architectures. This repository implements **ApproachNet**, 
a data-driven framework that combines the strengths of both paradigms: it enforces 
physical and operational hard constraints while retaining the computational efficiency 
of learning-based approaches.

ApproachNet predicts low-level control commands (thrust, load factor, bank) using two 
compatible neural backbones — **ResNet-101** and **FT-Transformer** — trained on 
historical ADS-B data. These predictions are grounded in a **4-DOF aircraft dynamics 
model** and projected onto a feasible, safe control set via a differentiable 
**control barrier function quadratic program (CBF-QP)** safety layer. As a classical 
reference, a **CasADi/IPOPT-based trajectory optimizer** is included for comparison.

The resulting trajectories are feasible, safe, and computable at interactive rates, and 
generalize across multiple airports, runways, and aircraft types.

## Get started

All experiments were conducted with Python 3.13 and the package versions specified in 
the `requirements.txt` file in the root folder of this project.

1. create and activate a virtual environment in the root folder of this project:
    ```python -m venv venv```
    Windows: ```venv\Scripts\activate```
    Linux/macOS: ```source venv/bin/activate```
2. install the required packages:
    ```pip install -r requirements.txt```
3. Always run scripts from the ```src``` folder or 
   run all scripts from the root folder of this project in Pycharms or vscode.

### Login / Account requirements

1. Copernicus sentinel 2 (terrain data) -> https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2
    credential: via env file, see following format in root folder as a .env file of this project
    ```
    COPERNICUS_USERNAME="example@example.com"
    COPERNICUS_PASSWORD="password"
   ```
2. OpenSky Trino Client (for historical flight data) -> https://openskynetwork.github.io/opensky-api/trino.html
    see docs for credential setup

### Data Retrieval

1. navigate to ```src/main_load_data.py``` and run the script from the root folder of this project, after running the script the airport data for the 
    following experiments will be saved under ```src/data```
2. navigate to ```src/utils/clients/open_sky_trino_client.py``` and run the scirpt from the root folder of this project. 
    note: you can adjust the number of arrivals via the target_n_arrivals_train and target_n_arrivals_test parameter. (this can take depending on the 
    size of the flight data up to multiple hours to days)

### Experiments

#### Training:

1. navigate to ```src/ml``` and run the scripts from the root folder of this project, while adjusting the dataset_name with the file data name of the downloaded train trino dataset. 
    -> ```src/ml/train_ft_transformer_all_airports.py``` and ```src/ml/train_ft_transformer_next_state_all_airports.py```
    -> ```src/ml/train_resnet_101_all_airports.py``` and ```src/ml/train_resnet_101_next_state_all_airports.py```

#### Evaluation:

1. navigate to ```src/evaluation``` and run the scripts from the root folder of this project.
2. adjust the dataset_name with the file data name of the downloaded test trino dataset. Also adjust the TIME parameter with the corresponding model time path.
3. run all following scripts:
    -> ```src/evaluation/main_data_ft_transformer_next_state.py```
    -> ```src/evaluation/main_data_ft_transformer_next_state_constraint_cvxpy.py```
    -> ```src/evaluation/main_data_ft_transformer_model.py```
    -> ```src/evaluation/main_data_ft_transformer_model_constraint_cvxpy.py```
    -> ```src/evaluation/main_data_optimizer_rnav.py```
    -> ```src/evaluation/main_data_resnet_101_model.py```
    -> ```src/evaluation/main_data_resnet_101_model_constraint_cvxpy.py```
    -> ```src/evaluation/main_data_resnet_101_next_state.py```
    -> ```src/evaluation/main_data_resnet_101_next_state_constraint_cvxpy.py```
4. after all scripts have finished, run ```src/evaluation/trajectory_evaluator_all.py``` to see the results saved under ````src/data/summary_experiments.csv`. (also adjust here the dataset_name to the test file name)

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

## geoid correction

see: https://en.wikipedia.org/wiki/Geoid

| Model     | Year     | Resolution |
|-----------|----------|------------|
| EGM96     | 1996     | ~30 km     |
| EGM2008   | 2008     | ~10 km     |
| EGM08_25  | aus 2008 | ~5 km      |