# Proposal Title: Multi-Agent Generative Modeling with Hard Constraints for Air Traffic Control

## Problem and Motivation
Air Traffic Control (ATC) is a highly safety-critical domain in which human operators are required to make rapid, 
sequential decisions under time pressure. Air traffic controllers continuously monitor multiple aircraft, 
predict their trajectories, and issue instructions to ensure safe separation and efficient traffic flow. 
Due to the dynamic and complex nature of airspace environments, 
even minor errors or delays in decision-making can lead to severe consequences, 
including loss of separation or, in the worst case, collisions.

The motivation of this work is to develop a model capable of learning and generating air traffic control decisions.
The system is designed to analyze the current air traffic situation and predict safe and efficient control actions in real time, 
represented as motion vectors for individual aircraft.
While the primary focus of this work is on building such a decision-making model, 
a long-term objective is to integrate it as an intelligent assistance system for human air traffic controllers.
In this context, the model could support controllers by providing decision recommendations, improving situational awareness, and reducing workload.

## related work e.g. state of the art
Current approaches either rely on probabilistic safety estimates or deterministic models that do not scale well to complex scenarios.
While probabilistic models can achieve high accuracy, they cannot guarantee absolute safety, 
which is critical in safety-sensitive domains such as air traffic control.
Recent research explores methods that embed safety constraints directly into the system design, 
like enabling provable guarantees of collision avoidance or other safety properties.

## Goal of this work
The goal of this work is to develop a system where safety constraints are embedded into the model architecture, 
ensuring that unsafe states (e.g., collisions) are unreachable. 
The system focuses on a single airport environment, where all aircraft within a specified radius around the airport are considered, 
but the model is designed to be extensible to other airports.
These constraints include RNAV approach plates with their minimum and maximum altitudes at designated approach points, 
aircraft speed limits during runway landing, terrain avoidance, and required separation between aircraft.
While the proposed model operates in discrete timesteps and predicts an optimal motion vector for each aircraft, 
continuous-time formulations represent a promising extension to more accurately capture the dynamic nature of air traffic systems.
The main data for training is collected from real-world data, but also synthetic data
can be used for creating conflict scenarios, like two planes heading towards the same point.

## Methodology
The methodology of this work consists of three main parts: data acquisition, model design, and evaluation.

### data acquisition:
Aircraft data will be collected from OpenSky. Airport data such as runways will be obtained from OpenStreetMap, 
FAA (ARINC 424), flight planning databases, or other APIs. 
RNAV approach templates will be collected from FAA sources (ARINC 424), the airac.net API, or published approach charts. 
Terrain data will be obtained from the Copernicus Data Space.

### model design / architecture:
The model architecture can be based on Graph Neural Networks, Transformers, or a combination of both, 
extended with constraint layers. 
The model integrates multiple data sources, including regulatory constraints (e.g., separation minima, obstacle avoidance), 
geometric data (e.g., terrain and airport layouts), and historical flight information (e.g., ADS-B trajectories).
Based on these inputs, the model predicts an optimal three-dimensional motion vector for each aircraft, 
while strictly enforcing safety constraints within the model architecture.

### evaluation criteria:
The evaluation criteria will be based on the following metrics:

  1. minimal separation
  2. Trajectory Prediction Accuracy (Comparison to real-world trajectories alone may not be enough, as these trajectories do not necessarily represent optimal or ideal behavior)
     - Accuracy of predicted trajectories compared to actual aircraft trajectories.
     - Metrics: RMSE, MAE, or deviation in horizontal and vertical directions
  3. Constraint Adherence
     - Minimum horizontal & vertical separation
     - Compliance with RNAV approach altitudes
     - Terrain avoidance
     - Speed limits during runway landing
  4. Operational Metrics
     - Runway throughput
     - Total time to detect and resolve conflicts
  5. Model inference time
  6. time comparison, e.g. real atc vs model performance
