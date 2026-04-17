# Proposal (Constraint-Aware Deep Learning for Safe Air Traffic Control Assistance)

## Problem and Motivation
Air Traffic Control (ATC) is a highly safety-critical domain in which human operators are required to make rapid, 
sequential decisions under time pressure. Air traffic controllers continuously monitor multiple aircraft, 
predict their trajectories, and issue instructions to ensure safe separation and efficient traffic flow. 
Due to the dynamic and complex nature of airspace environments, 
even minor errors or delays in decision-making can lead to severe consequences, 
including loss of separation or, in the worst case, collisions.

The motivation of this work is to enhance the decision-making capabilities of air traffic controllers
through intelligent assistance. Rather than replacing human operators, the goal is to provide a system
that continuously analyzes the current air traffic situation and suggests optimal control actions in real time.
In particular, it is designed to propose safe and efficient motion vectors for individual aircraft, 
allowing controllers to make informed decisions more quickly.
By acting as an additional safety layer, such a system can help mitigate human error,
improve situational awareness, and contribute to safer and more efficient air traffic management.

## related work e.g. state of the art
Current approaches either rely on probabilistic safety estimates or deterministic models that do not scale well to complex scenarios.
While probabilistic models can achieve high accuracy, they cannot guarantee absolute safety, 
which is critical in safety-sensitive domains such as air traffic control.
Recent research explores methods that embed safety constraints directly into the system design, 
enabling provable guarantees of collision avoidance.

## Goal of this work
The goal of this work is to develop a system where safety constraints are embedded into the model architecture, 
ensuring that unsafe states (e.g., collisions) are unreachable. 
The system focuses on a single airport environment, where all aircraft within a specified radius around the airport are considered.

These constraints include RNAV approach templates with their minimum and maximum altitudes at designated approach points, 
aircraft speed limits during runway landing, terrain avoidance, and required separation between aircraft.
The model should predict the optimal vector for each aircraft in each timestep, 
accounting for the constraints and real-time constraints. The main data for training is collected from real world data, but also synthetic data
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
extended with constraint layers. It takes as input aircraft states and separation information, trajectories, airport data, 
and terrain information, and outputs a three-dimensional motion vector for each aircraft or signs of potential conflicts. 
Safety constraints, such as minimum separation and terrain avoidance, are directly integrated into the model architecture.

### evaluation criteria:
The evaluation criteria will be based on the following metrics:

  1. minimal separation
  2. Trajectory Prediction Accuracy
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
