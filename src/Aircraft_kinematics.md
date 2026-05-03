# Aircraft Kinematics — Equations of Motion

## 1. State Vector and Aircraft Maximum Constraints

### 1.1 State Vector
Each aircraft is represented as a point in space with the following state vector:

$$
\mathbf{s}(t) = \begin{pmatrix} x(t) \\ y(t) \\ h(t) \\ v(t) \\ \gamma(t) \\ \chi(t) \end{pmatrix}
$$

| Symbol   | Description                                                       | Unit                                                        |
|----------|-------------------------------------------------------------------|-------------------------------------------------------------|
| $x$      | East position (Cartesian)                                         | $[\text{m}]$                                                |
| $y$      | North position (Cartesian)                                        | $[\text{m}]$                                                |
| $h$      | Altitude (MSL)                                                    | $[\text{m}]$                                                |
| $v$      | True airspeed                                                     | $[\text{m/s}]$ or $[\text{kts}]$ $\Rightarrow$ scalar value |
| $\gamma$ | Flight path angle (Flugbahnwinkel)                                | $[°]$                                                       |
| $\chi$   | Track angle / Course angle (Kursbahnwinkel) from North, clockwise | $[°]$                                                       |

> **Assumption (coordinated flight):** The aircraft is modelled as a point mass. No sideslip angle — the nose always points in the direction of motion. Therefore no separate heading angle is needed. The track $\chi$ is fully determined by the start–end geometry.

### 1.2 Aircraft Maximum Constraints
| Symbol         | Description                                    | Unit                                                        |
|----------------|------------------------------------------------|-------------------------------------------------------------|
| $vmin$         | min speed                                      | $[\text{m/s}]$                                              |
| $vmax$         | max speed                                      | $[\text{m/s}]$                                              |
| $rocmax$       | max rate of climb                              | $[\text{m/s}]$                                              |
| $rodmax$       | max rate of decent                             | $[\text{m/s}]$ or $[\text{kts}]$ $\Rightarrow$ scalar value |
| $gammadotmaxg$ | max change of the flight path angle per second | $[°/s]$                                                     |
| $chidotmax$    | max change of course angle per second          | $[°/s]$                                                     |


---

## 2. Angle Definitions

### 2.1 Flight Path Angle $\gamma$ (vertical plane)

The flight path angle describes the inclination of the velocity vector relative to the horizontal:

$$
\gamma > 0 \quad \Rightarrow \quad \text{climb (Steigflug)}
$$
$$
\gamma < 0 \quad \Rightarrow \quad \text{descent (Sinkflug)}
$$
$$
\gamma = 0 \quad \Rightarrow \quad \text{level flight (Horizontalflug)}
$$

Vertical and horizontal speed components:

$$
v_z = v \cdot \sin(\gamma) \qquad \text{(Rate of Climb / Descent per time)} \;\; [m/s]
$$

$$
v_{h} = v \cdot \cos(\gamma) \qquad \text{(horizontal ground speed component)} \;\; [m/s]
$$

### 2.2 Track Angle $\chi$ (horizontal plane)

The track angle is measured clockwise from True North:

$$
\chi = 0° \Rightarrow \text{North}, \quad \chi = 90° \Rightarrow \text{East}, \quad \chi = 180° \Rightarrow \text{South}, \quad \chi = 270° \Rightarrow \text{West}
$$

Decomposition of horizontal speed into East / North components:

$$
v_E = v \cdot \cos(\gamma) \cdot \sin(\chi)
$$

$$
v_N = v \cdot \cos(\gamma) \cdot \cos(\chi)
$$

---

## 3. Equations of Motion (Continuous Time)

For a flat-Earth approximation in Cartesian coordinates $(x, y, h)$ with $x$ = East, $y$ = North:

$$
\dot{x}(t) = v \cdot \cos(\gamma) \cdot \sin(\chi)
$$

$$
\dot{y}(t) = v \cdot \cos(\gamma) \cdot \cos(\chi)
$$

$$
\dot{h}(t) = v \cdot \sin(\gamma)
$$

The speed $v$ and angles $\gamma$, $\chi$ are treated as **control inputs** (actions) of the model. Within a single discrete timestep they are held constant (zero-order hold).

---

## 4. Discrete-Time Propagation

The model operates in discrete timesteps of duration $\Delta t$. A trajectory from start to goal consists of exactly $N$ steps, giving a total time horizon of:

$$
T = N \cdot \Delta t
$$

The action $(v, \gamma, \chi)$ is held **constant across all $N$ steps** (zero-order hold). Applying the Euler integration once per step:

$$
x(t + \Delta t) = x(t) + v \cdot \cos(\gamma) \cdot \sin(\chi) \cdot \Delta t
$$

$$
y(t + \Delta t) = y(t) + v \cdot \cos(\gamma) \cdot \cos(\chi) \cdot \Delta t
$$

$$
h(t + \Delta t) = h(t) + v \cdot \sin(\gamma) \cdot \Delta t
$$

Because the action is constant, each step contributes an identical displacement. After exactly $N$ steps the aircraft has travelled:

$$
\Delta x_{\text{total}} = v \cdot \cos(\gamma) \cdot \sin(\chi) \cdot T, \quad
\Delta y_{\text{total}} = v \cdot \cos(\gamma) \cdot \cos(\chi) \cdot T, \quad
\Delta h_{\text{total}} = v \cdot \sin(\gamma) \cdot T
$$

> Higher-order integrators (Runge–Kutta) are possible extensions but are not necessary for this discrete-timestep formulation.

---

## 6. Inverse Problem — Required Parameters from Start and Goal

Given a start state $\mathbf{s}_0$ and a goal position $(x_1, y_1, h_1)$ to be reached in **exactly $N$ steps**, the required constant action is derived once from the full start–goal geometry. The total time available is:

$$
T = N \cdot \Delta t
$$

### 6.1 Displacements

$$
\Delta x = x_1 - x_0, \qquad \Delta y = y_1 - y_0, \qquad \Delta h = h_1 - h_0
$$

### 6.2 Distances

$$
d_{\text{horiz}} = \sqrt{\Delta x^2 + \Delta y^2} \qquad \text{(horizontal distance)}
$$

$$
d_{3D} = \sqrt{\Delta x^2 + \Delta y^2 + \Delta h^2} \qquad \text{(3D Euclidean distance)}
$$

### 6.3 Required Speed

The aircraft must cover $d_{3D}$ in total time $T$. With a constant speed this gives:

$$
v_{\text{req}} = \frac{d_{3D}}{T} = \frac{d_{3D}}{N \cdot \Delta t}
$$

### 6.4 Required Flight Path Angle

$$
\gamma = \arctan\!\left(\frac{\Delta h}{d_{\text{horiz}}}\right)
$$

### 6.5 Required Track Angle

$$
\chi = \arctan2(\Delta x,\; \Delta y)
$$

> `arctan2` returns values in $(-180°, 180°]$. Add $360°$ if the result is negative to get the conventional $[0°, 360°)$ range.

The angles $\gamma$ and $\chi$ are **fully determined by the geometry** of start and goal and do not depend on $N$ or $\Delta t$. Only $v_{\text{req}}$ changes if the time horizon is adjusted.
