
# notes for derivation of the state kinematics

## for  $\chi$
$
\dot{\chi} = \frac{g \cdot tan(\phi)}{v}
$

## for  $\gamma$

### Definition for $\gamma$
$
\gamma = \arcsin(\frac{v_z}{v})
$ with 
$
v = \sqrt{v_x^2 + v_y^2 + v_z^2}
$

### goal for $\dot{\gamma}$
$
s = \frac{v_z}{v} \rightarrow \gamma = \arcsin(s)
$

$
\dot{\gamma} = \frac{1}{\sqrt{1 - s^2}} \cdot \dot{s}
$

### derivation of $s = \frac{v_z}{v}$
$
\dot{s} = \frac{\dot{v_z} v - v_z \dot{v}}{v^2}
$

### plug in $\dot{\gamma}$

$
\dot{\gamma} = \frac{1}{\sqrt{1 - \frac{v_z}{v}^2}} \cdot \frac{\dot{v_z} v - v_z \dot{v}}{v^2}
$

### simplification
$
\sqrt{1 - \frac{v_z}{v}^2} = \frac{v^2 - v_z^2}{v}
$

$
\sqrt{v^2 - v_z^2} = \sqrt{v_x^2 + v_y^2} = v_z
$

### final form
$
\boxed{
\dot{\gamma}
=
\frac{v \, \dot{v}_z - v_z \, \dot{v}}{v^2 \cos(\gamma)}
}
$

$
\boxed{
\dot{\gamma}
=
\frac{v \, \dot{v}_z - v_z \, \dot{v}}{v_h^2}
}
$ or 
$
\boxed{
\dot{\gamma}
=
\frac{v \, \dot{v}_z - v_z \, \dot{v}}{v^2 \cos(\gamma)}
}
$


## inverse solution 

### start
$
\dot{v} = \frac{T - D}{m} \cdot \frac{\vec{v}}{\|\vec{v}\|} + \frac{L}{m} \cdot \vec{n} + \vec{g}
$

$
\vec{t} = \frac{\vec{v}}{\|\vec{v}\|}
$

### required acceleration based on the goal state

velocity:
$
\vec{v_{req}} = \frac{\vec{p_g} - \vec{p_0}}{T_f}
$

acceleration:
$
\dot{\vec{v_{req}}} = \frac{\vec{v_{req}} - \vec{v_0}}{T_f}
$

### projection on flight direction

tangential direction:
$
\vec{t} = \frac{\vec{v}}{\|\vec{v}\|}
$

normal direction: $\vec{n}$, so 

$
\dot{\vec{v}} = a_t \vec{t} + a_n \vec{n} + \vec{g}
$

### control equation (inverse form):

**thrust**: scalar projection on $\vec{t}$:

$
T = m \left( \dot{\vec{v}} \cdot \vec{t} \right) + D
$

**load factor** $n$:

normal direction:
$
L = m \left( \dot{\vec{v}} - \vec{g} \right) \cdot \vec{n}
$

also see above: $n = \frac{L}{m g}$

therefore:

$
n = \frac{(\dot{\vec{v}} - \vec{g}) \cdot \vec{n}}{g}
$

**roll angle** $\phi$:

$
\dot{\chi} = \frac{g \cdot tan(\phi)}{v}
$ 

so:

$
\phi = \arctan{(\frac{v \dot{\chi}}{g})}
$ 

### solution
Thrust
$
\boxed{
T = m \left( \dot{\vec{v}} \cdot \vec{t} \right) + D
}
$

Load factor
$
\boxed{
n = \frac{(\dot{\vec{v}} - \vec{g}) \cdot \vec{n}}{g}
}
$

Roll angle
$
\boxed{
\phi = \arctan\left(\frac{v \, \dot{\chi}}{g}\right)
}
$

$
C_L = \frac{2 n m g}{\rho v^2 S}
$

$
C_D = C_{D0} + k C_L^2
$