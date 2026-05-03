import math

import numpy as np
from utils.vec3 import Vec3

def _mat4_identity():
    return np.eye(4, dtype=np.float32)

def _mat4_translate(v: Vec3):
    m = _mat4_identity()
    m[0,3] = v.x; m[1,3] = v.y; m[2,3] = v.z
    return m

def _mat4_scale(v: Vec3):
    m = _mat4_identity()
    m[0,0] = v.x; m[1,1] = v.y; m[2,2] = v.z
    return m

def _mat4_rotate_x(a):
    c, s = math.cos(a), math.sin(a)
    m = _mat4_identity()
    m[1,1]=c; m[1,2]=-s; m[2,1]=s; m[2,2]=c
    return m

def _mat4_rotate_y(a):
    c, s = math.cos(a), math.sin(a)
    m = _mat4_identity()
    m[0,0]=c; m[0,2]=s; m[2,0]=-s; m[2,2]=c
    return m

def _mat4_rotate_z(a):
    c, s = math.cos(a), math.sin(a)
    m = _mat4_identity()
    m[0,0]=c; m[0,1]=-s; m[1,0]=s; m[1,1]=c
    return m

def _mat4_perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4,4), dtype=np.float32)
    m[0,0] = f / aspect
    m[1,1] = f
    m[2,2] = (far + near) / (near - far)
    m[2,3] = (2*far*near) / (near - far)
    m[3,2] = -1.0
    return m

def _mat4_look_at(eye: Vec3, center: Vec3, up: Vec3):
    f = (center - eye).normalized()
    r = f.cross(up).normalized()
    u = r.cross(f)
    m = np.array([
        [ r.x,  r.y,  r.z, -r.dot(eye)],
        [ u.x,  u.y,  u.z, -u.dot(eye)],
        [-f.x, -f.y, -f.z,  f.dot(eye)],
        [ 0,    0,    0,    1          ]
    ], dtype=np.float32)
    return m