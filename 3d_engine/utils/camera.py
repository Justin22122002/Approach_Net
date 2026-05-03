import math

import pygame

from utils.vec3 import Vec3
from utils.helpers import _mat4_perspective, _mat4_look_at
import numpy as np

class Camera:
    def __init__(self, fov=75.0, near=0.1, far=1000.0):
        self.fov  = fov
        self.near = near
        self.far  = far
        self.position = Vec3(0, 2, 6)

    def get_view(self) -> np.ndarray:
        raise NotImplementedError

    def get_projection(self, width, height) -> np.ndarray:
        aspect = width / max(height, 1)
        return _mat4_perspective(self.fov, aspect, self.near, self.far)

    def handle_event(self, event): pass
    def update(self, dt, keys): pass


class FlyCamera(Camera):
    """WASD + mouse look free-fly camera."""
    def __init__(self, speed=5.0, sensitivity=0.15, **kw):
        super().__init__(**kw)
        self.speed       = speed
        self.sensitivity = sensitivity
        self.yaw         = -90.0
        self.pitch       = 0.0
        self._first_mouse = True
        self._last_mx     = 0
        self._last_my     = 0
        self._captured    = False

    def _front(self) -> Vec3:
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        return Vec3(math.cos(p)*math.cos(y),
                    math.sin(p),
                    math.cos(p)*math.sin(y)).normalized()

    def get_view(self) -> np.ndarray:
        f = self._front()
        return _mat4_look_at(self.position, self.position + f, Vec3.up())

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            self._captured = True
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
        if event.type == pygame.MOUSEBUTTONUP and event.button == 3:
            self._captured = False
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
        if event.type == pygame.MOUSEMOTION and self._captured:
            dx, dy = event.rel
            self.yaw   += dx * self.sensitivity
            self.pitch  = max(-89, min(89, self.pitch - dy * self.sensitivity))

    def update(self, dt, keys):
        f = self._front()
        r = f.cross(Vec3.up()).normalized()
        spd = self.speed * dt
        if keys[pygame.K_w] or keys[pygame.K_UP]:    self.position = self.position + f * spd
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:  self.position = self.position - f * spd
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:  self.position = self.position - r * spd
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: self.position = self.position + r * spd
        if keys[pygame.K_q]:                          self.position = self.position - Vec3.up() * spd
        if keys[pygame.K_e]:                          self.position = self.position + Vec3.up() * spd

