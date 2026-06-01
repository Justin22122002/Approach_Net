import math
from abc import ABCMeta, abstractmethod

import pygame
from typing_extensions import override

from utils.vec3 import Vec3
from utils.helpers import _mat4_perspective, _mat4_look_at
import numpy as np

class Camera(metaclass=ABCMeta):
    def __init__(
            self,
            fov: float = 75.0,
            near: float = 0.1,
            far: float = 1000.0
    ) -> None:
        self.fov: float  = fov
        self.near: float = near
        self.far: float  = far
        self.position: Vec3 = Vec3(0, 2, 6)

    def get_projection(self, width: float, height: float) -> np.ndarray:
        aspect = width / max(height, 1)
        return _mat4_perspective(self.fov, aspect, self.near, self.far)

    @abstractmethod
    def handle_event(self, event): pass

    @abstractmethod
    def update(self, dt, keys): pass

    @abstractmethod
    def get_view(self) -> np.ndarray: pass

class FlyCamera(Camera):
    """WASD + mouse look free-fly camera."""
    def __init__(
            self,
            speed: float = 5.0,
            sensitivity: float = 0.15,
            **kw
    ) -> None:
        super().__init__(**kw)
        self.speed: float       = speed
        self.sensitivity: float = sensitivity
        self.yaw: float         = -90.0
        self.pitch: float       = 0.0
        self._first_mouse: bool = True
        self._last_mx: int      = 0
        self._last_my: int      = 0
        self._captured: bool    = False

    def _front(self) -> Vec3:
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        return Vec3(math.cos(p) * math.cos(y),
                    math.sin(p),
                    math.cos(p) * math.sin(y)).normalized()

    @override
    def get_view(self) -> np.ndarray:
        f = self._front()
        return _mat4_look_at(self.position, self.position + f, Vec3.up())

    @override
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

    @override
    def update(self, dt, keys):
        f = self._front()
        r = f.cross(Vec3.up()).normalized()
        spd = self.speed * dt
        if keys[pygame.K_w] or keys[pygame.K_UP]:    self.position = self.position + f * spd
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:  self.position = self.position - f * spd
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:  self.position = self.position - r * spd
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: self.position = self.position + r * spd
        if keys[pygame.K_q]:                         self.position = self.position - Vec3.up() * spd
        if keys[pygame.K_e]:                         self.position = self.position + Vec3.up() * spd


class OrbitCamera(Camera):
    """Orbit around a target point with mouse drag + scroll zoom."""

    def __init__(
            self,
            target: Vec3 = None,
            distance: float = 8.0,
            **kw
    ) -> None:
        super().__init__(**kw)
        self.target: Vec3    = target or Vec3.zero()
        self.distance: float = distance
        self.yaw: float      = 0.0
        self.pitch: float    = 20.0
        self._dragging: bool = False

    def _calc_pos(self) -> Vec3:
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        x = self.distance * math.cos(p) * math.sin(y)
        yl= self.distance * math.sin(p)
        z = self.distance * math.cos(p) * math.cos(y)
        return Vec3(self.target.x+x, self.target.y+yl, self.target.z+z)

    @override
    def get_view(self) -> np.ndarray:
        self.position = self._calc_pos()
        return _mat4_look_at(self.position, self.target, Vec3.up())

    @override
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            self._dragging = True
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
        if event.type == pygame.MOUSEBUTTONUP and event.button == 3:
            self._dragging = False
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
        if event.type == pygame.MOUSEMOTION and self._dragging:
            dx, dy = event.rel
            self.yaw   += dx * 0.4
            self.pitch  = max(-85, min(85, self.pitch + dy * 0.4))
        if event.type == pygame.MOUSEWHEEL:
            self.distance = max(0.5, self.distance - event.y * 0.5)

    @override
    def update(self, dt, keys):
        pass