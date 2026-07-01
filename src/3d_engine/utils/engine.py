from collections.abc import Callable

import numpy as np
import pygame

from OpenGL.GL import (
    GL_BACK,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_MULTISAMPLE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    glBlendFunc,
    glClear,
    glClearColor,
    glCullFace,
    glEnable,
    glViewport
)
from pygame import Surface
import time as t
from pygame.time import Clock
import math

from utils.gizmo import TransformGizmo
from utils.gameObject import GameObject
from utils.interactable import Interactable
from utils.boxCollider import BoxCollider
from utils.raycast import screen_to_ray, ray_aabb_intersect

from utils.color import Color
from utils.camera import Camera, FlyCamera
from utils.hud import HUD
from utils.scene import Scene
from utils.input import Input
from utils.sky import Sky


class Engine:
    """
    Usage:
        engine = Engine(800, 600, "My Game")
        engine.run()

    Lifecycle callbacks (attach to engine):
        engine.on_start   = lambda: ...
        engine.on_update  = lambda dt: ...
        engine.on_gui     = lambda: ...          # called after scene draw
        engine.on_quit    = lambda: ...
    """

    def __init__(
            self,
            width: int = 800,
            height: int = 600,
            title: str = "3d engine",
            fps: int = 60,
            clear_color: Color | None = None,
    ) -> None:
        self.width: int         = width
        self.height: int        = height
        self.title: str         = title
        self.target_fps: int    = fps
        self.clear_color: Color = clear_color or Color(0.1, 0.1, 0.15, 1.0)

        self.scene: Scene       = Scene()
        self.camera: Camera     = FlyCamera()
        self.input: Input       = Input()
        self.time: float        = 0.0       # total elapsed seconds
        self.frame: int         = 0
        self.sky: Sky | None = None
        self._hovered: GameObject | None = None
        self.gizmo: TransformGizmo = TransformGizmo()

        self.rt_mode: bool = False

        # lifecycle hooks (assign lambdas or functions)
        self.on_start:  Callable[[], None] | None      = None
        self.on_update: Callable[[float], None] | None = None
        self.on_gui:    Callable[[], None] | None      = None
        self.on_quit:   Callable[[], None] | None      = None

        self._running: bool              = False
        self._window: Surface | None  = None
        self._clock: Clock | None     = None
        self.hud: HUD = HUD(self)

    def _init_pygame(self) -> None:
        pygame.init()

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)

        print(f"Opening window: {self.width}x{self.height}  '{self.title}'")

        self._window = pygame.display.set_mode(
            (self.width, self.height),
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        )

        pygame.display.set_caption(self.title)
        self._clock = pygame.time.Clock()

        print(f"OpenGL 3.3 Core context created")

    def _init_opengl(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        cc = self.clear_color
        glClearColor(cc.r, cc.g, cc.b, cc.a)

    def run(self) -> None:
        self._init_pygame()
        self._init_opengl()

        print(f"Scene starting...")

        # user on_start
        if self.on_start:
            self.on_start()
        self.scene._start_all()

        self._running = True
        prev = t.perf_counter()

        while self._running:
            now = t.perf_counter()
            dt  = min(now - prev, 0.1)  # cap dt to avoid spiral of death
            prev = now
            self.time  += dt
            self.frame += 1

            # events
            view = self.camera.get_view()
            proj = self.camera.get_projection(self.width, self.height)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                    self.gizmo.mode = "rotate" if self.gizmo.mode == "translate" else "translate"
                if event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    origin, direction = screen_to_ray(
                        mx, my, self.width, self.height, view, proj, self.camera.position
                    )
                    if not self.gizmo.try_start_drag(origin, direction):
                        if self._hovered is not None:
                            ia = self._hovered.get_component(Interactable)
                            if ia:
                                ia._handle_click()
                        else:
                            self.gizmo.detach()

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.gizmo.end_drag()

                if event.type == pygame.MOUSEMOTION and self.gizmo.is_dragging:
                    mx, my = event.pos
                    origin, direction = screen_to_ray(
                        mx, my, self.width, self.height, view, proj, self.camera.position
                    )
                    self.gizmo.update_drag(origin, direction)

                self.input._process_event(event)
                if not self.gizmo.is_dragging:
                    self.camera.handle_event(event)
                self.hud._handle_event(event)

            keys = pygame.key.get_pressed()

            if not self.gizmo.is_dragging:
                self.camera.update(dt, keys)
            self.scene._update_all(dt)
            if self.on_update:
                self.on_update(dt)

            # render
            glViewport(0, 0, self.width, self.height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            view = self.camera.get_view()
            proj = self.camera.get_projection(self.width, self.height)

            self._update_picking(view, proj)

            if self.sky:
                self.sky.render(
                    view_mat=view,
                    proj_mat=proj,
                    time=self.time,
                    width=self.width,
                    height=self.height,
                    cam_pos=self.camera.position
                )

            self.scene._draw_all(view, proj)
            self.gizmo.draw(view, proj)

            self.hud._begin()
            if self.on_gui:
                self.on_gui()
            self.hud._end()

            pygame.display.flip()
            self.input._end_frame()
            self._clock.tick(self.target_fps)

        if self.on_quit:
            self.on_quit()
        pygame.quit()

    def quit(self) -> None:
        self._running = False

    def _update_picking(self, view: np.ndarray, proj: np.ndarray) -> None:
        mx, my = pygame.mouse.get_pos()
        origin, direction = screen_to_ray(
            mx, my, self.width, self.height, view, proj, self.camera.position
        )

        closest_obj = None
        closest_t = math.inf
        for go in self.scene._objects:
            if not go.active:
                continue
            ia = go.get_component(Interactable)
            col = go.get_component(BoxCollider)
            if not ia or not col:
                continue
            t = ray_aabb_intersect(origin, direction, col.world_min(), col.world_max())
            if t is not None and t < closest_t:
                closest_t = t
                closest_obj = go

        if closest_obj is not self._hovered:
            if self._hovered is not None:
                old_ia = self._hovered.get_component(Interactable)
                if old_ia:
                    old_ia._set_hovered(False)
            self._hovered = closest_obj
            if self._hovered is not None:
                new_ia = self._hovered.get_component(Interactable)
                if new_ia:
                    new_ia._set_hovered(True)