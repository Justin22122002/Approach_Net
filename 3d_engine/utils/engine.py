from typing import Optional
from collections.abc import Callable

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

from utils.color import Color
from utils.camera import Camera, FlyCamera
from utils.scene import Scene
from utils.input import Input

class Engine:
    """
    The top-level engine object.

    Usage:
        engine = pygine3d.Engine(800, 600, "My Game")
        # ... set up scene ...
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
            clear_color: Optional[Color] = None,
    ) -> None:
        self.width: int         = width
        self.height: int        = height
        self.title: str         = title
        self.target_fps: int    = fps
        self.clear_color: Color = clear_color or Color(0.1, 0.1, 0.15, 1.0)

        self.scene: Scene   = Scene()
        self.camera: Camera = FlyCamera()
        self.input: Input   = Input()
        self.time: float    = 0.0       # total elapsed seconds
        self.frame: int     = 0

        # lifecycle hooks (assign lambdas or functions)
        self.on_start:  Optional[Callable[[], None]]      = None
        self.on_update: Optional[Callable[[float], None]] = None
        self.on_gui:    Optional[Callable[[], None]]      = None
        self.on_quit:   Optional[Callable[[], None]]      = None

        self._running: bool              = False
        self._window: Optional[Surface]  = None
        self._clock: Optional[Clock]     = None

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
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._running = False
                if event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                self.input._process_event(event)
                self.camera.handle_event(event)

            keys = pygame.key.get_pressed()

            # update
            self.camera.update(dt, keys)
            self.scene._update_all(dt)
            if self.on_update:
                self.on_update(dt)

            # render
            glViewport(0, 0, self.width, self.height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            view = self.camera.get_view()
            proj = self.camera.get_projection(self.width, self.height)
            self.scene._draw_all(view, proj)

            pygame.display.flip()
            self.input._end_frame()
            self._clock.tick(self.target_fps)

        # shutdown
        if self.on_quit:
            self.on_quit()
        pygame.quit()

    def quit(self) -> None:
        """Call from anywhere to stop the engine."""
        self._running = False