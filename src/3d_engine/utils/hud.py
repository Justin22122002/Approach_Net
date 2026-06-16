from pathlib import Path
from typing import Callable
import pygame

from OpenGL.GL import (
    GL_BLEND,
    GL_CULL_FACE,
    GL_DEPTH_TEST,
    GL_FRAGMENT_SHADER,
    GL_LINEAR,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLES,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glBindTexture,
    glBindVertexArray,
    glDisable,
    glDrawArrays,
    glEnable,
    glBlendFunc,
    glGenTextures,
    glGenVertexArrays,
    glGetUniformLocation,
    glTexImage2D,
    glTexParameteri,
    glUniform1i,
    glUseProgram,
)
from OpenGL.GL.shaders import compileProgram, compileShader, ShaderProgram

from utils.simplematerial import load_shader_source

RGBColor  = tuple[int, int, int]
RGBAColor = tuple[int, int, int, int]
AnyColor  = RGBColor | RGBAColor

_ButtonRecord = dict  # {x, y, w, h, callback, hovered}

class HUD:
    _HUD_VERT = load_shader_source(Path("utils/shaders/hud.vert"))
    _HUD_FRAG = load_shader_source(Path("utils/shaders/hud.frag"))

    def __init__(self, engine: "Engine") -> None:
        self._engine                          = engine
        self._surface: pygame.Surface | None = None
        self._tex_id:  int | None            = None
        self._prog:    ShaderProgram | None  = None
        self._vao:     int | None            = None
        self._fonts:     dict[int, pygame.font.Font]    = {}
        self._img_cache: dict[str, pygame.Surface]      = {}

        self._buttons: list[_ButtonRecord] = []
        self._mouse_pos: tuple[int, int]   = (0, 0)

    def _ensure_gpu(self):
        if self._prog is not None:
            return
        vert = compileShader(self._HUD_VERT, GL_VERTEX_SHADER)
        frag = compileShader(self._HUD_FRAG, GL_FRAGMENT_SHADER)
        self._prog  = compileProgram(vert, frag)
        self._vao   = glGenVertexArrays(1)
        self._tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _ensure_surface(self):
        w, h = self._engine.width, self._engine.height
        if self._surface is None or self._surface.get_size() != (w, h):
            self._surface = pygame.Surface((w, h), pygame.SRCALPHA)

    def _get_font(self, size: int) -> pygame.font.Font:
        if size not in self._fonts:
            self._fonts[size] = pygame.font.SysFont("segoeui", size)
        return self._fonts[size]

    def _begin(self):
        self._ensure_gpu()
        self._ensure_surface()
        self._surface.fill((0, 0, 0, 0))
        self._buttons.clear()
        self._mouse_pos = pygame.mouse.get_pos()

    def _end(self):
        surf = self._surface
        w, h = surf.get_size()
        raw  = pygame.image.tostring(surf, "RGBA", False)

        # Upload texture
        glBindTexture(GL_TEXTURE_2D, self._tex_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, raw)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Draw fullscreen quad with HUD shader
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glUseProgram(self._prog)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex_id)
        glUniform1i(glGetUniformLocation(self._prog, "uTex"), 0)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)

        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

    def _handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            for btn in self._buttons:
                if btn["x"] <= mx < btn["x"] + btn["w"] and \
                   btn["y"] <= my < btn["y"] + btn["h"]:
                    btn["callback"]()
                    break   # consume the first hit

    def button(
        self,
        label: str,
        x: int, y: int, w: int, h: int,
        callback: Callable[[], None],
        *,
        size: int = 18,
        color:        AnyColor = (60, 90, 160, 220),
        text_color:   AnyColor = (255, 255, 255, 255),
        border_color: AnyColor = (100, 130, 210, 255),
        hover_color:        AnyColor = (80, 120, 200, 240),
        hover_text_color:   AnyColor = (255, 255, 255, 255),
        hover_border_color: AnyColor = (160, 190, 255, 255),
        border_radius: int = 6,
        border_width:  int = 2,
    ) -> None:

        mx, my = self._mouse_pos
        hovered = x <= mx < x + w and y <= my < y + h

        bg     = hover_color        if hovered else color
        txt_c  = hover_text_color   if hovered else text_color
        brd_c  = hover_border_color if hovered else border_color

        self.rect(x, y, w, h,
                  color=bg,
                  border_color=brd_c,
                  border_radius=border_radius,
                  border_width=border_width)

        self.text_centered(label, x, y, w, h, size=size, color=txt_c)

        self._buttons.append({
            "x": x, "y": y, "w": w, "h": h,
            "callback": callback,
        })

    def rect(
        self,
        x: int, y: int, w: int, h: int,
        color: AnyColor = (0, 0, 0, 180),
        border_radius: int = 0,
        border_color: AnyColor | None = None,
        border_width: int = 2,
    ) -> None:
        c = color if len(color) == 4 else (*color, 255)
        pygame.draw.rect(self._surface, c, (x, y, w, h),
                         border_radius=border_radius)
        if border_color:
            bc = border_color if len(border_color) == 4 else (*border_color, 255)
            pygame.draw.rect(self._surface, bc, (x, y, w, h),
                             width=border_width, border_radius=border_radius)

    def text(
        self,
        txt: str,
        x: int, y: int,
        size: int = 18,
        color: AnyColor = (255, 255, 255),
        shadow: bool = False,
        shadow_color: AnyColor = (0, 0, 0),
    ) -> None:
        font = self._get_font(size)
        c    = color if len(color) == 4 else (*color, 255)
        if shadow:
            sc   = shadow_color if len(shadow_color) == 4 else (*shadow_color, 200)
            surf = font.render(txt, True, sc[:3])
            self._surface.blit(surf, (x+1, y+1))
        surf = font.render(txt, True, c[:3])
        self._surface.blit(surf, (x, y))

    def text_centered(
        self,
        txt: str,
        x: int, y: int, w: int, h: int,
        size: int = 18,
        color: AnyColor = (255, 255, 255),
    ) -> None:
        font = self._get_font(size)
        surf = font.render(txt, True, color[:3] if len(color) >= 3 else color)
        tx   = x + (w - surf.get_width())  // 2
        ty   = y + (h - surf.get_height()) // 2
        self._surface.blit(surf, (tx, ty))

    def image(
        self,
        source: str | pygame.Surface,
        x: int, y: int,
        w: int | None = None,
        h: int | None = None,
        alpha: int = 255,
    ) -> None:
        if isinstance(source, str):
            if source not in self._img_cache:
                self._img_cache[source] = pygame.image.load(source).convert_alpha()
            surf = self._img_cache[source]
        else:
            surf = source
        if w is not None and h is not None:
            surf = pygame.transform.scale(surf, (w, h))
        if alpha != 255:
            surf = surf.copy()
            surf.set_alpha(alpha)
        self._surface.blit(surf, (x, y))

    def line(
        self,
        x1: int, y1: int, x2: int, y2: int,
        color: AnyColor = (255, 255, 255),
        width: int = 1,
    ) -> None:
        pygame.draw.line(self._surface, color, (x1, y1), (x2, y2), width)

    def circle(
        self,
        x: int, y: int, radius: int,
        color: AnyColor = (255, 255, 255, 255),
        border_color: AnyColor | None = None,
        border_width: int = 2,
    ) -> None:
        c = color if len(color) == 4 else (*color, 255)
        pygame.draw.circle(self._surface, c, (x, y), radius)
        if border_color:
            bc = border_color if len(border_color) == 4 else (*border_color, 255)
            pygame.draw.circle(self._surface, bc, (x, y), radius, border_width)

    def crosshair(
        self,
        size: int = 10,
        thickness: int = 2,
        color: AnyColor = (255, 255, 255, 200),
        gap: int = 4,
    ) -> None:
        cx = self._engine.width  // 2
        cy = self._engine.height // 2
        c  = color if len(color) == 4 else (*color, 255)
        pygame.draw.rect(self._surface, c,
                         (cx - size - gap, cy - thickness//2,
                          size, thickness))
        pygame.draw.rect(self._surface, c,
                         (cx + gap, cy - thickness//2,
                          size, thickness))
        pygame.draw.rect(self._surface, c,
                         (cx - thickness//2, cy - size - gap,
                          thickness, size))
        pygame.draw.rect(self._surface, c,
                         (cx - thickness//2, cy + gap,
                          thickness, size))

    def panel(
        self,
        x: int, y: int, w: int, h: int,
        bg: AnyColor = (20, 20, 30, 210),
        border: AnyColor = (80, 80, 100, 255),
        border_radius: int = 6,
        border_width: int = 2,
    ) -> None:
        self.rect(x, y, w, h, color=bg,
                  border_color=border, border_radius=border_radius,
                  border_width=border_width)

    def bar(
        self,
        x: int, y: int, w: int, h: int,
        value: float,           # 0.0 – 1.0
        fg_color:  AnyColor = (80, 200, 80, 255),
        bg_color:  AnyColor = (40, 40, 40, 200),
        border_radius: int = 4,
    ) -> None:
        self.rect(x, y, w, h, color=bg_color, border_radius=border_radius)
        filled = max(0, min(1, value))
        if filled > 0:
            self.rect(x, y, int(w * filled), h,
                      color=fg_color, border_radius=border_radius)

    @property
    def width(self) -> int:  return self._engine.width
    @property
    def height(self) -> int: return self._engine.height