import ctypes
from pathlib import Path

import numpy as np
import pygame
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_CLAMP_TO_EDGE,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LINEAR,
    GL_RGBA,
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_TRUE,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glBindBuffer,
    glBindTexture,
    glBindVertexArray,
    glBufferData,
    glDeleteBuffers,
    glDeleteTextures,
    glDeleteVertexArrays,
    glDrawArrays,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetUniformLocation,
    glTexImage2D,
    glTexParameteri,
    glUniform1i,
    glUniform2f,
    glUniform4f,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
)
from OpenGL.GL.shaders import compileProgram, compileShader, ShaderProgram
from numpy.typing import NDArray
from typing_extensions import override

from engine.utils.color import Color
from engine.utils.component import Component
from engine.utils.renderer import Renderer
from engine.utils.simplematerial import load_shader_source

# Layout: x  y  u  v  (per 4 floats)
_QUAD_DATA = np.array([
    -0.5, -0.5,  0.0, 1.0,
     0.5, -0.5,  1.0, 1.0,
     0.5,  0.5,  1.0, 0.0,
    -0.5, -0.5,  0.0, 1.0,
     0.5,  0.5,  1.0, 0.0,
    -0.5,  0.5,  0.0, 0.0,
], dtype=np.float32)


class TextRenderer(Component, Renderer):
    _UTILS_DIR = Path(__file__).resolve().parent
    _SHADER_DIR = _UTILS_DIR / "shaders"

    _VERT = load_shader_source(_SHADER_DIR / "text.vert")
    _FRAG = load_shader_source(_SHADER_DIR / "text.frag")

    _program: ShaderProgram | None = None

    def __init__(
        self,
        text: str = "Text",
        font_size: int = 48,
        color: Color | None = None,
        billboard: bool = True,
        world_height: float = 0.5,
        font_name: str | None = None,
    ) -> None:
        self.text:         str            = text
        self.font_size:    int            = font_size
        self.color:        Color          = color or Color.white()
        self.billboard:    bool           = billboard
        self.world_height: float          = world_height
        self.font_name:    str | None     = font_name

        self._vao:    int | None    = None
        self._vbo:    int | None    = None
        self._tex_id: int | None    = None
        self._tex_w:  int           = 1
        self._tex_h:  int           = 1
        self._dirty:  bool          = True

    def set_text(self, text: str) -> None:
        if text != self.text:
            self.text   = text
            self._dirty = True
            self._destroy_texture()

    def set_color(self, color: Color) -> None:
        self.color  = color
        self._dirty = True
        self._destroy_texture()

    def _destroy_texture(self) -> None:
        if self._tex_id is not None:
            glDeleteTextures(1, [self._tex_id])
            self._tex_id = None

    def _upload_texture(self) -> None:
        if not self._dirty and self._tex_id is not None:
            return

        self._destroy_texture()

        if not pygame.font.get_init():
            pygame.font.init()

        font = pygame.font.SysFont(self.font_name, self.font_size)

        surf = font.render(self.text, True, (255, 255, 255))
        surf = surf.convert_alpha()

        w, h = surf.get_size()
        self._tex_w = w
        self._tex_h = h

        raw = pygame.image.tostring(surf, "RGBA", False)

        self._tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._tex_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._dirty = False

    def _upload_quad(self) -> None:
        if self._vao is not None:
            return

        stride = 4 * 4   # 4 floats × 4 Bytes

        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, _QUAD_DATA.nbytes, _QUAD_DATA, GL_STATIC_DRAW)

        # location 0 → aPos (xy)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        # location 1 → aUV (uv)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))

        glBindVertexArray(0)

    @classmethod
    def _get_program(cls) -> int:
        if cls._program is None:
            vert = compileShader(cls._VERT, GL_VERTEX_SHADER)
            frag = compileShader(cls._FRAG, GL_FRAGMENT_SHADER)
            cls._program = compileProgram(vert, frag)
        return cls._program

    @override
    def draw(
        self,
        view_mat:  NDArray[np.float32],
        proj_mat:  NDArray[np.float32],
        scene:     "Scene",
    ) -> None:
        self._upload_texture()
        self._upload_quad()

        aspect  = self._tex_w / max(self._tex_h, 1)
        world_w = self.world_height * aspect
        world_h = self.world_height

        prog = self._get_program()
        glUseProgram(prog)

        model = self.owner.transform.matrix()
        glUniformMatrix4fv(glGetUniformLocation(prog, "uModel"),      1, GL_TRUE, model)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uView"),       1, GL_TRUE, view_mat)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uProjection"), 1, GL_TRUE, proj_mat)

        glUniform2f(glGetUniformLocation(prog, "uSize"), world_w, world_h)
        glUniform1i(glGetUniformLocation(prog, "uBillboard"), 1 if self.billboard else 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._tex_id)
        glUniform1i(glGetUniformLocation(prog, "uTexture"), 0)

        c = self.color
        glUniform4f(glGetUniformLocation(prog, "uColor"), c.r, c.g, c.b, c.a)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)

    @override
    def on_destroy(self) -> None:
        if self._vao is not None:
            glDeleteVertexArrays(1, [self._vao])
            glDeleteBuffers(1, [self._vbo])
            self._vao = None
        self._destroy_texture()

    @override
    def on_update(self, dt: float):
        pass

    @override
    def on_start(self) -> None:
        pass