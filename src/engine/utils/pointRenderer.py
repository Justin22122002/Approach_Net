import ctypes
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_POINTS,
    GL_PROGRAM_POINT_SIZE,
    GL_STATIC_DRAW,
    GL_TRUE,
    GL_VERTEX_SHADER,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glGetUniformLocation,
    glUniform1f,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
)
from OpenGL.GL.shaders import compileProgram, compileShader, ShaderProgram
from numpy.typing import NDArray
from typing_extensions import override

from engine.utils.color import Color
from engine.utils.component import Component
from engine.utils.simplematerial import load_shader_source
from engine.utils.renderer import Renderer
from engine.utils.vec3 import Vec3


class PointRenderer(Component, Renderer):
    """
    renders a single or multiple points in 3D space.

    points  – list of Vec3. empty = only transform.position will be used
    color   – color of all points will be ignored when colors are set
    colors  – per point color, needs to be the same length ad points
    size    – point size ine pixel.
    """

    _UTILS_DIR = Path(__file__).resolve().parent
    _SHADER_DIR = _UTILS_DIR / "shaders"

    _VERT_SRC = load_shader_source(_SHADER_DIR / "point.vert")
    _FRAG_SRC = load_shader_source(_SHADER_DIR / "point.frag")

    _program: ShaderProgram | None = None

    def __init__(
            self,
            points: list[Vec3] | None = None,
            color: Color | None = None,
            colors: list[Color] | None = None,
            size: float = 6.0,
    ) -> None:
        self.points: list[Vec3] = points or [Vec3.zero()]
        self.color: Color = color or Color.white()
        self.colors: list[Color] | None = colors
        self.size: float = size

        self._vao: int | None = None
        self._vbo: int | None = None

    @classmethod
    def _get_program(cls) -> int:
        if cls._program is None:
            glEnable(GL_PROGRAM_POINT_SIZE)
            vert = compileShader(cls._VERT_SRC, GL_VERTEX_SHADER)
            frag = compileShader(cls._FRAG_SRC, GL_FRAGMENT_SHADER)
            cls._program = compileProgram(vert, frag)

        if cls._program is None: raise ValueError(f"class program {cls._program} is not defined")

        return cls._program

    def _upload(self) -> None:
        if self._vao is not None:
            return

        raw: list[float] = []
        for i, p in enumerate(self.points):
            raw += [p.x, p.y, p.z]
            c = self.colors[i] if (self.colors and i < len(self.colors)) else self.color
            raw += [c.r, c.g, c.b]

        data = np.array(raw, dtype=np.float32)
        stride = 6 * 4

        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glBindVertexArray(0)

    @override
    def draw(
            self,
            view_mat: NDArray[np.float32],
            proj_mat: NDArray[np.float32],
            scene: "Scene"
    ) -> None:
        self._upload()
        prog = self._get_program()
        glUseProgram(prog)

        model = self.owner.transform.matrix()
        glUniformMatrix4fv(glGetUniformLocation(prog, "uModel"), 1, GL_TRUE, model)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uView"), 1, GL_TRUE, view_mat)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uProjection"), 1, GL_TRUE, proj_mat)
        glUniform1f(glGetUniformLocation(prog, "uPointSize"), self.size)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_POINTS, 0, len(self.points))
        glBindVertexArray(0)
        glUseProgram(0)

    def on_destroy(self) -> None:
        if self._vao is not None:
            glDeleteVertexArrays(1, [self._vao])
            glDeleteBuffers(1, [self._vbo])
            self._vao = None