import ctypes
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LINES,
    GL_STATIC_DRAW,
    GL_TRUE,
    GL_VERTEX_SHADER,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glDrawArrays,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glGetUniformLocation,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glUniform1i
)
from OpenGL.GL.shaders import compileProgram, compileShader
from numpy.typing import NDArray
from typing_extensions import override

from utils.color import Color
from utils.component import Component
from utils.simplematerial import load_shader_source
from utils.renderer import Renderer
from utils.vec3 import Vec3

_VERT_SRC = load_shader_source(Path("utils/shaders/line.vert"))
_FRAG_SRC = load_shader_source(Path("utils/shaders/line.frag"))

class LineRenderer(Component, Renderer):
    """
    Renders a set of line segments in 3D space.

    points  – flat line of Vec3, always 2 for the directly following points = 1 segment
    color   - one color for all points (ignored when colors are set, see the comment down below)
    colors  - per vertex-color, needs to be the same length as the points
    """

    _program: int | None = None  # shared across all instances

    def __init__(
        self,
        points: list[Vec3],
        color: Color | None        = None,
        colors: list[Color] | None = None,
        depth_offset: bool         = False,
    ) -> None:
        self.points: list[Vec3]          = points
        self.color:  Color               = color or Color.white()
        self.colors: list[Color] | None  = colors
        self.depth_offset: bool          = depth_offset

        self._vao: int | None = None
        self._vbo: int | None = None

    @classmethod
    def _get_program(cls) -> int:
        if cls._program is None:
            vert = compileShader(_VERT_SRC, GL_VERTEX_SHADER)
            frag = compileShader(_FRAG_SRC, GL_FRAGMENT_SHADER)
            cls._program = compileProgram(vert, frag)

        if cls._program is None: raise ValueError(f"class program {cls._program} is not defined")

        return cls._program

    # GPU-Upload
    def _upload(self) -> None:
        if self._vao is not None:
            return

        raw: list[float] = []
        for i, p in enumerate(self.points):
            raw += [p.x, p.y, p.z]
            if self.colors and i < len(self.colors):
                c = self.colors[i]
            else:
                c = self.color
            raw += [c.r, c.g, c.b]

        data = np.array(raw, dtype=np.float32)
        stride = 6 * 4  # 6 floats × 4 Bytes

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
        glUniformMatrix4fv(glGetUniformLocation(prog, "uModel"),      1, GL_TRUE, model)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uView"),       1, GL_TRUE, view_mat)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uProjection"), 1, GL_TRUE, proj_mat)
        glUniform1i(glGetUniformLocation(prog, "uDepthOffset"), 1 if self.depth_offset else 0)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_LINES, 0, len(self.points))
        glBindVertexArray(0)
        glUseProgram(0)

    def on_destroy(self) -> None:
        if self._vao is not None:
            glDeleteVertexArrays(1, [self._vao])
            glDeleteBuffers(1, [self._vbo])
            self._vao = None

    @staticmethod
    def segments(
        pairs: list[tuple[Vec3, Vec3]],
        color: Color | None = None,
        colors: list[tuple[Color, Color]] | None = None,
    ) -> "LineRenderer":
        points: list[Vec3]  = []
        vcols:  list[Color] = []
        for idx, (a, b) in enumerate(pairs):
            points += [a, b]
            if colors:
                ca, cb = colors[idx]
                vcols += [ca, cb]
        return LineRenderer(points, color=color, colors=vcols or None)

    @staticmethod
    def strip(
        points: list[Vec3],
        color: Color | None = None,
        colors: list[Color] | None = None,
        depth_offset: bool = False,
    ) -> "LineRenderer":
        expanded: list[Vec3]  = []
        ecols:    list[Color] = []
        for i in range(len(points) - 1):
            expanded += [points[i], points[i + 1]]
            if colors:
                ecols += [colors[i], colors[i + 1]]
        return LineRenderer(
            expanded,
            color=color,
            colors=ecols or None,
            depth_offset=depth_offset
        )

    @staticmethod
    def grid(
        size: float = 10.0,
        divisions: int = 10,
        color: Color | None = None,
    ) -> "LineRenderer":
        c    = color or Color(0.35, 0.35, 0.35)
        half = size / 2.0
        step = size / divisions
        pts: list[Vec3] = []
        for i in range(divisions + 1):
            t = -half + i * step
            pts += [Vec3(t, 0.0, -half), Vec3(t, 0.0,  half)]
            pts += [Vec3(-half, 0.0, t), Vec3( half, 0.0, t)]
        return LineRenderer(pts, color=c)

    @staticmethod
    def axes(length: float = 1.0) -> "LineRenderer":
        o = Vec3.zero()
        pts = [
            o, Vec3(length, 0, 0),   # X
            o, Vec3(0, length, 0),   # Y
            o, Vec3(0, 0, length),   # Z
        ]
        cols = [
            Color.red(),   Color.red(),
            Color.green(), Color.green(),
            Color.blue(),  Color.blue(),
        ]
        return LineRenderer(pts, colors=cols)

    @staticmethod
    def wire_box(
        min_pt: Vec3,
        max_pt: Vec3,
        color: Color | None = None,
    ) -> "LineRenderer":
        c = color or Color.white()
        x0, y0, z0 = min_pt.x, min_pt.y, min_pt.z
        x1, y1, z1 = max_pt.x, max_pt.y, max_pt.z
        corners = [
            Vec3(x0, y0, z0), Vec3(x1, y0, z0),
            Vec3(x1, y0, z1), Vec3(x0, y0, z1),
            Vec3(x0, y1, z0), Vec3(x1, y1, z0),
            Vec3(x1, y1, z1), Vec3(x0, y1, z1),
        ]
        edges = [
            (0,1),(1,2),(2,3),(3,0),   # lower edge
            (4,5),(5,6),(6,7),(7,4),   # upper edge
            (0,4),(1,5),(2,6),(3,7),   # connections
        ]
        pts: list[Vec3] = []
        for a, b in edges:
            pts += [corners[a], corners[b]]
        return LineRenderer(pts, color=c)

    @staticmethod
    def circle(
        radius: float = 1.0,
        segments: int = 32,
        axis: str = "y",
        color: Color | None = None,
    ) -> "LineRenderer":
        """circle in one of the three axes (axis='x'|'y'|'z')."""
        import math
        c   = color or Color.white()
        pts: list[Vec3] = []
        for i in range(segments):
            a0 = 2 * math.pi * i       / segments
            a1 = 2 * math.pi * (i + 1) / segments
            c0, s0 = math.cos(a0) * radius, math.sin(a0) * radius
            c1, s1 = math.cos(a1) * radius, math.sin(a1) * radius
            if axis == "y":
                pts += [Vec3(c0, 0, s0), Vec3(c1, 0, s1)]
            elif axis == "x":
                pts += [Vec3(0, c0, s0), Vec3(0, c1, s1)]
            else:  # z
                pts += [Vec3(c0, s0, 0), Vec3(c1, s1, 0)]
        return LineRenderer(pts, color=c)

    def set_points_strip(self, ordered: list[Vec3], color: Color | None = None) -> None:
        if len(ordered) < 2:
            return
        expanded: list[Vec3] = []
        for i in range(len(ordered) - 1):
            expanded.append(ordered[i])
            expanded.append(ordered[i + 1])
        self.points = expanded
        if color:
            self.color = color
        if self._vao is not None:
            glDeleteVertexArrays(1, [self._vao])
            glDeleteBuffers(1, [self._vbo])
            self._vao = None
            self._vbo = None