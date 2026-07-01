import math
import numpy as np

from OpenGL.GL import (
    GL_LINE_LOOP,
    GL_DEPTH_TEST,
    GL_TRUE,
    GL_VERTEX_SHADER,
    GL_FRAGMENT_SHADER,
    glDisable,
    glEnable,
    glLineWidth,
    glGetUniformLocation,
    glUniform4f,
    glUniformMatrix4fv,
    glUseProgram,
)
from OpenGL.GL.shaders import compileShader, compileProgram, ShaderProgram

from utils.mesh import Mesh
from utils.color import Color
from utils.vec3 import Vec3
from utils.helpers import _mat4_translate, _mat4_rotate_x, _mat4_rotate_z, _mat4_identity


_GIZMO_VERT = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
void main() {
    gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);
}
"""

_GIZMO_FRAG = """
#version 330 core
out vec4 FragColor;
uniform vec4 uColor;
void main() {
    FragColor = uColor;
}
"""


class _GizmoShader:
    _program: ShaderProgram | None = None

    @classmethod
    def get(cls) -> ShaderProgram:
        if cls._program is None:
            vert = compileShader(_GIZMO_VERT, GL_VERTEX_SHADER)
            frag = compileShader(_GIZMO_FRAG, GL_FRAGMENT_SHADER)
            cls._program = compileProgram(vert, frag)
        return cls._program

    @classmethod
    def draw_mesh(cls, mesh: Mesh, model, view, proj, color: Color, mode=None) -> None:
        prog = cls.get()
        glUseProgram(prog)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uModel"), 1, GL_TRUE, model)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uView"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uProjection"), 1, GL_TRUE, proj)
        glUniform4f(glGetUniformLocation(prog, "uColor"), color.r, color.g, color.b, color.a)
        mesh.draw() if mode is None else mesh.draw(mode)
        glUseProgram(0)


AXIS_COLORS = {
    "x": Color(0.9, 0.15, 0.15, 1.0),
    "y": Color(0.15, 0.85, 0.15, 1.0),
    "z": Color(0.2, 0.45, 0.95, 1.0),
}
AXIS_DIRS = {
    "x": Vec3(1, 0, 0),
    "y": Vec3(0, 1, 0),
    "z": Vec3(0, 0, 1),
}

ARROW_SHAFT_LEN = 0.75
ARROW_HEAD_LEN  = 0.25
ARROW_TOTAL_LEN = ARROW_SHAFT_LEN + ARROW_HEAD_LEN
RING_RADIUS     = 0.9
PICK_TOLERANCE  = 0.10
RING_TOLERANCE  = 0.08


def _axis_basis(axis: str) -> tuple[Vec3, Vec3, Vec3]:
    if axis == "x":
        return Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1)
    if axis == "y":
        return Vec3(0,1,0), Vec3(1,0,0), Vec3(0,0,1)
    return Vec3(0,0,1), Vec3(1,0,0), Vec3(0,1,0)


def _arrow_model_matrix(origin: Vec3, axis: str) -> np.ndarray:
    T = _mat4_translate(origin)
    if axis == "x":
        R = _mat4_rotate_z(-math.pi / 2)
    elif axis == "z":
        R = _mat4_rotate_x(math.pi / 2)
    else:
        R = _mat4_identity()
    return T @ R


def _closest_t_and_dist(ray_o: Vec3, ray_d: Vec3, line_p: Vec3, line_d: Vec3) -> tuple[float, float]:
    line_d = line_d.normalized()
    ray_d  = ray_d.normalized()
    w0 = line_p - ray_o
    b  = line_d.dot(ray_d)
    d  = line_d.dot(w0)
    e  = ray_d.dot(w0)
    denom = 1.0 - b * b
    s = (b * e - d) / denom if abs(denom) > 1e-9 else 0.0
    closest_on_line = line_p + line_d * s
    t = ray_d.dot(closest_on_line - ray_o)
    closest_on_ray = ray_o + ray_d * t
    return s, (closest_on_line - closest_on_ray).length()


class TransformGizmo:
    def __init__(self) -> None:
        self.target: "GameObject | None" = None
        self.mode: str = "translate"

        self._arrow_shaft = Mesh.cylinder(radius=0.035, height=ARROW_SHAFT_LEN, segments=10)
        self._arrow_head  = Mesh.cone(radius=0.09, height=ARROW_HEAD_LEN, segments=10)
        self._rings = {ax: Mesh.ring(radius=RING_RADIUS, segments=64, axis=ax) for ax in ("x", "y", "z")}

        self._drag_axis: str | None = None
        self._drag_kind: str | None = None  # "translate" | "rotate"
        self._drag_start_pos: Vec3  = Vec3.zero()
        self._drag_start_rot: Vec3  = Vec3.zero()
        self._grab_value: float     = 0.0

    def attach(self, go: "GameObject") -> None:
        self.target = go

    def detach(self) -> None:
        self.target = None
        self._drag_axis = None
        self._drag_kind = None

    @property
    def is_dragging(self) -> bool:
        return self._drag_axis is not None

    def try_start_drag(self, ray_o: Vec3, ray_d: Vec3) -> bool:
        if self.target is None:
            return False
        origin = self.target.transform.position

        if self.mode == "translate":
            best_axis, best_dist, best_s = None, math.inf, 0.0
            for axis, axis_dir in AXIS_DIRS.items():
                s, dist = _closest_t_and_dist(ray_o, ray_d, origin, axis_dir)
                if 0.0 <= s <= ARROW_TOTAL_LEN and dist < PICK_TOLERANCE and dist < best_dist:
                    best_axis, best_dist, best_s = axis, dist, s
            if best_axis is None:
                return False
            self._drag_axis, self._drag_kind = best_axis, "translate"
            self._drag_start_pos = origin
            self._grab_value = best_s
            return True

        best_axis, best_dist, best_angle = None, math.inf, 0.0
        for axis in ("x", "y", "z"):
            axis_dir, u, v = _axis_basis(axis)
            denom = ray_d.dot(axis_dir)
            if abs(denom) < 1e-9:
                continue
            t = (origin - ray_o).dot(axis_dir) / denom
            if t < 0:
                continue
            rel = (ray_o + ray_d * t) - origin
            ring_dist = abs(rel.length() - RING_RADIUS)
            if ring_dist < RING_TOLERANCE and ring_dist < best_dist:
                best_axis, best_dist = axis, ring_dist
                best_angle = math.atan2(rel.dot(v), rel.dot(u))
        if best_axis is None:
            return False
        self._drag_axis, self._drag_kind = best_axis, "rotate"
        self._drag_start_rot = self.target.transform.rotation
        self._grab_value = best_angle
        return True

    def update_drag(self, ray_o: Vec3, ray_d: Vec3) -> None:
        if self.target is None or self._drag_axis is None:
            return

        if self._drag_kind == "translate":
            axis_dir = AXIS_DIRS[self._drag_axis]
            s, _ = _closest_t_and_dist(ray_o, ray_d, self._drag_start_pos, axis_dir)
            self.target.transform.position = self._drag_start_pos + axis_dir * (s - self._grab_value)
            return

        axis_dir, u, v = _axis_basis(self._drag_axis)
        origin = self.target.transform.position
        denom = ray_d.dot(axis_dir)
        if abs(denom) < 1e-9:
            return
        t = (origin - ray_o).dot(axis_dir) / denom
        if t < 0:
            return
        rel = (ray_o + ray_d * t) - origin
        angle = math.atan2(rel.dot(v), rel.dot(u))
        delta_deg = math.degrees(angle - self._grab_value)
        rot = self._drag_start_rot
        if self._drag_axis == "x":
            self.target.transform.rotation = Vec3(rot.x + delta_deg, rot.y, rot.z)
        elif self._drag_axis == "y":
            self.target.transform.rotation = Vec3(rot.x, rot.y + delta_deg, rot.z)
        else:
            self.target.transform.rotation = Vec3(rot.x, rot.y, rot.z + delta_deg)

    def end_drag(self) -> None:
        self._drag_axis = None
        self._drag_kind = None

    def draw(self, view: np.ndarray, proj: np.ndarray) -> None:
        if self.target is None:
            return
        origin = self.target.transform.position

        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.5)

        if self.mode == "translate":
            for axis in ("x", "y", "z"):
                model = _arrow_model_matrix(origin, axis)
                color = Color(1.0, 1.0, 0.3, 1.0) if axis == self._drag_axis else AXIS_COLORS[axis]
                _GizmoShader.draw_mesh(self._arrow_shaft, model, view, proj, color)
                head_model = model @ _mat4_translate(Vec3(0, ARROW_SHAFT_LEN, 0))
                _GizmoShader.draw_mesh(self._arrow_head, head_model, view, proj, color)
        else:
            model = _mat4_translate(origin)
            for axis in ("x", "y", "z"):
                color = Color(1.0, 1.0, 0.3, 1.0) if axis == self._drag_axis else AXIS_COLORS[axis]
                _GizmoShader.draw_mesh(self._rings[axis], model, view, proj, color, mode=GL_LINE_LOOP)

        glEnable(GL_DEPTH_TEST)