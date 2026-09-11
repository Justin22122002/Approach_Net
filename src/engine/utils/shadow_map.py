from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from OpenGL.GL import (
    GL_CLAMP_TO_BORDER,
    GL_DEPTH_ATTACHMENT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_COMPONENT,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_FRAMEBUFFER,
    GL_LINEAR,
    GL_NONE,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_BORDER_COLOR,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRUE,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glBindFramebuffer,
    glBindTexture,
    glClear,
    glDrawBuffer,
    glFramebufferTexture2D,
    glGenFramebuffers,
    glGenTextures,
    glGetUniformLocation,
    glReadBuffer,
    glTexImage2D,
    glTexParameterfv,
    glTexParameteri,
    glUniformMatrix4fv,
    glUseProgram,
    glViewport,
)
from OpenGL.GL.shaders import compileProgram, compileShader

from engine.utils.helpers import _mat4_look_at, _mat4_ortho
from engine.utils.vec3 import Vec3
from engine.utils.simplematerial import load_shader_source

_UTILS_DIR = Path(__file__).resolve().parent
_SHADER_DIR = _UTILS_DIR / "shaders"

_DEPTH_VERT_SRC = load_shader_source(_SHADER_DIR / "shadow_depth.vert")
_DEPTH_FRAG_SRC = load_shader_source(_SHADER_DIR / "shadow_depth.frag")

class ShadowMap:
    """Single 2D depth map for the scene's directional light. Sampled in the
    material shaders with slope-scaled depth bias + normal-offset + 3x3 PCF
    for soft, acne-free edges (curved geometry like spheres included)."""

    def __init__(self, size: int = 4096) -> None:
        self.size: int                 = size
        self.depth_bias_min: float     = 0.0007
        self.depth_bias_max: float     = 0.0030
        self.normal_bias_texels: float = 1.5   # normal-offset in shadow texels -> auto-scales with shadow_radius

        self.light_space_matrix: NDArray[np.float32] = np.identity(4, dtype=np.float32)
        self.normal_bias: float = 0.0  # resolved world-space offset, recomputed each render()

        self._fbo = None
        self._depth_tex = None
        self._program = None

    def _ensure_gl_resources(self) -> None:
        if self._fbo is not None:
            return

        self._depth_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._depth_tex)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
            self.size, self.size, 0,
            GL_DEPTH_COMPONENT, GL_FLOAT, None
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1.0, 1.0, 1.0, 1.0])
        glBindTexture(GL_TEXTURE_2D, 0)

        self._fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self._depth_tex, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        vert = compileShader(_DEPTH_VERT_SRC, GL_VERTEX_SHADER)
        frag = compileShader(_DEPTH_FRAG_SRC, GL_FRAGMENT_SHADER)
        self._program = compileProgram(vert, frag)

    def _compute_light_space_matrix(self, light_dir: Vec3, center: Vec3, radius: float) -> NDArray[np.float32]:
        d    = light_dir.normalized()
        eye  = center - d * (radius * 2.0)
        up   = Vec3.right() if abs(d.y) > 0.99 else Vec3.up()
        view = _mat4_look_at(eye, center, up)
        proj = _mat4_ortho(-radius, radius, -radius, radius, radius * 0.5, radius * 3.5)
        return proj @ view

    def render(self, light_dir: Vec3, center: Vec3, radius: float, casters) -> None:
        """casters: iterable of (model_matrix, Mesh) pairs to render into the depth map."""
        self._ensure_gl_resources()
        self.light_space_matrix = self._compute_light_space_matrix(light_dir, center, radius)
        self.normal_bias = self.normal_bias_texels * (2.0 * radius / self.size)

        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, self.size, self.size)
        glClear(GL_DEPTH_BUFFER_BIT)

        glUseProgram(self._program)
        loc_ls = glGetUniformLocation(self._program, "uLightSpaceMatrix")
        loc_m  = glGetUniformLocation(self._program, "uModel")
        glUniformMatrix4fv(loc_ls, 1, GL_TRUE, self.light_space_matrix)
        for model_mat, mesh in casters:
            glUniformMatrix4fv(loc_m, 1, GL_TRUE, model_mat)
            mesh.draw()
        glUseProgram(0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind_for_sampling(self, unit: int = 1) -> None:
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self._depth_tex)