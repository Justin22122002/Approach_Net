import math
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_ALWAYS,
    GL_DEPTH_TEST,
    GL_CULL_FACE,
    GL_FRAGMENT_SHADER,
    GL_LESS,
    GL_TRIANGLES,
    GL_TRUE,
    GL_VERTEX_SHADER,
    glBindVertexArray,
    glDepthFunc,
    glDepthMask,
    glDisable,
    glDrawArrays,
    glEnable,
    glGenVertexArrays,
    glGetUniformLocation,
    glUniform1f,
    glUniform2f,
    glUniform3f,
    glUniformMatrix4fv,
    glUseProgram,
)
from OpenGL.GL.shaders import compileProgram, compileShader, ShaderProgram

from engine.utils.color import Color
from engine.utils.directionalLight import DirectionalLight
from engine.utils.vec3 import Vec3
from utils.simplematerial import load_shader_source


class Sky:
    _UTILS_DIR = Path(__file__).resolve().parent
    _SHADER_DIR = _UTILS_DIR / "shaders"

    _SKY_VERT = load_shader_source(_SHADER_DIR / "sky.vert")
    _SKY_FRAG = load_shader_source(_SHADER_DIR / "sky.frag")

    def __init__(
        self,
        sun_elevation:  float = 45.0,
        sun_azimuth:    float = 135.0,
        cloud_density:  float = 0.45,
        cloud_speed:    float = 0.008,
        cloud_altitude: float = 80.0,
        zenith_color:   tuple[float, float, float] = (0.07, 0.22, 0.68),
        horizon_color:  tuple[float, float, float] = (0.52, 0.75, 0.94),
    ) -> None:
        self.sun_elevation:  float = sun_elevation
        self.sun_azimuth:    float = sun_azimuth
        self.cloud_density:  float = cloud_density
        self.cloud_speed:    float = cloud_speed
        self.cloud_altitude: float = cloud_altitude
        self.zenith_color:   tuple[float, float, float] = zenith_color
        self.horizon_color:  tuple[float, float, float] = horizon_color

        self.directional_light: DirectionalLight = DirectionalLight(
            direction=self.get_light_direction(),
            color=Color(1.0, 0.95, 0.85)
        )

        self._prog: ShaderProgram | None = None
        self._vao:  int | None           = None

    def _ensure_gpu(self) -> None:
        if self._prog is not None:
            return
        vert = compileShader(self._SKY_VERT, GL_VERTEX_SHADER)
        frag = compileShader(self._SKY_FRAG, GL_FRAGMENT_SHADER)
        self._prog = compileProgram(vert, frag)
        self._vao  = glGenVertexArrays(1)

    def _sun_direction(self) -> Vec3:
        el  = math.radians(self.sun_elevation)
        az  = math.radians(self.sun_azimuth)
        x   = math.cos(el) * math.sin(az)
        y   = math.sin(el)
        z   = math.cos(el) * math.cos(az)
        return Vec3(x, y, z)

    def render(
        self,
        view_mat:  np.ndarray,
        proj_mat:  np.ndarray,
        time:      float,
        width:     int,
        height:    int,
        cam_pos:   Vec3 = None,
    ) -> None:
        self._ensure_gpu()

        view_rot       = view_mat.copy()
        view_rot[0, 3] = 0.0
        view_rot[1, 3] = 0.0
        view_rot[2, 3] = 0.0
        vp      = proj_mat @ view_rot
        inv_vp  = np.linalg.inv(vp).astype(np.float32)
        full_vp = (proj_mat @ view_mat).astype(np.float32)

        sun    = self._sun_direction()
        cp     = cam_pos or Vec3.zero()

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_ALWAYS)
        glDisable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)

        prog = self._prog
        glUseProgram(prog)

        glUniformMatrix4fv(glGetUniformLocation(prog, "uInvVP"),        1, GL_TRUE, inv_vp)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uVP"),           1, GL_TRUE, full_vp)
        glUniform3f(       glGetUniformLocation(prog, "uSunDir"),        sun.x, sun.y, sun.z)
        glUniform1f(       glGetUniformLocation(prog, "uTime"),          time)
        glUniform1f(       glGetUniformLocation(prog, "uCloudDensity"),  self.cloud_density)
        glUniform1f(       glGetUniformLocation(prog, "uCloudSpeed"),    self.cloud_speed)
        glUniform3f(       glGetUniformLocation(prog, "uZenithColor"),   *self.zenith_color)
        glUniform3f(       glGetUniformLocation(prog, "uHorizonColor"),  *self.horizon_color)
        glUniform2f(       glGetUniformLocation(prog, "uResolution"),    float(width), float(height))
        glUniform3f(       glGetUniformLocation(prog, "uCamPos"),        cp.x, cp.y, cp.z)
        glUniform1f(       glGetUniformLocation(prog, "uCloudAltitude"), self.cloud_altitude)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)

        glUseProgram(0)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)  # glDisable(GL_CULL_FACE) -> todo: GL_CULL_FACE from engine

    def get_light_direction(self) -> Vec3:
        sun = self._sun_direction()
        return Vec3(-sun.x, -sun.y, -sun.z)


def sky_day() -> Sky:
    return Sky(
        sun_elevation=55.0,
        sun_azimuth=135.0,
        cloud_density=0.35,
        zenith_color=(0.07, 0.22, 0.68),
        horizon_color=(0.52, 0.75, 0.94)
    )

def sky_overcast() -> Sky:
    return Sky(
        sun_elevation=40.0,
        sun_azimuth=180.0,
        cloud_density=0.85,
        zenith_color=(0.30, 0.33, 0.38),
        horizon_color=(0.55, 0.57, 0.60)
    )

def sky_sunset() -> Sky:
    return Sky(
        sun_elevation=4.0,
        sun_azimuth=270.0,
        cloud_density=0.4,
        zenith_color=(0.08, 0.10, 0.35),
        horizon_color=(0.90, 0.42, 0.10)
    )

def sky_dawn() -> Sky:
    return Sky(
        sun_elevation=6.0,
        sun_azimuth=90.0,
        cloud_density=0.3,
        zenith_color=(0.12, 0.15, 0.45),
        horizon_color=(0.95, 0.65, 0.35)
    )

def sky_noon_clear() -> Sky:
    return Sky(
        sun_elevation=80.0,
        sun_azimuth=180.0,
        cloud_density=0.10,
        zenith_color=(0.04, 0.18, 0.72),
        horizon_color=(0.48, 0.72, 0.95)
    )