from abc import ABCMeta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from utils.texture import Texture
from utils.color import Color

from OpenGL.GL import (
    GL_FALSE,
    GL_FRAGMENT_SHADER,
    GL_TRUE,
    GL_VERTEX_SHADER,
    glGetUniformLocation,
    glUniform1f,
    glUniform1i,
    glUniform3f,
    glUniform4f,
    glUniformMatrix3fv,
    glUniformMatrix4fv,
    glUseProgram,
)
from OpenGL.GL.shaders import compileShader, compileProgram, ShaderProgram


def load_shader_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")

_VERT_SRC = load_shader_source(Path("utils/shaders/default.vert"))
_FRAG_SRC = load_shader_source(Path("utils/shaders/default.frag"))

class Material(metaclass=ABCMeta):
    pass

class ComplexMaterial(Material):
    _shared_program: ShaderProgram | None = None

    def __init__(
            self,
            texture: Texture | None  = None,
            color: Color | None      = None,
            specular_strength: float = 0.5,
            shininess: float         = 16.0,
    ) -> None:
        self.texture: Texture           = texture or Texture.solid_color(1, 1, 1)
        self.color: Color               = color   or Color.white()
        self.specular_strength: float   = specular_strength
        self.shininess:         float   = shininess

    @classmethod
    def _get_program(cls) -> ShaderProgram | int:
        if cls._shared_program is None:
            vert = compileShader(_VERT_SRC, GL_VERTEX_SHADER)
            frag = compileShader(_FRAG_SRC, GL_FRAGMENT_SHADER)
            cls._shared_program = compileProgram(vert, frag)
        return cls._shared_program

    def bind(
        self,
        model_mat: NDArray[np.float32],
        view_mat: NDArray[np.float32],
        proj_mat: NDArray[np.float32],
        scene: "Scene"
    ) -> None:
        prog = self._get_program()
        glUseProgram(prog)

        # matrices
        glUniformMatrix4fv(glGetUniformLocation(prog, "uModel"),      1, GL_TRUE, model_mat)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uView"),       1, GL_TRUE, view_mat)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uProjection"), 1, GL_TRUE, proj_mat)

        # normal matrix (inverse transpose of upper-left 3×3)
        nm = np.linalg.inv(model_mat[:3,:3]).T.astype(np.float32)
        glUniformMatrix3fv(glGetUniformLocation(prog, "uNormalMat"), 1, GL_FALSE, nm)

        cam_pos = np.linalg.inv(view_mat)[:3, 3].astype(np.float32)
        glUniform3f(glGetUniformLocation(prog, "uCamPos"), float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))

        # lighting
        amb = scene.ambient_light
        glUniform3f(glGetUniformLocation(prog, "uAmbient"), amb.r, amb.g, amb.b)

        dl = scene.directional_light
        if dl:
            d = dl.direction.normalized()
            glUniform1i(glGetUniformLocation(prog, "uDirLightEnabled"), 1)
            glUniform3f(glGetUniformLocation(prog, "uDirLightDir"),       d.x,          d.y,          d.z)
            glUniform3f(glGetUniformLocation(prog, "uDirLightColor"),     dl.color.r,   dl.color.g,   dl.color.b)
            glUniform1f(glGetUniformLocation(prog, "uDirLightIntensity"), dl.intensity)
        else:
            glUniform1i(glGetUniformLocation(prog, "uDirLightEnabled"), 0)

        # point lights
        point_lights = scene.point_lights
        n_pt = min(len(point_lights), 8)
        glUniform1i(glGetUniformLocation(prog, "uNumPointLights"), n_pt)
        for i, pl in enumerate(point_lights[:n_pt]):
            p = pl.position
            glUniform3f(glGetUniformLocation(prog, f"uPointLightPos[{i}]"),   p.x, p.y, p.z)
            glUniform3f(glGetUniformLocation(prog, f"uPointLightColor[{i}]"), pl.color.r, pl.color.g, pl.color.b)
            glUniform1f(glGetUniformLocation(prog, f"uPointLightIntensity[{i}]"), pl.intensity)
            glUniform1f(glGetUniformLocation(prog, f"uPointLightA[{i}]"), pl.a)
            glUniform1f(glGetUniformLocation(prog, f"uPointLightB[{i}]"), pl.b)

        # Spot Lights
        spot_lights = scene.spot_lights
        n_sp = min(len(spot_lights), 4)
        glUniform1i(glGetUniformLocation(prog, "uNumSpotLights"), n_sp)
        for i, sl in enumerate(spot_lights[:n_sp]):
            p = sl.position
            d = sl.direction.normalized()
            glUniform3f(glGetUniformLocation(prog, f"uSpotLightPos[{i}]"), p.x, p.y, p.z)
            glUniform3f(glGetUniformLocation(prog, f"uSpotLightDir[{i}]"), d.x, d.y, d.z)
            glUniform3f(glGetUniformLocation(prog, f"uSpotLightColor[{i}]"), sl.color.r, sl.color.g, sl.color.b)
            glUniform1f(glGetUniformLocation(prog, f"uSpotLightIntensity[{i}]"), sl.intensity)
            glUniform1f(glGetUniformLocation(prog, f"uSpotInnerCone[{i}]"), sl.inner_cos)
            glUniform1f(glGetUniformLocation(prog, f"uSpotOuterCone[{i}]"), sl.outer_cos)

        glUniform1f(glGetUniformLocation(prog, "uSpecularStrength"), self.specular_strength)
        glUniform1f(glGetUniformLocation(prog, "uShininess"),        self.shininess)

        # colour tint
        c = self.color
        glUniform4f(glGetUniformLocation(prog, "uColor"), c.r, c.g, c.b, c.a)

        # texture
        self.texture.bind(0)
        glUniform1i(glGetUniformLocation(prog, "uTexture"), 0)


_SIMPLE_VERT_SRC = load_shader_source(Path("utils/shaders/simple_default.vert"))
_SIMPLE_FRAG_SRC = load_shader_source(Path("utils/shaders/simple_default.frag"))

class SimpleMaterial(Material):
    _shared_program: ShaderProgram | None = None

    def __init__(
            self,
            texture: Texture | None = None,
            color: Color | None = None
    ) -> None:
        self.texture: Texture = texture or Texture.solid_color(1,1,1)
        self.color: Color     = color   or Color.white()

    @classmethod
    def _get_program(cls) -> ShaderProgram | int:
        if cls._shared_program is None:
            vert = compileShader(_SIMPLE_VERT_SRC, GL_VERTEX_SHADER)
            frag = compileShader(_SIMPLE_FRAG_SRC, GL_FRAGMENT_SHADER)
            cls._shared_program = compileProgram(vert, frag)
        return cls._shared_program

    def bind(
        self,
        model_mat: NDArray[np.float32],
        view_mat: NDArray[np.float32],
        proj_mat: NDArray[np.float32],
        scene: "Scene"
    ) -> None:
        prog = self._get_program()
        glUseProgram(prog)

        # matrices
        glUniformMatrix4fv(glGetUniformLocation(prog, "uModel"),      1, GL_TRUE, model_mat)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uView"),       1, GL_TRUE, view_mat)
        glUniformMatrix4fv(glGetUniformLocation(prog, "uProjection"), 1, GL_TRUE, proj_mat)

        # normal matrix (inverse transpose of upper-left 3×3)
        nm = np.linalg.inv(model_mat[:3,:3]).T.astype(np.float32)
        glUniformMatrix3fv(glGetUniformLocation(prog, "uNormalMat"), 1, GL_FALSE, nm)

        # lighting
        amb = scene.ambient_light
        glUniform3f(glGetUniformLocation(prog, "uAmbient"), amb.r, amb.g, amb.b)

        dl = scene.directional_light
        if dl:
            d = dl.direction.normalized()
            glUniform3f(glGetUniformLocation(prog, "uLightDir"), d.x, d.y, d.z)
            glUniform3f(glGetUniformLocation(prog, "uLightColor"), dl.color.r, dl.color.g, dl.color.b)
            glUniform1f(glGetUniformLocation(prog, "uLightIntensity"), dl.intensity)
        else:
            glUniform3f(glGetUniformLocation(prog, "uLightDir"), 0,-1,0)
            glUniform3f(glGetUniformLocation(prog, "uLightColor"), 0,0,0)
            glUniform1f(glGetUniformLocation(prog, "uLightIntensity"), 0)

        # colour tint
        c = self.color
        glUniform4f(glGetUniformLocation(prog, "uColor"), c.r, c.g, c.b, c.a)

        # texture
        self.texture.bind(0)
        glUniform1i(glGetUniformLocation(prog, "uTexture"), 0)

