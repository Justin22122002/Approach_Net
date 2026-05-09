from pathlib import Path
from typing import Optional

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
from OpenGL.GL.shaders import compileShader, compileProgram


def load_shader_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")

_VERT_SRC = load_shader_source(Path("utils/shaders/default.vert"))
_FRAG_SRC = load_shader_source(Path("utils/shaders/default.frag"))

class Material:
    _shared_program: Optional[int] = None

    def __init__(
            self,
            texture: Optional[Texture] = None,
            color: Optional[Color] = None
    ) -> None:
        self.texture: Texture = texture or Texture.solid_color(1,1,1)
        self.color: Color     = color   or Color.white()

    @classmethod
    def _get_program(cls) -> int:
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

        # lighting
        amb = scene.ambient_light
        glUniform3f(glGetUniformLocation(prog, "uAmbient"),
                    amb.r, amb.g, amb.b)

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

