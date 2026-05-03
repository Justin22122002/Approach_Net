from typing import Optional

import numpy as np
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


_VERT_SRC = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat3 uNormalMat;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal   = normalize(uNormalMat * aNormal);
    vUV       = aUV;
    gl_Position = uProjection * uView * worldPos;
}
"""

_FRAG_SRC = """
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;

uniform sampler2D uTexture;
uniform vec3  uAmbient;
uniform vec3  uLightDir;
uniform vec3  uLightColor;
uniform float uLightIntensity;
uniform vec4  uColor;

out vec4 FragColor;

void main() {
    vec4 texColor = texture(uTexture, vUV) * uColor;

    // Diffuse (Lambert)
    vec3  ld      = normalize(-uLightDir);
    float diff    = max(dot(vNormal, ld), 0.0);
    vec3  diffuse = diff * uLightColor * uLightIntensity;

    vec3 result = (uAmbient + diffuse) * texColor.rgb;
    FragColor   = vec4(result, texColor.a);
}
"""

class Material:
    _shared_program: Optional[int] = None

    def __init__(self, texture: Optional[Texture] = None, color: Optional[Color] = None):
        self.texture = texture or Texture.solid_color(1,1,1)
        self.color   = color   or Color.white()

    @classmethod
    def _get_program(cls) -> int:
        if cls._shared_program is None:
            vert = compileShader(_VERT_SRC, GL_VERTEX_SHADER)
            frag = compileShader(_FRAG_SRC, GL_FRAGMENT_SHADER)
            cls._shared_program = compileProgram(vert, frag)
        return cls._shared_program

    def bind(self, model_mat, view_mat, proj_mat, scene):
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

