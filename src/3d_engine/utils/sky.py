import math

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

from utils.color import Color
from utils.directionalLight import DirectionalLight
from utils.vec3 import Vec3

_SKY_VERT = """
#version 330 core
out vec2 vUV;
void main() {
    vec2 pos = vec2(
        float((gl_VertexID & 1) * 4) - 1.0,
        float((gl_VertexID & 2) * 2) - 1.0
    );
    vUV         = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.9999, 1.0);
}
"""

_SKY_FRAG = """
#version 330 core
in  vec2 vUV;
out vec4 FragColor;

uniform mat4  uInvVP;
uniform mat4  uVP;
uniform vec3  uSunDir;
uniform float uTime;
uniform float uCloudDensity;
uniform float uCloudSpeed;
uniform vec3  uZenithColor;
uniform vec3  uHorizonColor;
uniform vec2  uResolution;
uniform vec3  uCamPos;
uniform float uCloudAltitude;

float hash(vec2 p) {
    p  = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}

float smoothNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash(i),             hash(i + vec2(1,0)), u.x),
        mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), u.x),
        u.y
    );
}

float fbm(vec2 p) {
    float value = 0.0;
    float amp   = 0.5;
    mat2 rot = mat2(0.809, 0.588, -0.588, 0.809);
    for (int i = 0; i < 6; i++) {
        value += amp * smoothNoise(p);
        p      = rot * p * 2.1;
        amp   *= 0.5;
    }
    return value;
}

void main() {
    vec4 ndc   = vec4(vUV * 2.0 - 1.0, 1.0, 1.0);
    vec4 world = uInvVP * ndc;
    vec3 dir   = normalize(world.xyz / world.w);

    float elev = dir.y;

    gl_FragDepth = 1.0;

    vec3 skyColor;
    if (elev >= 0.0) {
        skyColor = mix(uHorizonColor, uZenithColor, pow(elev, 0.55));
    } else {
        skyColor = mix(uHorizonColor, vec3(0.18, 0.14, 0.11), pow(min(-elev, 1.0), 0.4));
    }

    float haze = exp(-abs(elev) * 5.0);
    skyColor   = mix(skyColor, vec3(0.82, 0.89, 0.97), haze * 0.35);

    float distToCloud = uCloudAltitude - uCamPos.y;
    float cloudSign   = sign(distToCloud);

    if (abs(distToCloud) > 0.01 && elev * cloudSign > 0.005) {
        float tCloud   = abs(distToCloud) / max(abs(dir.y), 0.001);
        vec2  cloudHit = uCamPos.xz + dir.xz * tCloud;
        vec2  cloudUV  = cloudHit * 0.008;

        vec2  uv1   = cloudUV + uTime * uCloudSpeed * vec2(1.0, 0.35);
        float c1    = fbm(uv1);
        float c2    = fbm(uv1 * 2.3 + vec2(5.7, 3.1));
        float cloud = c1 * 0.65 + c2 * 0.35;

        vec2  uv2    = cloudUV * 3.3 + uTime * uCloudSpeed * 1.8 * vec2(0.6, 1.0);
        float cirrus = fbm(uv2) * 0.3;

        float threshold = 0.78 - uCloudDensity * 0.32;
        cloud  = smoothstep(threshold, threshold + 0.15, cloud);
        cirrus = smoothstep(0.62, 0.75, cirrus);

        float horizonFade = smoothstep(0.005, 0.15, abs(elev));
        cloud  *= horizonFade;
        cirrus *= horizonFade * 0.5;

        float litY    = cloudSign * 0.6;
        float cloudLit = dot(normalize(vec3(dir.x, litY, dir.z)), uSunDir) * 0.5 + 0.5;

        vec3 cloudColor  = mix(vec3(0.72, 0.76, 0.85), vec3(1.00, 0.97, 0.93), cloudLit);
        cloudColor      += vec3(1.0, 0.9, 0.7) * pow(max(dot(dir, uSunDir), 0.0), 12.0) * 0.15;
        vec3 cirrusColor = mix(vec3(0.82, 0.87, 0.95), vec3(1.0, 0.98, 0.94), cloudLit);

        skyColor = mix(skyColor, cloudColor,  cloud);
        skyColor = mix(skyColor, cirrusColor, cirrus);

        if (cloud > 0.02) {
            vec3  hitPoint = vec3(cloudHit.x, uCloudAltitude, cloudHit.y);
            vec4  clipPos  = uVP * vec4(hitPoint, 1.0);
            float ndcZ     = clipPos.z / clipPos.w;
            gl_FragDepth   = clamp(ndcZ * 0.5 + 0.5, 0.0, 1.0);
        }
    }

    float sunDot   = dot(dir, uSunDir);
    float halo     = pow(max(sunDot, 0.0), 6.0);
    float glowWide = pow(max(sunDot, 0.0), 64.0);
    skyColor += uSunDir.y > -0.05
        ? vec3(1.0, 0.80, 0.45) * halo * 0.55
        : vec3(1.0, 0.40, 0.10) * halo * 0.40;
    skyColor += vec3(1.1, 1.0, 0.8) * glowWide * 0.8;
    skyColor += vec3(2.5, 2.2, 1.6) * step(0.99985, sunDot);

    skyColor = skyColor / (skyColor + vec3(0.6));
    skyColor = pow(max(skyColor, vec3(0.0)), vec3(1.0 / 2.2));

    FragColor = vec4(skyColor, 1.0);
}
"""


class Sky:
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
        vert = compileShader(_SKY_VERT, GL_VERTEX_SHADER)
        frag = compileShader(_SKY_FRAG, GL_FRAGMENT_SHADER)
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
        glEnable(GL_CULL_FACE)

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