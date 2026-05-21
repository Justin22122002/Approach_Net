"""
Sky – Prozeduraler Himmel mit animierten Wolken
================================================
Rendert einen vollprozedualen Himmel als Fullscreen-Quad VOR der 3D-Szene.
Kein Texture-Atlas, keine Skybox – alles läuft im Fragment-Shader
via FBM-Noise (Fractional Brownian Motion).

Features:
  • Atmosphärischer Farbgradient  (Zenit → Horizont → Boden)
  • Sonnenscheibe + Lichtglocke
  • Zwei überlagerte Wolken-Ebenen mit unterschiedlicher Windgeschwindigkeit
  • Weiche Horizont-Haze
  • Tonemapping + Gamma-Korrektur

Verwendung:
    engine.sky = Sky()                          # Standard-Tageslicht
    engine.sky = Sky(
        sun_elevation=30.0,                     # Sonnenhöhe in Grad
        sun_azimuth=45.0,                       # Sonnenrichtung (0=Nord, 90=Ost)
        cloud_density=0.6,                      # 0.0 (klar) … 1.0 (bedeckt)
        cloud_speed=0.012,
        zenith_color=(0.07, 0.22, 0.68),
        horizon_color=(0.52, 0.75, 0.94),
    )
    engine.sky.sun_elevation = 10.0             # Jederzeit änderbar (Sonnenuntergang)
"""

import math
from typing import Optional, Tuple

import numpy as np
from OpenGL.GL import (
    GL_DEPTH_TEST,
    GL_CULL_FACE,
    GL_FRAGMENT_SHADER,
    GL_TRIANGLES,
    GL_TRUE,
    GL_VERTEX_SHADER,
    glBindVertexArray,
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
    GL_FALSE,
)
from OpenGL.GL.shaders import compileProgram, compileShader

from utils.color import Color
from utils.directionalLight import DirectionalLight
from utils.vec3 import Vec3

# ── Vertex-Shader  (Fullscreen-Dreieck, kein VBO nötig) ─────────────────────
_SKY_VERT = """
#version 330 core

out vec2 vUV;

void main() {
    // Fullscreen-Triangle-Trick: 3 Vertices decken den ganzen Screen ab
    vec2 pos = vec2(
        float((gl_VertexID & 1) * 4) - 1.0,
        float((gl_VertexID & 2) * 2) - 1.0
    );
    vUV         = pos * 0.5 + 0.5;   // 0..1, Y nicht gespiegelt
    gl_Position = vec4(pos, 0.9999, 1.0);  // fast Far-Plane → hinter allem
}
"""

# ── Fragment-Shader  (Atmosphäre + FBM-Wolken + Sonne) ──────────────────────
_SKY_FRAG = """
#version 330 core

in  vec2 vUV;
out vec4 FragColor;

uniform mat4  uInvVP;          // inverse(Projection * View)
uniform vec3  uSunDir;         // normalisiert, zeigt ZUR Sonne
uniform float uTime;
uniform float uCloudDensity;   // 0.0 = klar, 1.0 = bedeckt
uniform float uCloudSpeed;
uniform vec3  uZenithColor;
uniform vec3  uHorizonColor;
uniform vec2  uResolution;

// ── Hash / Noise ─────────────────────────────────────────────────────────────
float hash(vec2 p) {
    p  = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}

float smoothNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);   // Smoothstep-Interpolation
    return mix(
        mix(hash(i),               hash(i + vec2(1,0)), u.x),
        mix(hash(i + vec2(0,1)),   hash(i + vec2(1,1)), u.x),
        u.y
    );
}

// Fractional Brownian Motion – 6 Oktaven, rotierende Basis
float fbm(vec2 p) {
    float value = 0.0;
    float amp   = 0.5;
    // leichte Rotation pro Oktave → verhindert Achsenartefakte
    mat2 rot = mat2(0.809, 0.588, -0.588, 0.809);
    for (int i = 0; i < 6; i++) {
        value += amp * smoothNoise(p);
        p      = rot * p * 2.1;
        amp   *= 0.5;
    }
    return value;
}

// ── Hilfsfunktionen ───────────────────────────────────────────────────────────
// Weiches Remap
float remap(float v, float lo, float hi) {
    return clamp((v - lo) / (hi - lo), 0.0, 1.0);
}

void main() {
    // ── Weltrichtungsvektor aus NDC rekonstruieren ───────────────────────
    vec4 ndc   = vec4(vUV * 2.0 - 1.0, 1.0, 1.0);
    vec4 world = uInvVP * ndc;
    vec3 dir   = normalize(world.xyz / world.w);

    float elev = dir.y;   // -1 (unten) … +1 (oben)

    // ── Himmels-Gradient ──────────────────────────────────────────────────
    vec3 skyColor;
    if (elev >= 0.0) {
        float t  = pow(elev, 0.55);
        skyColor = mix(uHorizonColor, uZenithColor, t);
    } else {
        // Boden / Dunst unterhalb des Horizonts
        float t  = pow(min(-elev, 1.0), 0.4);
        skyColor = mix(uHorizonColor, vec3(0.18, 0.14, 0.11), t);
    }

    // Horizont-Haze (nahe dem Äquator besonders stark)
    float haze = exp(-abs(elev) * 5.0);
    skyColor   = mix(skyColor, vec3(0.82, 0.89, 0.97), haze * 0.35);

    // ── Wolken ────────────────────────────────────────────────────────────
    if (elev > 0.005) {
        // Projektion auf horizontale Wolken-Ebene
        vec2 cloudUV  = dir.xz / max(dir.y, 0.01);

        // Untere Wolkenschicht (groß, langsam)
        vec2 uv1 = cloudUV * 0.12 + uTime * uCloudSpeed * vec2(1.0, 0.35);
        float c1 = fbm(uv1);
        float c2 = fbm(uv1 * 2.3 + vec2(5.7, 3.1));
        float cloud = c1 * 0.65 + c2 * 0.35;

        // Obere Hoch-Wolkenschicht (fein, schneller)
        vec2 uv2    = cloudUV * 0.4 + uTime * uCloudSpeed * 1.8 * vec2(0.6, 1.0);
        float cirrus = fbm(uv2) * 0.3;

        // Dichte-Schwelle
        float threshold = 0.78 - uCloudDensity * 0.32;
        cloud  = smoothstep(threshold, threshold + 0.15, cloud);
        cirrus = smoothstep(0.62, 0.75, cirrus);

        // Sanft ausblenden am Horizont
        float horizonFade = smoothstep(0.0, 0.12, elev);
        cloud  *= horizonFade;
        cirrus *= horizonFade * 0.5;

        // Wolken-Beleuchtung (Sonnenseite heller)
        vec3 cloudNorm  = normalize(vec3(dir.x, 0.6, dir.z));
        float cloudLit  = dot(cloudNorm, uSunDir) * 0.5 + 0.5;
        vec3 cloudColor = mix(
            vec3(0.72, 0.76, 0.85),   // Schattenseite (blaugrau)
            vec3(1.00, 0.97, 0.93),   // Sonnenseite
            cloudLit
        );
        // Sonne durch dünne Wolken
        float sunThrough = dot(dir, uSunDir);
        cloudColor += vec3(1.0, 0.9, 0.7) * pow(max(sunThrough, 0.0), 12.0) * 0.15;

        vec3 cirrusColor = mix(vec3(0.82, 0.87, 0.95), vec3(1.0, 0.98, 0.94), cloudLit);

        skyColor = mix(skyColor, cloudColor,  cloud);
        skyColor = mix(skyColor, cirrusColor, cirrus);
    }

    // ── Sonne ─────────────────────────────────────────────────────────────
    float sunDot  = dot(dir, uSunDir);
    // Lichtglocke / Glühhalo
    float halo    = pow(max(sunDot, 0.0), 6.0);
    float glowWide = pow(max(sunDot, 0.0), 64.0);
    skyColor += uSunDir.y > -0.05
        ? vec3(1.0, 0.80, 0.45) * halo    * 0.55
        : vec3(1.0, 0.40, 0.10) * halo    * 0.40;   // Sonnenuntergang
    skyColor += vec3(1.1, 1.0, 0.8) * glowWide * 0.8;

    // Sonnenscheibe
    float sunDisk = step(0.99985, sunDot);
    skyColor += vec3(2.5, 2.2, 1.6) * sunDisk;

    // ── Tonemapping + Gamma ───────────────────────────────────────────────
    skyColor  = skyColor / (skyColor + vec3(0.6));   // Reinhard-ähnlich
    skyColor  = pow(max(skyColor, vec3(0.0)), vec3(1.0 / 2.2));

    FragColor = vec4(skyColor, 1.0);
}
"""


class Sky:
    """
    Prozeduraler Himmel-Renderer.
    Wird von der Engine automatisch aufgerufen wenn engine.sky gesetzt ist.
    """

    def __init__(
        self,
        sun_elevation: float = 45.0,
        sun_azimuth:   float = 135.0,
        cloud_density: float = 0.45,
        cloud_speed:   float = 0.008,
        zenith_color:  Tuple[float, float, float] = (0.07, 0.22, 0.68),
        horizon_color: Tuple[float, float, float] = (0.52, 0.75, 0.94),
    ) -> None:
        self.sun_elevation:  float = sun_elevation
        self.sun_azimuth:    float = sun_azimuth
        self.cloud_density:  float = cloud_density
        self.cloud_speed:    float = cloud_speed
        self.zenith_color:   Tuple[float, float, float] = zenith_color
        self.horizon_color:  Tuple[float, float, float] = horizon_color

        self.directional_light: DirectionalLight = DirectionalLight(
            direction=self.get_light_direction(),
            color=Color(1.0, 0.95, 0.85)
        )

        self._prog: Optional[int] = None
        self._vao:  Optional[int] = None

    # ── GPU-Initialisierung (lazy) ────────────────────────────────────────────
    def _ensure_gpu(self) -> None:
        if self._prog is not None:
            return
        vert = compileShader(_SKY_VERT, GL_VERTEX_SHADER)
        frag = compileShader(_SKY_FRAG, GL_FRAGMENT_SHADER)
        self._prog = compileProgram(vert, frag)
        self._vao  = glGenVertexArrays(1)   # leeres VAO für den Fullscreen-Draw

    def _sun_direction(self) -> Vec3:
        """Normalisierter Vektor der zur Sonne zeigt."""
        el  = math.radians(self.sun_elevation)
        az  = math.radians(self.sun_azimuth)
        x   = math.cos(el) * math.sin(az)
        y   = math.sin(el)
        z   = math.cos(el) * math.cos(az)
        return Vec3(x, y, z)

    # ── Render (aufgerufen von Engine vor scene._draw_all) ────────────────────
    def render(
        self,
        view_mat:  np.ndarray,
        proj_mat:  np.ndarray,
        time:      float,
        width:     int,
        height:    int,
    ) -> None:
        self._ensure_gpu()

        # Inverse View-Projection Matrix
        vp     = proj_mat @ view_mat
        inv_vp = np.linalg.inv(vp).astype(np.float32)

        sun = self._sun_direction()

        # Kein Depth-Test → Himmel landet immer im Hintergrund
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glDepthMask(GL_FALSE)

        prog = self._prog
        glUseProgram(prog)

        glUniformMatrix4fv(glGetUniformLocation(prog, "uInvVP"),       1, GL_TRUE, inv_vp)
        glUniform3f(glGetUniformLocation(prog, "uSunDir"),             sun.x, sun.y, sun.z)
        glUniform1f(glGetUniformLocation(prog, "uTime"),               time)
        glUniform1f(glGetUniformLocation(prog, "uCloudDensity"),       self.cloud_density)
        glUniform1f(glGetUniformLocation(prog, "uCloudSpeed"),         self.cloud_speed)
        glUniform3f(glGetUniformLocation(prog, "uZenithColor"),        *self.zenith_color)
        glUniform3f(glGetUniformLocation(prog, "uHorizonColor"),       *self.horizon_color)
        glUniform2f(glGetUniformLocation(prog, "uResolution"),         float(width), float(height))

        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)

        glUseProgram(0)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

    def get_light_direction(self) -> Vec3:
        sun = self._sun_direction()
        return Vec3(-sun.x, -sun.y, -sun.z)

# ── Vorgefertigte Presets ─────────────────────────────────────────────────────

def sky_day() -> Sky:
    """Klarer Sommertag."""
    return Sky(sun_elevation=55.0, sun_azimuth=135.0,
               cloud_density=0.35,
               zenith_color=(0.07, 0.22, 0.68),
               horizon_color=(0.52, 0.75, 0.94))

def sky_overcast() -> Sky:
    """Bedeckter Himmel."""
    return Sky(sun_elevation=40.0, sun_azimuth=180.0,
               cloud_density=0.85,
               zenith_color=(0.30, 0.33, 0.38),
               horizon_color=(0.55, 0.57, 0.60))

def sky_sunset() -> Sky:
    """Sonnenuntergang / Abenddämmerung."""
    return Sky(sun_elevation=4.0, sun_azimuth=270.0,
               cloud_density=0.4,
               zenith_color=(0.08, 0.10, 0.35),
               horizon_color=(0.90, 0.42, 0.10))

def sky_dawn() -> Sky:
    """Morgendämmerung."""
    return Sky(sun_elevation=6.0, sun_azimuth=90.0,
               cloud_density=0.3,
               zenith_color=(0.12, 0.15, 0.45),
               horizon_color=(0.95, 0.65, 0.35))

def sky_noon_clear() -> Sky:
    """Mittagssonne, fast wolkenlos."""
    return Sky(sun_elevation=80.0, sun_azimuth=180.0,
               cloud_density=0.10,
               zenith_color=(0.04, 0.18, 0.72),
               horizon_color=(0.48, 0.72, 0.95))