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