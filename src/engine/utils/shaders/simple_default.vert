#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform mat3 uNormalMat;
uniform mat4 uLightSpaceMatrix;
uniform float uNormalBias;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;
out vec4 vFragPosLightSpace;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal   = normalize(uNormalMat * aNormal);
    vUV       = aUV;

    vFragPosLightSpace = uLightSpaceMatrix * vec4(vWorldPos + vNormal * uNormalBias, 1.0);

    gl_Position = uProjection * uView * worldPos;
}
