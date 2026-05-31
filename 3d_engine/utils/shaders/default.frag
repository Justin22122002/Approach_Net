#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;

uniform sampler2D uTexture;
uniform vec4      uColor;

uniform vec3 uCamPos;

uniform vec3 uAmbient;

uniform int   uDirLightEnabled;
uniform vec3  uDirLightDir;
uniform vec3  uDirLightColor;
uniform float uDirLightIntensity;

#define MAX_POINT_LIGHTS 8
uniform int   uNumPointLights;
uniform vec3  uPointLightPos[MAX_POINT_LIGHTS];
uniform vec3  uPointLightColor[MAX_POINT_LIGHTS];
uniform float uPointLightIntensity[MAX_POINT_LIGHTS];
uniform float uPointLightA[MAX_POINT_LIGHTS];
uniform float uPointLightB[MAX_POINT_LIGHTS];

#define MAX_SPOT_LIGHTS 4
uniform int   uNumSpotLights;
uniform vec3  uSpotLightPos[MAX_SPOT_LIGHTS];
uniform vec3  uSpotLightDir[MAX_SPOT_LIGHTS];
uniform vec3  uSpotLightColor[MAX_SPOT_LIGHTS];
uniform float uSpotLightIntensity[MAX_SPOT_LIGHTS];
uniform float uSpotInnerCone[MAX_SPOT_LIGHTS];
uniform float uSpotOuterCone[MAX_SPOT_LIGHTS];

uniform float uSpecularStrength;
uniform float uShininess;

out vec4 FragColor;

float specular(vec3 lightDir, vec3 normal, vec3 viewDir) {
    vec3 reflDir = reflect(-lightDir, normal);
    float s = pow(max(dot(viewDir, reflDir), 0.0), uShininess);
    return s * uSpecularStrength;
}

vec3 calcDirectional(vec3 normal, vec3 viewDir, vec3 texRGB) {
    if (uDirLightEnabled == 0) return vec3(0.0);
    vec3  ld     = normalize(-uDirLightDir);
    float diff   = max(dot(normal, ld), 0.0);
    float spec   = specular(ld, normal, viewDir);
    return (diff + spec) * uDirLightColor * uDirLightIntensity * texRGB;
}

vec3 calcPointLight(int i, vec3 normal, vec3 viewDir, vec3 texRGB) {
    vec3  lightVec = uPointLightPos[i] - vWorldPos;
    float dist     = length(lightVec);
    float atten    = 1.0 / (uPointLightA[i] * dist * dist
                           + uPointLightB[i] * dist + 1.0);
    vec3  ld       = normalize(lightVec);
    float diff     = max(dot(normal, ld), 0.0);
    float spec     = specular(ld, normal, viewDir);
    return (diff + spec) * atten * uPointLightColor[i] * uPointLightIntensity[i] * texRGB;
}

vec3 calcSpotLight(int i, vec3 normal, vec3 viewDir, vec3 texRGB) {
    vec3  ld    = normalize(uSpotLightPos[i] - vWorldPos);
    float diff  = max(dot(normal, ld), 0.0);
    float spec  = specular(ld, normal, viewDir);

    float angle = dot(normalize(-uSpotLightDir[i]), ld);
    float inten = clamp(
        (angle - uSpotOuterCone[i]) / (uSpotInnerCone[i] - uSpotOuterCone[i]),
        0.0, 1.0
    );
    return (diff + spec) * inten * uSpotLightColor[i] * uSpotLightIntensity[i] * texRGB;
}

void main() {
    vec4 texColor = texture(uTexture, vUV) * uColor;
    vec3 texRGB   = texColor.rgb;

    vec3 normal  = normalize(vNormal);
    vec3 viewDir = normalize(uCamPos - vWorldPos);

    vec3 result = uAmbient * texRGB;

    result += calcDirectional(normal, viewDir, texRGB);

    for (int i = 0; i < uNumPointLights; i++)
        result += calcPointLight(i, normal, viewDir, texRGB);

    for (int i = 0; i < uNumSpotLights; i++)
        result += calcSpotLight(i, normal, viewDir, texRGB);

    FragColor = vec4(result, texColor.a);
}