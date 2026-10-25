#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;
in vec4 vFragPosLightSpace;

uniform sampler2D uTexture;
uniform vec3  uAmbient;
uniform vec3  uLightDir;
uniform vec3  uLightColor;
uniform float uLightIntensity;
uniform vec4  uColor;

uniform sampler2D uShadowMap;
uniform int   uShadowsEnabled;
uniform float uDepthBiasMin;
uniform float uDepthBiasMax;

out vec4 FragColor;

const vec2 POISSON_DISK[16] = vec2[](
    vec2(-0.94201624, -0.39906216), vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870), vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845), vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554), vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507), vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367), vec2( 0.14383161, -0.14100790)
);

// cheap per-fragment hash -> rotation angle, breaks the sample pattern up into
// noise instead of visible banding steps
float _shadowRandomAngle(vec3 seed) {
    float d = dot(seed, vec3(12.9898, 78.233, 45.164));
    return fract(sin(d) * 43758.5453) * 6.2831853;
}

float calcShadow(vec3 normal, vec3 lightDir) {
    if (uShadowsEnabled == 0) return 0.0;

    vec3 proj = vFragPosLightSpace.xyz / vFragPosLightSpace.w;
    proj = proj * 0.5 + 0.5;

    if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0)
        return 0.0;

    float bias = max(uDepthBiasMax * (1.0 - dot(normal, lightDir)), uDepthBiasMin);

    vec2 texelSize = 1.0 / vec2(textureSize(uShadowMap, 0));
    float angle = _shadowRandomAngle(vWorldPos);
    float s = sin(angle), c = cos(angle);
    mat2 rot = mat2(c, -s, s, c);

    float shadow = 0.0;
    for (int i = 0; i < 16; i++) {
        vec2 offset = rot * POISSON_DISK[i] * 4.0 * texelSize;
        float pcfDepth = texture(uShadowMap, proj.xy + offset).r;
        shadow += (proj.z - bias > pcfDepth) ? 1.0 : 0.0;
    }
    return shadow / 16.0;
}

void main() {
    vec4 texColor = texture(uTexture, vUV) * uColor;

    vec3  normal  = normalize(vNormal);
    vec3  ld      = normalize(-uLightDir);
    float diff    = max(dot(normal, ld), 0.0);
    vec3  diffuse = diff * uLightColor * uLightIntensity;

    float shadow = calcShadow(normal, ld);

    vec3 result = (uAmbient + (1.0 - shadow) * diffuse) * texColor.rgb;
    FragColor   = vec4(result, texColor.a);
}
