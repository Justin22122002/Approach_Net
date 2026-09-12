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