#version 330 core
in  vec2 vUV;
out vec4 FragColor;

uniform sampler2D uTexture;
uniform vec4      uColor;

void main() {
    vec4 t = texture(uTexture, vUV);
    FragColor = t * uColor;
    if (FragColor.a < 0.02) discard;
}