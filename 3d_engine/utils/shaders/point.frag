#version 330 core
in  vec3 vColor;
out vec4 FragColor;

void main() {
    // round point instead of square
    vec2  coord = gl_PointCoord * 2.0 - 1.0;
    float dist  = dot(coord, coord);
    if (dist > 1.0) discard;

    // soft edge, comment out for sharp edge
    float alpha = 1.0 - smoothstep(0.7, 1.0, dist);
    FragColor = vec4(vColor, alpha);
}