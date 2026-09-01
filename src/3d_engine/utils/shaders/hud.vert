#version 330 core
out vec2 vUV;
void main() {
    // Full-screen triangle trick — no VBO needed
    vec2 pos = vec2((gl_VertexID & 1) * 4.0 - 1.0,
                    (gl_VertexID & 2) * 2.0 - 1.0);
    vUV = pos * 0.5 + 0.5;
    vUV.y = 1.0 - vUV.y;   // flip Y so pygame top-left matches GL bottom-left
    gl_Position = vec4(pos, 0.0, 1.0);
}