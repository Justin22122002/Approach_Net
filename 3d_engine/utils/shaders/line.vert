#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vColor;

void main() {
    vec4 pos    = uProjection * uView * uModel * vec4(aPos, 1.0);
    vColor      = aColor;
    pos.z -= 0.0001 * pos.w; // this serves as a prevention for lines interseting with meshes
    gl_Position = pos;
}