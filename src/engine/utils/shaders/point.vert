#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4  uModel;
uniform mat4  uView;
uniform mat4  uProjection;
uniform float uPointSize;

out vec3 vColor;

void main() {
    gl_Position  = uProjection * uView * uModel * vec4(aPos, 1.0);
    gl_PointSize = uPointSize / gl_Position.w;   // perspective scaling of the point
    vColor = aColor;
}