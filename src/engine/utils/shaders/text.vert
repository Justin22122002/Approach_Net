#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform vec2 uSize;
uniform int  uBillboard;

out vec2 vUV;

void main() {
    if (uBillboard != 0) {
        vec3 worldPos = uModel[3].xyz;

        vec3 right = vec3(uView[0][0], uView[1][0], uView[2][0]);
        vec3 up    = vec3(uView[0][1], uView[1][1], uView[2][1]);

        vec3 pos = worldPos
                 + right * aPos.x * uSize.x
                 + up    * aPos.y * uSize.y;

        gl_Position = uProjection * uView * vec4(pos, 1.0);
    } else {
        vec4 local  = vec4(aPos.x * uSize.x, aPos.y * uSize.y, 0.0, 1.0);
        gl_Position = uProjection * uView * uModel * local;
    }
    vUV = aUV;
}