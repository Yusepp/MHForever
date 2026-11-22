#version 460

layout(push_constant) uniform BlurPush {
    vec2 uv_off;
    vec2 uv_scale;
    vec2 dir;
    float dim;
    uint tex_idx;
    uint sampler_idx;
} pc;

layout(location = 0) out vec2 v_uv;

void main() {
    vec2 p = vec2((gl_VertexIndex & 1) * 2.0, (gl_VertexIndex & 2) * 1.0);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    v_uv = pc.uv_off + p * pc.uv_scale;
}
