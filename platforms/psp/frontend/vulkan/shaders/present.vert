#version 460

layout(push_constant) uniform PresentPush {
    vec2 uv_max;
    uint tex_idx;
    uint sampler_idx;
    float alpha;
} pc;

layout(location = 0) out vec2 v_uv;

void main() {
    // Full-screen triangle (3 vertices, no VBO). Row 0 at top (Vulkan Y-down NDC).
    vec2 p = vec2((gl_VertexIndex & 1) * 2.0, (gl_VertexIndex & 2) * 1.0);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    v_uv = p * pc.uv_max;
}
