#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// Depth-only orthographic Sun Shadow Map vertex shader (world-space cm -> shadow clip space).

struct ShadowDrawParams {
    mat4 shadow_vp;   // world -> shadow clip space (Vulkan Z in [0, 1])
    vec4 uv_xform;    // scale.xy, offset.zw
    uvec4 tex_alpha;  // x = tex_slot, y = sampler_idx, z = alpha_bits, w = textured (0 or 1)
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer ShadowDrawParamsBuf {
    ShadowDrawParams p;
};

layout(push_constant) uniform ShadowPush {
    ShadowDrawParamsBuf buf;
} pc;

layout(location = 0) in vec3 a_world;
layout(location = 1) in vec2 a_uv;

layout(location = 0) out vec2 v_uv;

void main() {
    ShadowDrawParams p = pc.buf.p;
    gl_Position = p.shadow_vp * vec4(a_world, 1.0);
    v_uv = a_uv * p.uv_xform.xy + p.uv_xform.zw;
}
