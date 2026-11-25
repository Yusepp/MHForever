#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// Depth-only Sun Shadow Map fragment shader (alpha-test cutout for trees, grass, fences).

layout(set = 0, binding = 0) uniform texture2D u_textures[];
layout(set = 0, binding = 1) uniform sampler u_samplers[];

struct ShadowDrawParams {
    mat4 shadow_vp;
    vec4 uv_xform;
    uvec4 tex_alpha; // x = tex_slot, y = sampler_idx, z = alpha_bits, w = textured
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer ShadowDrawParamsBuf {
    ShadowDrawParams p;
};

layout(push_constant) uniform ShadowPush {
    ShadowDrawParamsBuf buf;
} pc;

layout(location = 0) in vec2 v_uv;

bool alpha_compare(uint f, uint a, uint b) {
    if (f == 0u) return false;
    if (f == 1u) return true;
    if (f == 2u) return a == b;
    if (f == 3u) return a != b;
    if (f == 4u) return a < b;
    if (f == 5u) return a <= b;
    if (f == 6u) return a > b;
    return a >= b;
}

void main() {
    ShadowDrawParams p = pc.buf.p;
    uint af = p.tex_alpha.z & 7u;
    if (af != 1u && p.tex_alpha.w != 0u && p.tex_alpha.x != 0u) {
        float ta = texture(sampler2D(u_textures[nonuniformEXT(p.tex_alpha.x)],
                                     u_samplers[nonuniformEXT(p.tex_alpha.y)]),
                           v_uv).a;
        uint mask = (p.tex_alpha.z >> 16u) & 0xFFu;
        uint ref = ((p.tex_alpha.z >> 8u) & 0xFFu) & mask;
        uint a = uint(clamp(ta, 0.0, 1.0) * 255.0 + 0.5) & mask;
        if (!alpha_compare(af, a, ref)) discard;
    }
}
