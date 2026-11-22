#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture2D u_textures[];

layout(push_constant) uniform ShadowMapPush {
    vec4 center_push;
    vec4 right_scenery;
    vec4 up_pad;
    vec4 dir_pad;
    vec4 half_pad;
    vec4 uv_xform;
    uint tex_idx;
    uint sampler_idx;
    uint alpha_bits;
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_kind;

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
    uint af = pc.alpha_bits & 7u;
    if (af != 1u) {
        ivec2 sz = textureSize(u_textures[nonuniformEXT(pc.tex_idx)], 0);
        ivec2 t = ivec2(floor((v_uv * pc.uv_xform.xy + pc.uv_xform.zw) * vec2(sz)));
        bool clamp_s = (pc.alpha_bits & (1u << 24u)) != 0u;
        bool clamp_t = (pc.alpha_bits & (1u << 25u)) != 0u;
        t.x = clamp_s ? clamp(t.x, 0, sz.x - 1) : ((t.x % sz.x) + sz.x) % sz.x;
        t.y = clamp_t ? clamp(t.y, 0, sz.y - 1) : ((t.y % sz.y) + sz.y) % sz.y;
        uint mask = (pc.alpha_bits >> 16u) & 0xFFu;
        uint ref = ((pc.alpha_bits >> 8u) & 0xFFu) & mask;
        uint a = uint(clamp(texelFetch(u_textures[nonuniformEXT(pc.tex_idx)], t, 0).a, 0.0, 1.0) * 255.0 + 0.5) & mask;
        if (!alpha_compare(af, a, ref)) discard;
    }
    float slope = max(abs(dFdx(gl_FragCoord.z)), abs(dFdy(gl_FragCoord.z)));
    gl_FragDepth = gl_FragCoord.z + min(3.0 * slope, pc.center_push.w);
    o_kind = vec4(pc.right_scenery.w);
}
