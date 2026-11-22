#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform texture2D u_textures[];
layout(set = 0, binding = 1) uniform sampler u_samplers[];

struct DrawParams {
    vec4 uv_xform;
    vec4 env_and_flags;
    vec4 fog_and_zmin;
    vec2 target_size;
    vec2 zmax_pad;
    uint tex_idx;
    uint sampler_idx;
    uint tex_func_bits;
    uint alpha_bits;
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer DrawParamsBuf {
    DrawParams items[];
};

layout(push_constant) uniform DrawPush {
    DrawParamsBuf params;
    uint draw_idx;
} pc;

// Specialization constants eliminate divergent branches for the common opaque/textured/untextured/clear cases.
layout(constant_id = 0) const uint kSpecClear = 0u;
layout(constant_id = 1) const uint kSpecTextured = 1u;
layout(constant_id = 2) const uint kSpecAlphaTest = 1u;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) in vec3 v_spec;
layout(location = 3) in float v_fog;

layout(location = 0) out vec4 o_color;

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
    vec4 c = v_color;
    if (kSpecClear == 0u) {
        DrawParams p = pc.params.items[pc.draw_idx];
        uint bits = p.tex_func_bits;
        if (kSpecTextured != 0u && ((bits & 0x40u) != 0u)) {
            vec4 t = texture(sampler2D(u_textures[nonuniformEXT(p.tex_idx)],
                                       u_samplers[nonuniformEXT(p.sampler_idx)]), v_uv);
            uint func = bits & 7u;
            bool rgba = (bits & 8u) != 0u;
            if (func == 0u) { c.rgb *= t.rgb; if (rgba) c.a *= t.a; }
            else if (func == 1u) { c.rgb = rgba ? mix(c.rgb, t.rgb, t.a) : t.rgb; }
            else if (func == 2u) { c.rgb = mix(c.rgb, p.env_and_flags.rgb, t.rgb); if (rgba) c.a *= t.a; }
            else if (func == 3u) { c.rgb = t.rgb; if (rgba) c.a = t.a; }
            else { c.rgb += t.rgb; if (rgba) c.a *= t.a; }
            if ((bits & 16u) != 0u) c.rgb *= 2.0;
        }
        c.rgb = min(c.rgb + v_spec, vec3(1.0));
        if ((bits & 0x80u) != 0u) {
            c.rgb = mix(p.fog_and_zmin.rgb, c.rgb, clamp(v_fog, 0.0, 1.0));
        }
        if (kSpecAlphaTest != 0u) {
            uint af = p.alpha_bits & 7u;
            if (af != 1u) {
                uint mask = (p.alpha_bits >> 16u) & 0xFFu;
                uint ref = ((p.alpha_bits >> 8u) & 0xFFu) & mask;
                uint a = uint(clamp(c.a, 0.0, 1.0) * 255.0 + 0.5) & mask;
                if (!alpha_compare(af, a, ref)) discard;
            }
        }
        if ((bits & 0x100u) != 0u) {
            float z = gl_FragCoord.z * 65535.0;
            if (z < p.fog_and_zmin.w || z > p.zmax_pad.x + 0.5) discard;
        }
    }
    o_color = clamp(c, 0.0, 1.0);
}
