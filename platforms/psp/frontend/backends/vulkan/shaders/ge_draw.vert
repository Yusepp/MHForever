#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

struct DrawParams {
    vec4 uv_xform;      // scale.xy, offset.zw
    vec4 env_and_flags; // env.rgb, flags (bits)
    vec4 fog_and_zmin;  // fog_color.rgb, zrange_min
    vec2 target_size;   // render target size in PSP pixels
    vec2 zmax_pad;      // zrange_max, unused
    uint tex_idx;
    uint sampler_idx;
    uint tex_func_bits; // bits 0..2: tex_func, bit 3: tex_rgba, bit 4: tex_double, bit 5: clear, bit 6: tex_enable, bit 7: fog_enable, bit 8: zrange_enable
    uint alpha_bits;    // bits 0..2: alpha_func, bits 8..15: alpha_ref, bits 16..23: alpha_mask
    vec4 jitter_and_lod; // xy = NDC jitter (TAA), z = mip LOD bias, w = unused
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer DrawParamsBuf {
    DrawParams items[];
};

layout(push_constant) uniform DrawPush {
    DrawParamsBuf params;
    uint draw_idx;
} pc;

layout(location = 0) in vec4 a_pos;   // x, y in PSP pixels, z 0..65535, 1/w
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;
layout(location = 3) in vec4 a_spec;
layout(location = 4) in float a_fog;
layout(location = 5) in uint a_normal_oct;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;
layout(location = 2) out vec3 v_spec;
layout(location = 3) out float v_fog;
layout(location = 4) out vec3 v_normal;
layout(location = 5) out float v_has_normal;

vec3 unpack_oct_normal(uint packed) {
    vec2 oct = unpackSnorm2x16(packed);
    vec3 n = vec3(oct.x, oct.y, 1.0 - abs(oct.x) - abs(oct.y));
    float t = max(-n.z, 0.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

void main() {
    DrawParams p = pc.params.items[pc.draw_idx];
    float w = 1.0 / a_pos.w;
    // Vulkan clip space: X in [-1, 1], Y in [-1, 1] (top = -1, bottom = +1, matching PSP row 0 at top), Z in [0, 1].
    vec2 ndc = a_pos.xy / p.target_size * 2.0 - 1.0 + p.jitter_and_lod.xy;
    float z = clamp(a_pos.z / 65535.0, 0.0, 1.0);
    gl_Position = vec4(ndc * w, z * w, w);
    v_uv = a_uv * p.uv_xform.xy + p.uv_xform.zw;
    v_color = a_color;
    v_spec = a_spec.rgb;
    v_fog = a_fog;
    if (a_normal_oct != 0u) {
        v_normal = unpack_oct_normal(a_normal_oct);
        v_has_normal = 1.0;
    } else {
        v_normal = vec3(0.0, 0.0, 1.0);
        v_has_normal = 0.0;
    }
}
