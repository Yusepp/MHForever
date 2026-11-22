#version 460

layout(push_constant) uniform ShadowMapPush {
    vec4 center_push; // xyz = sun box center, w = push_limit
    vec4 right_scenery; // xyz = right axis, w = scenery (0 or 1)
    vec4 up_pad;      // xyz = up axis
    vec4 dir_pad;     // xyz = dir to sun
    vec4 half_pad;    // xyz = half extents
    vec4 uv_xform;    // scale.xy, offset.zw
    uint tex_idx;
    uint sampler_idx;
    uint alpha_bits;  // bits 0..2: func, 8..15: ref, 16..23: mask, bit 24: clamp_s, bit 25: clamp_t
} pc;

layout(location = 0) in vec3 a_eye;
layout(location = 1) in vec2 a_uv;
layout(location = 0) out vec2 v_uv;

void main() {
    vec3 d = a_eye - pc.center_push.xyz;
    float z = clamp((-dot(d, pc.dir_pad.xyz) / pc.half_pad.z) * 0.5 + 0.5, 0.0, 1.0);
    gl_Position = vec4(dot(d, pc.right_scenery.xyz) / pc.half_pad.x,
                       dot(d, pc.up_pad.xyz) / pc.half_pad.y,
                       z, 1.0);
    v_uv = a_uv;
}
