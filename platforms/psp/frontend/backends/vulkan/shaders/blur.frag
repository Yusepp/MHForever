#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform texture2D u_textures[];
layout(set = 0, binding = 1) uniform sampler u_samplers[];

layout(push_constant) uniform BlurPush {
    vec2 uv_off;
    vec2 uv_scale;
    vec2 dir;
    float dim;
    uint tex_idx;
    uint sampler_idx;
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

vec3 tap(vec2 uv) {
    return texture(sampler2D(u_textures[pc.tex_idx],
                             u_samplers[pc.sampler_idx]), uv).rgb;
}

void main() {
    vec3 c = tap(v_uv) * 0.2270270270;
    c += tap(v_uv + pc.dir * 1.3846153846) * 0.3162162162;
    c += tap(v_uv - pc.dir * 1.3846153846) * 0.3162162162;
    c += tap(v_uv + pc.dir * 3.2307692308) * 0.0702702703;
    c += tap(v_uv - pc.dir * 3.2307692308) * 0.0702702703;
    o_color = vec4(c * pc.dim, 1.0);
}
