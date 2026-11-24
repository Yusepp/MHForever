#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform texture2D u_textures[];
layout(set = 0, binding = 1) uniform sampler u_samplers[];

layout(push_constant) uniform PresentPush {
    vec2 uv_max;
    uint tex_idx;
    uint sampler_idx;
    float alpha;
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main() {
    vec3 rgb = texture(sampler2D(u_textures[pc.tex_idx],
                                 u_samplers[pc.sampler_idx]), v_uv).rgb;
    o_color = vec4(rgb, 1.0);
}
