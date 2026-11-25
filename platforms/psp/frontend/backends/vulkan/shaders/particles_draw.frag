#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// Premultiplied-alpha HDR Particle Fragment Shader:
// - Outputs vec4(rgb * alpha, alpha) for soft snow/pollen/ash (alpha-blended)
// - Outputs vec4(rgb * alpha, 0.0)   for glowing embers/fireflies/sparks (additive HDR glow!)

layout(set = 0, binding = 0) uniform texture2D u_textures[];

struct GpuParticle {
    vec4 pos_life;
    vec4 vel_size;
    vec4 color_kind;
    vec4 aux;
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer ParticleBuf {
    GpuParticle items[];
};

struct DrawParams {
    mat4 view;
    mat4 proj;
    vec4 cam_right;
    vec4 cam_up;
    vec4 screen_info;
    uvec4 slots; // x = soft_depth_slot
    ParticleBuf particles;
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer DrawParamsBuf {
    DrawParams p;
};

layout(push_constant) uniform DrawPush {
    DrawParamsBuf buf;
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) in vec4 v_meta; // x = kind, y = additive, z = psp_z, w = seed

layout(location = 0) out vec4 o_color;

void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 >= 1.0) discard;

    float kind = v_meta.x;
    float shape = 0.0;
    if (kind < 0.5) {
        // Soft glowing ember / firefly core + smooth halo.
        float core = exp(-r2 * 6.5);
        float halo = (1.0 - r2) * (1.0 - r2) * 0.45;
        shape = clamp(core + halo, 0.0, 1.0);
    } else if (kind < 1.5) {
        // Crisp snowflake / ice crystal with soft edge.
        float r = sqrt(r2);
        shape = smoothstep(1.0, 0.25, r);
    } else if (kind < 2.5) {
        // Soft pollen / dust mote / ash flake.
        shape = (1.0 - r2) * (1.0 - r2);
    } else {
        // Velocity-stretched spark / sand streak (tapered ellipse).
        float core = exp(-(v_uv.x * v_uv.x * 5.5 + v_uv.y * v_uv.y * 2.2));
        shape = core * (1.0 - r2);
    }

    float alpha = shape * v_color.a;
    if (alpha < 0.004) discard;

    // Soft-depth fade against opaque geometry when soft_depth_copy_ is available.
    uint sd_slot = pc.buf.p.slots.x;
    if (sd_slot != 0u) {
        ivec2 ip = ivec2(gl_FragCoord.xy);
        ivec2 sz = textureSize(u_textures[sd_slot], 0);
        float scene_d = texelFetch(u_textures[sd_slot], clamp(ip, ivec2(0), sz - ivec2(1)), 0).r;
        float soft = smoothstep(0.0, 0.0012, scene_d - v_meta.z);
        alpha *= soft;
        if (alpha < 0.004) discard;
    }

    // Premultiplied alpha: additive particles use out_a = 0.0 so they add HDR light without darkening behind!
    float out_a = (v_meta.y > 0.5) ? 0.0 : alpha;
    o_color = vec4(v_color.rgb * alpha, out_a);
}
