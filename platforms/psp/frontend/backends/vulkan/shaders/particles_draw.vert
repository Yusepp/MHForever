#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// Instanced 120/180 Hz HDR World-Space Particle Vertex Shader (billboards + velocity-stretched streaks).

struct GpuParticle {
    vec4 pos_life;   // xyz = world pos (cm), w = life remaining (s)
    vec4 vel_size;   // xyz = world vel (cm/s), w = world radius (cm)
    vec4 color_kind; // rgb = linear HDR color, a = visual kind (0=glow, 1=snow, 2=mote/ash, 3=streak)
    vec4 aux;        // x = max_life, y = seed, z = gravity_scale, w = additive (0=alpha, 1=additive)
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer ParticleBuf {
    GpuParticle items[];
};

struct DrawParams {
    mat4 view;         // world -> PSP view space
    mat4 proj;         // PSP view -> Vulkan clip space
    vec4 cam_right;    // xyz = world-space camera right vector, w = vz_scale
    vec4 cam_up;       // xyz = world-space camera up vector, w = vz_center
    vec4 screen_info;  // xy = 1/width, 1/height, z = time, w = global_alpha
    uvec4 slots;       // x = soft_depth_slot, y = unused, z = unused, w = unused
    ParticleBuf particles;
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer DrawParamsBuf {
    DrawParams p;
};

layout(push_constant) uniform DrawPush {
    DrawParamsBuf buf;
} pc;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;    // rgb = linear HDR color, a = fade alpha
layout(location = 2) out vec4 v_meta;     // x = kind, y = additive, z = clip_depth, w = seed

const vec2 kQuad[6] = vec2[6](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

void main() {
    DrawParams p = pc.buf.p;
    GpuParticle pt = p.particles.items[gl_InstanceIndex];
    if (pt.pos_life.w <= 0.0 || pt.vel_size.w <= 0.001) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        v_uv = vec2(0.0);
        v_color = vec4(0.0);
        v_meta = vec4(0.0);
        return;
    }

    vec4 view_pos = p.view * vec4(pt.pos_life.xyz, 1.0);
    // Reject behind camera or too close to near plane (prevents giant lens-flare blobs!).
    float dist = -view_pos.z;
    if (dist < 75.0 || dist > 2600.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        v_uv = vec2(0.0);
        v_color = vec4(0.0);
        v_meta = vec4(0.0);
        return;
    }

    vec2 corner = kQuad[gl_VertexIndex];
    float rad = pt.vel_size.w;
    float kind = pt.color_kind.a;

    vec3 world_offset;
    if (kind > 2.5) {
        // Velocity-stretched spark / blowing sand streak!
        vec3 vel = pt.vel_size.xyz;
        float speed = length(vel);
        vec3 axis = (speed > 8.0) ? (vel / speed) : p.cam_up.xyz;
        vec3 cam_fwd = normalize(cross(p.cam_right.xyz, p.cam_up.xyz));
        vec3 side = cross(axis, cam_fwd);
        float sl = length(side);
        side = (sl > 1e-4) ? (side / sl) : p.cam_right.xyz;
        float stretch = clamp(1.4 + speed * 0.022, 1.5, 5.5);
        world_offset = side * (corner.x * rad * 0.55) + axis * (corner.y * rad * stretch);
    } else {
        // Camera-facing billboard with gentle spin for snow/ash.
        float ang = (kind > 0.5) ? (pt.aux.y + p.screen_info.z * 0.8) : 0.0;
        float c = cos(ang);
        float s = sin(ang);
        vec2 rot = vec2(corner.x * c - corner.y * s, corner.x * s + corner.y * c);
        world_offset = p.cam_right.xyz * (rot.x * rad) + p.cam_up.xyz * (rot.y * rad);
    }

    vec4 clip = p.proj * (p.view * vec4(pt.pos_life.xyz + world_offset, 1.0));
    // Match GE's PSP depth mapping: ndc_z -> [0, 65535] -> [0, 1] so hardware depth test aligns 1:1!
    float ndc_z = clip.z / max(clip.w, 1e-5);
    float psp_z = clamp((ndc_z * p.cam_right.w + p.cam_up.w) / 65535.0, 0.0, 1.0);
    clip.z = psp_z * clip.w;
    gl_Position = clip;

    // Smooth fade-in, fade-out, near-camera fade, and far-distance fade.
    float life_frac = clamp(pt.pos_life.w / max(pt.aux.x, 0.05), 0.0, 1.0);
    float fade_in = smoothstep(1.0, 0.84, life_frac);
    float fade_out = smoothstep(0.0, 0.22, life_frac);
    float near_fade = smoothstep(75.0, 185.0, dist);
    float far_fade = smoothstep(2500.0, 1400.0, dist);
    // Gentle firefly/ember flicker.
    float pulse = (kind < 0.5) ? (0.72 + 0.28 * sin(p.screen_info.z * 5.2 + pt.aux.y * 3.0)) : 1.0;

    v_uv = corner;
    v_color = vec4(pt.color_kind.rgb, fade_in * fade_out * near_fade * far_fade * pulse * p.screen_info.w);
    v_meta = vec4(kind, pt.aux.w, psp_z, pt.aux.y);
}
