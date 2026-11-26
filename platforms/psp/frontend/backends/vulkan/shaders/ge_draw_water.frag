#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

// Preserve the GE water shading and add reflections only where the scene trace hits.

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
    vec4 jitter_and_lod;
    uvec4 pbr_slots;     // xy = uvec2 BDA of WaterFrameParams, z = water_flags (bit 0 = shallow), w = mat_id
    vec4 sun_dir_view;   // xyz = view-space sun dir (+Y up), w = outdoor (1 or 0)
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer DrawParamsBuf {
    DrawParams items[];
};

struct WaterFrameParams {
    mat4 inv_view;        // column-major: PSP view (+Y up) -> area-space world (cm, +Y up)
    vec4 proj_info;       // x = 1/proj[0], y = 1/proj[5], z = proj[10], w = proj[14]
    vec4 vz_and_size;     // x = vz_scale, y = vz_offset, z = viewport_w, w = viewport_h
    vec4 sun_dir_view;    // xyz = view-space sun dir (+Y down), w = is_cave (1 or 0)
    vec4 world_up_view;   // xyz = view-space world up (+Y down), w = has_inv_view (1 or 0)
    vec4 sky_zenith;      // rgb = linear biome zenith sky reflection, w = elapsed_sec
    vec4 sky_horizon;     // rgb = linear biome horizon reflection, w = max_fresnel
    vec4 deep_col;        // rgb = linear biome deep-water scatter accent, w = wave_chop
    vec4 sigma_abs;       // rgb = Beer-Lambert absorption (1/cm), w = caustic_str
    uvec4 slots;          // x = water_bg_color_slot, y = water_bg_depth_slot, z = noise3d_slot, w = linear_sampler_idx
    uint ripple_count;
    uint pad0;
    uint pad1;
    uint pad2;
    vec4 ripples[24];     // xy = world (x, z), z = age_sec, w = radius_cm
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer WaterFrameBuf {
    WaterFrameParams wf;
};

layout(push_constant) uniform DrawPush {
    DrawParamsBuf params;
    uint draw_idx;
} pc;

layout(constant_id = 0) const uint kSpecClear = 0u;
layout(constant_id = 1) const uint kSpecTextured = 1u;
layout(constant_id = 2) const uint kSpecAlphaTest = 1u;
layout(constant_id = 3) const uint kSpecZRange = 0u;
layout(constant_id = 4) const uint kSpecHdr = 0u;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) in vec3 v_spec;
layout(location = 3) in float v_fog;
layout(location = 4) in vec3 v_normal;
layout(location = 5) in float v_has_normal;

layout(location = 0) out vec4 o_color;
layout(location = 1) out uint o_mat_id;

vec3 srgb_to_linear(vec3 c) {
    vec3 a = max(c, vec3(0.0));
    return a * (a * (a * 0.305306011 + 0.682171111) + 0.012522878);
}

vec3 linear_to_srgb(vec3 c) {
    vec3 a = max(c, vec3(0.0));
    vec3 s1 = sqrt(a);
    vec3 s2 = sqrt(s1);
    vec3 s3 = sqrt(s2);
    return clamp(0.662002687 * s1 + 0.684122060 * s2 - 0.323583601 * s3 - 0.0225411470 * a, vec3(0.0), vec3(1.0));
}

float linear_eye_z(WaterFrameParams wf, float d) {
    float zn = (d * 65535.0 - wf.vz_and_size.y) / max(abs(wf.vz_and_size.x), 1e-3);
    float denom = zn + wf.proj_info.z;
    if (abs(denom) < 1e-5) denom = (denom >= 0.0 ? 1e-5 : -1e-5);
    return clamp(abs(wf.proj_info.w / denom), 0.1, 20000.0);
}

float ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// Smooth, glassy 4-wave analytic directional Gerstner/sine slope (pure ALU, 0 texture fetches!).
// Produces clean, natural ocean/lake swells and readable mirror-like reflections with zero fog-noise static.
vec2 eval_analytic_wave_slope(vec2 wxz, float t, float chop, float close_fade) {
    // 4 crossing wave trains (directions, spatial frequencies in rad/cm, phase speeds, gentle amplitudes).
    const vec2 d0 = vec2( 0.866,  0.500);
    const vec2 d1 = vec2(-0.423,  0.906);
    const vec2 d2 = vec2( 0.940, -0.342);
    const vec2 d3 = vec2(-0.766, -0.643);

    float p0 = dot(wxz, d0) * 0.0155 + t * 1.15;
    float p1 = dot(wxz, d1) * 0.0245 - t * 1.45;
    float p2 = dot(wxz, d2) * 0.0430 + t * 1.95;
    float p3 = dot(wxz, d3) * 0.0780 - t * 2.55;

    // Gentle asymmetric swell (trochoidal-like sharpening via sin + 0.22*sin(2p)) for crisp, readable 3D waves.
    float g0 = (cos(p0) + 0.24 * cos(2.0 * p0)) * 0.115;
    float g1 = (cos(p1) + 0.20 * cos(2.0 * p1)) * 0.088;
    float g2 = cos(p2) * 0.058;
    float g3 = cos(p3) * (0.036 * close_fade) * (1.0 - smoothstep(1.0, 3.14, fwidth(p3)));

    vec2 slope = (d0 * g0 + d1 * g1 + d2 * g2 + d3 * g3) * chop;
    return clamp(slope, vec2(-0.28), vec2(0.28));
}

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
    DrawParams p = pc.params.items[pc.draw_idx];
    uint mat_id = p.pbr_slots.w & 0xFu;
    bool shallow = (p.pbr_slots.z & 1u) != 0u;
    bool cutout_puddle = (p.pbr_slots.z & 2u) != 0u;

    if (kSpecZRange != 0u && ((p.tex_func_bits & 0x100u) != 0u)) {
        float z_psp = gl_FragCoord.z * 65535.0;
        if (z_psp < p.fog_and_zmin.w || z_psp > p.zmax_pad.x + 0.5) discard;
    }

    // Sample MHFU's original water texture & vertex color so we RESPECT each map's exact palette & art style!
    vec4 orig_c = v_color;
    if (kSpecTextured != 0u && ((p.tex_func_bits & 0x40u) != 0u)) {
        vec4 t_orig = texture(sampler2D(u_textures[p.tex_idx], u_samplers[p.sampler_idx]), v_uv, p.jitter_and_lod.z);
        uint func = p.tex_func_bits & 7u;
        bool rgba = (p.tex_func_bits & 8u) != 0u;
        if (func == 0u) { orig_c.rgb *= t_orig.rgb; if (rgba) orig_c.a *= t_orig.a; }
        else if (func == 1u) { orig_c.rgb = rgba ? mix(orig_c.rgb, t_orig.rgb, t_orig.a) : t_orig.rgb; }
        else if (func == 2u) { orig_c.rgb = mix(orig_c.rgb, p.env_and_flags.rgb, t_orig.rgb); if (rgba) orig_c.a *= t_orig.a; }
        else if (func == 3u) { orig_c.rgb = t_orig.rgb; if (rgba) orig_c.a = t_orig.a; }
        else { orig_c.rgb += t_orig.rgb; if (rgba) orig_c.a *= t_orig.a; }
        if ((p.tex_func_bits & 16u) != 0u) orig_c.rgb *= 2.0;
    }
    bool alpha_passed = true;
    if (kSpecAlphaTest != 0u) {
        uint af = p.alpha_bits & 7u;
        if (af != 1u) {
            uint mask = (p.alpha_bits >> 16u) & 0xFFu;
            uint ref = ((p.alpha_bits >> 8u) & 0xFFu) & mask;
            uint a = uint(clamp(orig_c.a, 0.0, 1.0) * 255.0 + 0.5) & mask;
            alpha_passed = alpha_compare(af, a, ref);
        }
    }
    if (!cutout_puddle && !alpha_passed) discard;
    if (cutout_puddle && alpha_passed && orig_c.a >= 0.35) {
        // Solid rock crust around the alpha-cutout puddle hole: render 100% identically to standard ge_draw.frag.
        vec3 rock_srgb = clamp(orig_c.rgb + v_spec, vec3(0.0), vec3(1.0));
        if ((p.tex_func_bits & 0x80u) != 0u) {
            float f = clamp(v_fog, 0.0, 1.0);
            if (p.jitter_and_lod.w > 0.5) f = smoothstep(0.0, 0.08, f);
            rock_srgb = mix(p.fog_and_zmin.rgb, rock_srgb, f);
        }
        float ra = clamp(orig_c.a, 0.0, 1.0);
        o_color = vec4(rock_srgb, ra);
        uint ra4 = uint(ra * 15.0 + 0.5);
        o_mat_id = (ra4 << 4u);
        return;
    }
    float hole_mask = cutout_puddle ? (1.0 - smoothstep(0.08, 0.35, clamp(orig_c.a, 0.0, 1.0))) : 1.0;
    float shore_alpha = cutout_puddle ? 1.0 : clamp(orig_c.a, 0.0, 1.0);
    orig_c.rgb += v_spec;
    vec3 orig_lin = cutout_puddle ? vec3(0.0) : srgb_to_linear(clamp(orig_c.rgb, vec3(0.0), vec3(1.0)));

    WaterFrameBuf wf_buf = WaterFrameBuf(p.pbr_slots.xy);
    WaterFrameParams wf = wf_buf.wf;

    ivec2 ip = ivec2(gl_FragCoord.xy);
    vec2 vp_size = max(wf.vz_and_size.zw, vec2(1.0));
    vec2 uv = clamp((vec2(ip) + 0.5) / vp_size, vec2(0.001), vec2(0.999));

    uint bg_col_slot = wf.slots.x;
    uint bg_dep_slot = wf.slots.y;
    ivec2 bg_max = max(textureSize(u_textures[bg_col_slot], 0) - ivec2(1), ivec2(0));
    float z_water = linear_eye_z(wf, gl_FragCoord.z);
    vec2 ndc = uv * 2.0 - 1.0;
    vec3 P = vec3(ndc.x * z_water * wf.proj_info.x, ndc.y * z_water * wf.proj_info.y, -z_water);
    vec3 V = normalize(-P);
    vec3 U = normalize(wf.world_up_view.xyz);
    if (cutout_puddle) {
        // Only treat horizontal floor cutouts as reflective puddles; discard steep rock/cliff cutouts normally.
        vec3 n_flat = cross(dFdx(P), dFdy(P));
        float nlen = length(n_flat);
        if (nlen < 1e-6 || abs(dot(n_flat / nlen, U)) < 0.55) discard;
    }
    float t = wf.sky_zenith.w;
    float max_fresnel = wf.sky_horizon.w;
    float wave_chop = wf.deep_col.w * (shallow ? 0.20 : 0.35);

    // 1. Area-space world position & smooth glassy analytic wave normal + analytic actor wake ripples.
    vec2 wxz = P.xz;
    bool has_world = (wf.world_up_view.w > 0.5);
    if (has_world) {
        vec3 P_psp = vec3(P.x, -P.y, P.z);
        vec3 P_world = (wf.inv_view * vec4(P_psp, 1.0)).xyz;
        wxz = P_world.xz;
    }

    float dist_fade = 1.0 - smoothstep(1000.0, 4800.0, z_water);
    float close_fade = 1.0 - smoothstep(500.0, 1600.0, z_water);
    vec2 slope = eval_analytic_wave_slope(wxz, t, wave_chop * mix(0.45, 1.0, dist_fade), close_fade);

    // Expanding concentric footstep/wading ripple rings around hunter & large monsters (world-locked, 120 Hz).
    float ripple_crest = 0.0;
    uint rcnt = min(wf.ripple_count, 24u);
    if (has_world && rcnt > 0u) {
        for (uint i = 0u; i < rcnt; ++i) {
            vec4 rp = wf.ripples[i];
            float age = rp.z;
            if (age < 0.0 || age > 2.2) continue;
            vec2 dv = wxz - rp.xy;
            float d2 = dot(dv, dv);
            float max_r = rp.w + age * 155.0;
            if (d2 > max_r * max_r || d2 < 2.0) continue;
            float dist = sqrt(d2);
            float front = age * 115.0 + 14.0;
            float env = exp(-abs(dist - front) * 0.052) * exp(-age * 1.15) *
                        (1.0 - smoothstep(max_r * 0.55, max_r, dist));
            float phase = (dist - front) * 0.34;
            float dhd = cos(phase) * env * 0.14;
            slope += (dv / dist) * dhd;
            // Crisp, glassy expanding ring highlight that doesn't wash out the reflection underneath.
            float crest = pow(max(sin(phase), 0.0), 2.2) * env;
            ripple_crest += crest;
        }
        slope = clamp(slope, vec2(-0.24), vec2(0.24));
        ripple_crest = clamp(ripple_crest, 0.0, 1.0) * close_fade;
    }

    vec3 N_water = U;
    if (has_world) {
        vec3 Nw = normalize(vec3(-slope.x, 1.0, -slope.y));
        vec3 Np = normalize(transpose(mat3(wf.inv_view)) * Nw);
        N_water = normalize(vec3(Np.x, -Np.y, Np.z));
        if (dot(N_water, U) < 0.0) N_water = U;
    } else {
        vec3 T = normalize(cross(U, vec3(0.0, 0.0, 1.0)));
        vec3 B = normalize(cross(T, U));
        N_water = normalize(U - T * slope.x - B * slope.y);
    }

    // Trace against the scene snapshot, including later geometry where replay is available.
    // Compare the sampled geometry against the unperturbed water plane as well as
    // the ray depth. The lakebed and submerged parts of actors cannot reflect.
    // Third-person readable Fresnel curve: keeps reflections of the hunter, monsters, piers, trees, and mountains
    // crisp and immediately recognizable at normal third-person gameplay camera angles while preserving MHFU's water palette.
    float ndotv = clamp(dot(N_water, V), 0.0, 1.0);
    float fresnel = mix(0.55, max(max_fresnel * 1.65, 0.82), pow(1.0 - ndotv, 1.7));
    vec3 ssr_col = orig_lin;
    float ssr_conf = 0.0;
    if (z_water < 18000.0) {
        vec2 inv_proj = wf.proj_info.xy;
        vec3 ray = reflect(-V, normalize(mix(U, N_water, 0.12)));
        // 2-stage raymarch (32 steps total):
        //   steps 0..17: fine near/mid-field precision (hunter, thin monsters like Ioprey, piers, boats, lily pads)
        //   steps 18..31: exponential horizon reach (distant trees, hills, cliffs, snowy mountains up to 200m+)
        float step_len = max(4.0, z_water * 0.007);
        float prev_t = 0.0;
        float ray_t = step_len * (0.45 + 0.55 * ign(gl_FragCoord.xy));
        if (dot(ray, U) > -0.02) {
            if (dot(ray, U) < 0.01) ray = normalize(ray + U * (0.01 - dot(ray, U)));
            for (int i = 0; i < 32; ++i) {
                vec3 q = P + ray * ray_t;
                float eye_z = -q.z;
                if (eye_z <= 12.0 || eye_z > 19500.0) break;
                vec2 sample_uv = (q.xy / (eye_z * inv_proj)) * 0.5 + 0.5;
                if (any(lessThan(sample_uv, vec2(0.002))) ||
                    any(greaterThan(sample_uv, vec2(0.998)))) break;
                ivec2 sample_ip = clamp(ivec2(sample_uv * vp_size), ivec2(0), bg_max);
                float sample_d = texelFetch(u_textures[bg_dep_slot], sample_ip, 0).r;
                float scene_z = linear_eye_z(wf, sample_d);
                float delta = eye_z - scene_z;
                float tolerance = max(28.0, step_len * 2.6);
                if (sample_d < 0.99998 && delta > 0.0 && delta < tolerance) {
                    float lo = prev_t;
                    float hi = ray_t;
                    for (int j = 0; j < 4; ++j) {
                        float mid = (lo + hi) * 0.5;
                        vec3 qmid = P + ray * mid;
                        float mid_z = -qmid.z;
                        vec2 mid_uv = (qmid.xy / (mid_z * inv_proj)) * 0.5 + 0.5;
                        ivec2 mid_ip = clamp(ivec2(mid_uv * vp_size), ivec2(0), bg_max);
                        float mid_d = texelFetch(u_textures[bg_dep_slot], mid_ip, 0).r;
                        if (mid_d < 0.99998 && mid_z > linear_eye_z(wf, mid_d)) {
                            hi = mid;
                            sample_uv = mid_uv;
                            sample_ip = mid_ip;
                            scene_z = linear_eye_z(wf, mid_d);
                            delta = mid_z - scene_z;
                        } else {
                            lo = mid;
                        }
                    }
                    vec2 sample_ndc = ((vec2(sample_ip) + 0.5) / vp_size) * 2.0 - 1.0;
                    vec3 scene_p = vec3(sample_ndc * inv_proj * scene_z, -scene_z);
                    float above_water = dot(scene_p - P, U);
                    if (above_water > 0.0 && delta < tolerance) {
                        vec2 edge = smoothstep(vec2(0.0), vec2(0.05), sample_uv) *
                                    (1.0 - smoothstep(vec2(0.95), vec2(1.0), sample_uv));
                        ssr_conf = edge.x * edge.y * (1.0 - smoothstep(12000.0, 18000.0, z_water)) *
                                   smoothstep(0.0, 2.0, above_water) *
                                   (1.0 - smoothstep(tolerance * 0.78, tolerance, delta));
                        ssr_col = srgb_to_linear(texelFetch(u_textures[bg_col_slot], sample_ip, 0).rgb);
                        break;
                    }
                }
                prev_t = ray_t;
                ray_t += step_len;
                step_len *= (i < 18) ? 1.12 : 1.35;
            }
        }
    }

    float reflection_weight = clamp(fresnel * ssr_conf, 0.0, 0.82);
    vec3 water_lin = mix(orig_lin, ssr_col, reflection_weight);
    // Crisp, glassy expanding footstep ripple ring highlight (SDR-safe, never washes out hunter reflection).
    if (ripple_crest > 0.008) {
        vec3 ring_col = mix(vec3(0.32, 0.36, 0.39), wf.sky_horizon.rgb * 0.90, 0.40);
        water_lin += ring_col * (ripple_crest * 0.28);
    }
    vec3 out_srgb = (reflection_weight > 0.0 || ripple_crest > 0.008)
                        ? linear_to_srgb(clamp(water_lin, vec3(0.0), vec3(1.0)))
                        : orig_c.rgb;
    if ((p.tex_func_bits & 0x80u) != 0u) {
        float f = clamp(v_fog, 0.0, 1.0);
        if (p.jitter_and_lod.w > 0.5) {
            f = smoothstep(0.0, 0.08, f);
        }
        out_srgb = mix(p.fog_and_zmin.rgb, out_srgb, f);
    }

    if (cutout_puddle) {
        // Inside an alpha-cutout puddle hole: blend dynamic SSR reflections (hunter, monsters, pillars) + 3D ripples
        // cleanly OVER the underlying static planar reflection; discard misses so the underlying mirror stays untouched!
        if (reflection_weight <= 0.005 && ripple_crest <= 0.008) discard;
        vec3 puddle_lin = ssr_col;
        if (ripple_crest > 0.008) {
            vec3 ring_col = mix(vec3(0.32, 0.36, 0.39), wf.sky_horizon.rgb * 0.90, 0.40);
            puddle_lin = mix(puddle_lin, ring_col * 1.4, clamp(ripple_crest * 0.85, 0.0, 1.0));
        }
        vec3 puddle_srgb = linear_to_srgb(clamp(puddle_lin, vec3(0.0), vec3(1.0)));
        if ((p.tex_func_bits & 0x80u) != 0u) {
            float f = clamp(v_fog, 0.0, 1.0);
            if (p.jitter_and_lod.w > 0.5) f = smoothstep(0.0, 0.08, f);
            puddle_srgb = mix(p.fog_and_zmin.rgb, puddle_srgb, f);
        }
        float puddle_a = clamp(reflection_weight * 1.25 + ripple_crest * 0.75, 0.0, 0.88) * hole_mask;
        o_color = vec4(puddle_srgb, puddle_a);
        uint pa4 = uint(puddle_a * 15.0 + 0.5);
        o_mat_id = (2u) | (pa4 << 4u);
        return;
    }

    // Smooth shoreline feathering via MHFU's vertex alpha: 0 at dry sand -> 1 across the water body.
    // On translucent puddles (shore_alpha ~0.3-0.6), Fresnel surface reflection & ripple crests sit on top of the
    // air-water interface, so boost alpha smoothly without ever creating a hard quad edge at dry shorelines.
    float water_presence = smoothstep(0.06, 0.30, shore_alpha);
    float out_alpha = mix(shore_alpha, 1.0, clamp(reflection_weight * 0.85 + ripple_crest * 0.65, 0.0, 1.0) * water_presence);
    o_color = vec4(clamp(out_srgb, vec3(0.0), vec3(1.0)), out_alpha);

    uint a4 = uint(clamp(out_alpha, 0.0, 1.0) * 15.0 + 0.5);
    o_mat_id = (mat_id & 0xFu) | (a4 << 4u);
}
