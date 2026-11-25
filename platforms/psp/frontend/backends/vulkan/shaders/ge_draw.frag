#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

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
    uvec4 pbr_slots;     // x = _n normal, y = _r rough/metal, z = _e emissive, w = 0
    vec4 sun_dir_view;   // xyz = view-space sun dir (+Y up), w = outdoor (1 or 0)
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer DrawParamsBuf {
    DrawParams items[];
};

layout(push_constant) uniform DrawPush {
    DrawParamsBuf params;
    uint draw_idx;
} pc;

// Specialization constants eliminate divergent branches and keep Early-Z/Hi-Z enabled for opaque draws.
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
        if (kSpecTextured != 0u && ((bits & 0x400u) != 0u)) {
            // Quest-clear 4x killcam subtle peripheral cinematic bokeh (center subject stays razor-sharp, 0ms in combat!).
            vec2 dc = v_uv - vec2(0.5, 0.48);
            float coc = smoothstep(0.24, 0.62, length(dc)) * 0.0042;
            vec4 t = texture(sampler2D(u_textures[p.tex_idx],
                                       u_samplers[p.sampler_idx]), v_uv);
            if (coc > 0.0003) {
                vec4 b = t * 0.28;
                const vec2 kRing[8] = vec2[8](
                    vec2( 1.0,  0.0), vec2(-1.0,  0.0), vec2( 0.0,  1.0), vec2( 0.0, -1.0),
                    vec2( 0.707,  0.707), vec2(-0.707,  0.707), vec2( 0.707, -0.707), vec2(-0.707, -0.707)
                );
                for (int i = 0; i < 8; ++i) {
                    vec2 suv = clamp(v_uv + kRing[i] * vec2(coc, coc * 1.77), vec2(0.001), vec2(0.999));
                    b += texture(sampler2D(u_textures[p.tex_idx],
                                           u_samplers[p.sampler_idx]), suv) * 0.09;
                }
                t = b;
            }
            uint func = bits & 7u;
            bool rgba = (bits & 8u) != 0u;
            if (func == 0u) { c.rgb *= t.rgb; if (rgba) c.a *= t.a; }
            else if (func == 1u) { c.rgb = rgba ? mix(c.rgb, t.rgb, t.a) : t.rgb; }
            else if (func == 2u) { c.rgb = mix(c.rgb, p.env_and_flags.rgb, t.rgb); if (rgba) c.a *= t.a; }
            else if (func == 3u) { c.rgb = t.rgb; if (rgba) c.a = t.a; }
            else { c.rgb += t.rgb; if (rgba) c.a *= t.a; }
            if ((bits & 16u) != 0u) c.rgb *= 2.0;
        } else if (kSpecTextured != 0u && ((bits & 0x40u) != 0u)) {
            vec4 t = texture(sampler2D(u_textures[p.tex_idx],
                                       u_samplers[p.sampler_idx]), v_uv, p.jitter_and_lod.z);
            uint func = bits & 7u;
            bool rgba = (bits & 8u) != 0u;
            if (func == 0u) { c.rgb *= t.rgb; if (rgba) c.a *= t.a; }
            else if (func == 1u) { c.rgb = rgba ? mix(c.rgb, t.rgb, t.a) : t.rgb; }
            else if (func == 2u) { c.rgb = mix(c.rgb, p.env_and_flags.rgb, t.rgb); if (rgba) c.a *= t.a; }
            else if (func == 3u) { c.rgb = t.rgb; if (rgba) c.a = t.a; }
            else { c.rgb += t.rgb; if (rgba) c.a *= t.a; }
            if ((bits & 16u) != 0u) c.rgb *= 2.0;
        }
        if (kSpecHdr != 0u) {
            c.rgb = max(c.rgb + v_spec, vec3(0.0));
        } else {
            c.rgb = min(c.rgb + v_spec, vec3(1.0));
        }
        // Optional Moddable PBR Material Packs (_n normal, _r roughness/metallic, _e emissive):
        // Uniform-branched to 0.00ms cost when absent on standard PPSSPP/original PSP textures!
        if (kSpecTextured != 0u && ((p.pbr_slots.x | p.pbr_slots.y | p.pbr_slots.z) != 0u)) {
            vec3 N = (v_has_normal > 0.5) ? normalize(v_normal) : vec3(0.0, 0.0, 1.0);
            vec3 L = normalize(p.sun_dir_view.xyz);
            vec3 V = vec3(0.0, 0.0, 1.0);
            float base_ndotl = clamp(dot(N, L), 0.0, 1.0);

            if (p.pbr_slots.x != 0u) {
                vec3 tn = texture(sampler2D(u_textures[p.pbr_slots.x],
                                            u_samplers[p.sampler_idx]), v_uv, p.jitter_and_lod.z).xyz * 2.0 - 1.0;
                // Screen-space cotangent frame (requires zero PSP vertex tangents).
                vec3 dp1 = dFdx(vec3(gl_FragCoord.xy, gl_FragCoord.z * 65535.0));
                vec3 dp2 = dFdy(vec3(gl_FragCoord.xy, gl_FragCoord.z * 65535.0));
                vec2 duv1 = dFdx(v_uv);
                vec2 duv2 = dFdy(v_uv);
                vec3 dp2perp = cross(dp2, N);
                vec3 dp1perp = cross(N, dp1);
                vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
                vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
                float invmax = inversesqrt(max(dot(T, T), dot(B, B)) + 1e-8);
                N = normalize(T * (tn.x * invmax) + B * (tn.y * invmax) + N * max(tn.z, 0.05));
                float pbr_ndotl = clamp(dot(N, L), 0.0, 1.0);
                // Modulate diffuse relief cleanly relative to the base vertex normal.
                float relief = clamp((pbr_ndotl + 0.35) / (base_ndotl + 0.35), 0.65, 1.35);
                c.rgb *= mix(1.0, relief, 0.72);
            }

            if (p.pbr_slots.y != 0u) {
                vec4 rm = texture(sampler2D(u_textures[p.pbr_slots.y],
                                            u_samplers[p.sampler_idx]), v_uv, p.jitter_and_lod.z);
                float rough = clamp(rm.g, 0.08, 1.0);
                // Support both glTF ORM (G=rough, B=metal) and grayscale roughness maps (R==G==B -> metal=0).
                float is_gray = step(abs(rm.r - rm.g) + abs(rm.g - rm.b), 0.015);
                float metal = mix(clamp(rm.b, 0.0, 1.0), 0.0, is_gray);
                vec3 H = normalize(L + V);
                float ndotl = clamp(dot(N, L), 0.0, 1.0);
                float ndotv = max(dot(N, V), 0.02);
                float ndoth = clamp(dot(N, H), 0.0, 1.0);
                float vdoth = clamp(dot(V, H), 0.0, 1.0);
                float a = rough * rough;
                float a2 = a * a;
                float denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
                float D = a2 / (3.14159265 * denom * denom + 1e-5);
                vec3 F0 = mix(vec3(0.04), clamp(c.rgb, vec3(0.02), vec3(1.0)), metal);
                vec3 F = F0 + (vec3(1.0) - F0) * pow(1.0 - vdoth, 5.0);
                float k = (rough + 1.0) * (rough + 1.0) * 0.125;
                float G = (ndotl / (ndotl * (1.0 - k) + k)) * (ndotv / (ndotv * (1.0 - k) + k));
                vec3 spec_brdf = (D * G * F) / max(4.0 * ndotl * ndotv, 0.08);
                float env_scale = mix(0.32, 0.75, p.sun_dir_view.w);
                c.rgb = c.rgb * (1.0 - metal * 0.45) + min(spec_brdf * ndotl * env_scale, vec3(1.35));
            }

            if (p.pbr_slots.z != 0u) {
                vec3 em = texture(sampler2D(u_textures[p.pbr_slots.z],
                                            u_samplers[p.sampler_idx]), v_uv, p.jitter_and_lod.z).rgb;
                c.rgb += em * (kSpecHdr != 0u ? 1.45 : 0.85);
            }
        }
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
        if (kSpecZRange != 0u && ((bits & 0x100u) != 0u)) {
            float z = gl_FragCoord.z * 65535.0;
            if (z < p.fog_and_zmin.w || z > p.zmax_pad.x + 0.5) discard;
        }
        // Soft 3D Particles: smoothly fade depth-tested non-depth-writing blended billboards
        // (smoke, snow puffs, dust, fireballs, flames) near solid geometry intersections!
        uint soft_depth_slot = bits >> 12u;
        if (soft_depth_slot != 0u && p.zmax_pad.y > 0.0) {
            ivec2 dsize = textureSize(u_textures[soft_depth_slot], 0);
            ivec2 dip = clamp(ivec2(gl_FragCoord.xy), ivec2(0), dsize - ivec2(1));
            float scene_d = texelFetch(u_textures[soft_depth_slot], dip, 0).r;
            if (scene_d < 0.99992) {
                // Perspective depth delta scales with (1 - d)^2; scale fade smoothly across view distance.
                float inv_z = max(1.0 - gl_FragCoord.z, 0.002);
                float fade_range = p.zmax_pad.y * inv_z;
                float diff = scene_d - gl_FragCoord.z;
                float soft_fade = clamp(diff / max(fade_range, 1e-6), 0.0, 1.0);
                soft_fade = soft_fade * soft_fade * (3.0 - 2.0 * soft_fade);
                if ((bits & 0x200u) != 0u) {
                    c.rgb *= soft_fade; // additive (ONE+ONE) fire/sparks
                }
                c.a *= soft_fade;
            }
        }
    }
    if (kSpecHdr != 0u) {
        o_color = vec4(clamp(c.rgb, vec3(0.0), vec3(6.0)), clamp(c.a, 0.0, 1.0));
    } else {
        o_color = clamp(c, 0.0, 1.0);
    }
}
