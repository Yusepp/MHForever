#pragma once

// Backend-agnostic types and GE register/pixel helpers shared between the OpenGL and Vulkan hardware renderers.

#include "backend.hpp"
#include "../semantic_scene.hpp"

#include "psprecomp/guest_memory.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace mhf::psp {

// An image drawn over the game (system dialogs, in-game menu, quest overlays), handed from the emulation thread to the
// drawing thread. Premultiplied RGBA covering the letterboxed game area; `generation` changes whenever the image does.
struct OverlayImage {
    std::mutex mutex;
    std::vector<std::uint32_t> pixels;
    int width = 0, height = 0;
    bool visible = false;
    // The image is the in-game menu: the game runs on behind it, blurred, and the backend draws it at least as often as the
    // display refreshes, also when the game shows nothing new.
    bool continuous = false;
    // The shape it covers: 0 the PSP's 480:272, centred in the window; wider, the stretched widescreen picture (the
    // hunting horn's songs, which sit by the quest HUD).
    float aspect = 0.0f;
    std::uint64_t generation = 0;
};

// What the player can change while the game runs (read by the drawing thread).
struct LiveSettings {
    std::atomic<bool> smooth_textures{false};
    // Requests for the drawing thread, taken at the start of the next picture (0 = none): the internal resolution
    // multiplier (1..10) and the output rate in fps (a multiple of 30 up to 180).
    std::atomic<int> scale{0};
    std::atomic<int> fps{0};
    // Texture pack (frontend/texture_pack.hpp): drawn or not while the game runs; and, set before the backend is made,
    // the pack's folder (empty: none, or MHF_TEXTURE_PACK) and the release it is drawn for ("ULES01213").
    std::atomic<bool> texture_pack{true};
    std::string texture_pack_folder, texture_pack_game;
    // The textures MHForever ships (a pack in PPSSPP's format, fonts/hd-text: the HD text by Kaneda), drawn with or
    // without the player's pack; the player's pack wins for a texture both have. Empty: none.
    std::string builtin_textures_folder;
    // Its filtering, changed in the menu while the game runs: mip levels made for images that ship none (softer far
    // away, no shimmer; off draws them as PPSSPP does), and the anisotropic filtering of its images, 1 (off) to 16.
    std::atomic<bool> texture_pack_mipmaps{false};
    std::atomic<int> texture_pack_anisotropy{16};
    // Projected shadows (enhancement): on or off while the game runs (MHF_SHADOW=1 starts with them on). Skinned draws (the
    // hunter, the monsters) always cast. shadow_whole_map: all the solid scenery casts too, and surfaces turned away from the sun
    // are shaded; off, only the hunter and monsters cast. shadow_quality: the size of the map of shadows, 0 low (2048), 1 medium
    // (4096), 2 high (8192); low unless chosen. shadow_strength: how dark, 0..1; at 0.5 each zone is as it was tuned, below it lighter down to no
    // shade at 0, above it darker up to half as dark again at 1.
    std::atomic<bool> shadow{false};
    std::atomic<bool> shadow_whole_map{true};
    std::atomic<int> shadow_quality{0};
    // What the game knows about its own textures (Frontend::set_shadow_textures), set before the backend is made: air (mist,
    // haze), solid (rock that looks like mist to the guesses), and the insides of caves, where only objects cast.
    std::vector<std::uint64_t> shadow_air_textures, shadow_solid_textures, shadow_cave_textures, shadow_haze_textures;
    // Where the game is: the zones of its maps ("areas/hunting/desert/area6"), the game's textures that only a zone uses (a hash
    // of the game's own texture, as packs key them, and an index into the names), and the zones that are caves.
    std::vector<std::string> shadow_zone_names, shadow_cave_zones;
    std::vector<std::pair<std::string, std::array<float, 3>>> shadow_zone_suns; // some zones' sun: height in degrees, darkness (< 0: 0.4), turn in degrees
    std::vector<std::pair<std::uint32_t, int>> shadow_zone_textures;
    // Optional per-zone lighting biome (biome preset index, is_cave) aligned with `shadow_zone_names`.
    std::vector<std::pair<std::uint8_t, bool>> shadow_zone_biomes;
    // Optional per-zone ground-mist profile aligned with `shadow_zone_names`.
    std::vector<GroundMistProfile> shadow_zone_mist_profiles;
    // Clean-room deterministic texture-identity -> MaterialClass registry (supplied by the game host).
    std::vector<MaterialRegistryEntry> material_registry;
    std::atomic<float> shadow_strength{0.5f};
    // What casts, set by the backend as it goes in and out of caves: objects up to shadow_max_object long (world units), and
    // inside the scenery slender pieces up to shadow_max_piece (0: none).
    std::atomic<float> shadow_max_object{39450.0f};
    std::atomic<float> shadow_max_piece{39450.0f};
    // HUD size in percent (100: as the game draws it): in quests, the HUD shrinks into the corners it sits in.
    std::atomic<int> hud_size{100};
    // Widescreen (enhancement, off by default): the game's 3D is drawn at the window's shape and the picture stretched
    // to it, 2D kept in shape. wide_aspect is that shape (width over height), worked out by the frontend at every poll:
    // 0 while off, with the software renderer or in a window no wider than 480:272.
    std::atomic<bool> widescreen{false};
    std::atomic<float> wide_aspect{0.0f};
    // Written by the drawing thread at every flip: that flip drew the quest HUD (its clock). The hunting horn's song
    // panel is only shown over it.
    std::atomic<bool> quest_hud{false};
    // Next-gen Vulkan lighting & atmosphere (multi-bounce GTAO, contact & projected sun shadows,
    // zone-aware volumetric fog/sun shafts, selective emissive bloom, SSGI bounce/flash, specular sheen,
    // soft 3D particles, pre-HUD FXAA, moody vignette, and 4x killcam bokeh).
    enum VulkanEffectBits : std::uint32_t {
        kVkEffectAo             = 1u << 0,
        kVkEffectContactShadows = 1u << 1,
        kVkEffectVolumetricFog  = 1u << 2,
        kVkEffectBloom          = 1u << 3,
        kVkEffectSsgi           = 1u << 4,
        kVkEffectSpecular       = 1u << 5,
        kVkEffectSoftParticles  = 1u << 6,
        kVkEffectFxaa           = 1u << 7,
        kVkEffectVignette       = 1u << 8,
        kVkEffectKillcamDof     = 1u << 9,
        kVkEffectSunShadows     = 1u << 10,
        kVkEffectDynamicLights  = 1u << 11,
        kVkEffectPbrMaterials   = 1u << 12,
        kVkEffectFroxelFog      = 1u << 13,
        kVkEffectRayTracing     = 1u << 14,
        kVkEffectWaterReflections = 1u << 15,
        kVkEffectComputeMask    = kVkEffectAo | kVkEffectContactShadows | kVkEffectVolumetricFog |
                                  kVkEffectBloom | kVkEffectSsgi | kVkEffectSpecular |
                                  kVkEffectFxaa | kVkEffectVignette | kVkEffectSunShadows |
                                  kVkEffectDynamicLights | kVkEffectFroxelFog | kVkEffectRayTracing |
                                  kVkEffectWaterReflections,
        kVkEffectDefault        = 0xBFFFu, // all ON except opt-in Ultra Ray Tracing
        kVkEffectAll            = 0xFFFFu,
    };
    std::atomic<bool> enhanced_lighting{false};
    std::atomic<std::uint32_t> vulkan_effects_mask{kVkEffectDefault};
    // Vulkan 3D anti-aliasing mode: 0 = Off, 1 = FXAA, 2 = MSAA 4x, 3 = TAA.
    enum AaMode : int {
        kAaOff = 0,
        kAaFxaa = 1,
        kAaMsaa4x = 2,
        kAaTaa = 3,
    };
    std::atomic<int> aa_mode{kAaFxaa};
    std::atomic<bool> msaa_4x_supported{false};
    std::atomic<bool> ray_query_supported{false};
    // Optional game-supplied world-space dynamic lights (updated from the host poll, read by Vulkan).
    mutable std::mutex semantic_mutex;
    SemanticSceneFrame semantic_frame{};
    SceneCamera scene_camera{};
};

namespace gpu_common {

// GE registers consumed by hardware backends (see platforms/psp/gpu/ge.cpp).
enum Reg : std::uint32_t {
    kVertexType = 0x12, kTextureMapEnable = 0x1E, kFogEnable = 0x1F, kAlphaBlendEnable = 0x21,
    kAlphaTestEnable = 0x22, kZTestEnable = 0x23, kStencilTestEnable = 0x24, kShadeMode = 0x50,
    kFramebufPtr = 0x9C, kFramebufWidth = 0x9D, kTexAddr0 = 0xA0, kTexBufWidth0 = 0xA8, kClutAddr = 0xB0,
    kClutAddrUpper = 0xB1, kTexSize0 = 0xB8, kTexMode = 0xC2, kTexFormat = 0xC3, kClutFormat = 0xC5,
    kTexFilter = 0xC6, kTexWrap = 0xC7, kTexFunc = 0xC9, kTexEnvColor = 0xCA, kFogColor = 0xCF,
    kFramebufPixFormat = 0xD2, kClearMode = 0xD3, kScissor1 = 0xD4, kScissor2 = 0xD5, kMinZ = 0xD6, kMaxZ = 0xD7,
    kAlphaTest = 0xDB, kZTest = 0xDE, kBlendMode = 0xDF, kBlendFixedA = 0xE0, kBlendFixedB = 0xE1,
    kZWriteDisable = 0xE7, kMaskRgb = 0xE8, kMaskAlpha = 0xE9, kViewportXScale = 0x42, kViewportYScale = 0x43,
    kViewportZScale = 0x44, kViewportXCenter = 0x45, kViewportYCenter = 0x46, kViewportZCenter = 0x47, kOffsetX = 0x4C,
    kOffsetY = 0x4D,
};

constexpr std::array<std::uint32_t, 36> kStateRegs = {
    kVertexType, kTextureMapEnable, kFogEnable, kAlphaBlendEnable, kAlphaTestEnable, kZTestEnable,
    kStencilTestEnable, kShadeMode, kFramebufPtr, kFramebufWidth, kTexAddr0, kTexBufWidth0, kClutAddr,
    kClutAddrUpper, kTexSize0, kTexMode, kTexFormat, kClutFormat, kTexFilter, kTexWrap, kTexFunc, kTexEnvColor,
    kFogColor, kFramebufPixFormat, kClearMode, kScissor1, kScissor2, kMinZ, kMaxZ, kAlphaTest, kZTest, kBlendMode,
    kBlendFixedA, kBlendFixedB, kZWriteDisable, kMaskRgb,
};

using DrawKey = std::array<std::uint32_t, kStateRegs.size() + 2>;

constexpr std::array<int, 256> kKeyIndex = [] {
    std::array<int, 256> index{};
    index.fill(-1);
    for (std::size_t i = 0; i < kStateRegs.size(); ++i) index[kStateRegs[i] & 0xFFu] = static_cast<int>(i);
    return index;
}();

[[nodiscard]] inline std::uint32_t key_reg(const DrawKey &key, std::uint32_t reg) noexcept {
    if (reg == kMaskAlpha) return key[kStateRegs.size() + 1];
    const int i = kKeyIndex[reg & 0xFFu];
    return i >= 0 ? key[static_cast<std::size_t>(i)] : 0u;
}

// Screen-space vertex layout shared by both OpenGL and Vulkan draw pipelines (40 bytes).
struct GpuVertex {
    float x, y, z, inv_w;
    float u, v;
    std::uint8_t color[4];
    std::uint8_t spec[4];
    float fog;
    std::uint32_t normal_oct; // packed 2x16-bit snorm octahedral view-space normal (0 = none)
};
static_assert(sizeof(GpuVertex) == 40);

// Encodes a unit view-space normal into 2x16-bit snorm octahedral coordinates (0u reserved for "absent").
[[nodiscard]] inline std::uint32_t pack_oct_normal(float nx, float ny, float nz) noexcept {
    const float l1 = (nx < 0.0f ? -nx : nx) + (ny < 0.0f ? -ny : ny) + (nz < 0.0f ? -nz : nz);
    if (l1 < 1e-5f) return 0u;
    float ox = nx / l1;
    float oy = ny / l1;
    if (nz < 0.0f) {
        const float tx = (1.0f - (oy < 0.0f ? -oy : oy)) * (ox >= 0.0f ? 1.0f : -1.0f);
        const float ty = (1.0f - (ox < 0.0f ? -ox : ox)) * (oy >= 0.0f ? 1.0f : -1.0f);
        ox = tx;
        oy = ty;
    }
    auto to_s16 = [](float v) -> std::int16_t {
        if (v < -1.0f) v = -1.0f;
        if (v > 1.0f) v = 1.0f;
        return static_cast<std::int16_t>(v * 32767.0f + (v >= 0.0f ? 0.5f : -0.5f));
    };
    std::uint16_t ux = static_cast<std::uint16_t>(to_s16(ox));
    std::uint16_t uy = static_cast<std::uint16_t>(to_s16(oy));
    std::uint32_t packed = static_cast<std::uint32_t>(ux) | (static_cast<std::uint32_t>(uy) << 16u);
    return packed == 0u ? 1u : packed;
}

[[nodiscard]] inline std::uint32_t expand_bits(std::uint32_t v, int bits) noexcept {
    return bits == 5 ? (v << 3u) | (v >> 2u) : bits == 6 ? (v << 2u) | (v >> 4u) : (v << 4u) | v;
}

// PSP framebuffer pixel (format 0..3) to ABGR8888.
[[nodiscard]] inline std::uint32_t framebuffer_pixel(const std::uint8_t *p, std::uint32_t format) noexcept {
    if (format == 3u) {
        std::uint32_t c;
        std::memcpy(&c, p, 4);
        return c;
    }
    std::uint16_t v;
    std::memcpy(&v, p, 2);
    std::uint32_t r, g, b, a;
    if (format == 0u) {
        r = expand_bits(v & 31u, 5);
        g = expand_bits((v >> 5u) & 63u, 6);
        b = expand_bits((v >> 11u) & 31u, 5);
        a = 255u;
    } else if (format == 1u) {
        r = expand_bits(v & 31u, 5);
        g = expand_bits((v >> 5u) & 31u, 5);
        b = expand_bits((v >> 10u) & 31u, 5);
        a = (v & 0x8000u) ? 255u : 0u;
    } else {
        r = expand_bits(v & 15u, 4);
        g = expand_bits((v >> 4u) & 15u, 4);
        b = expand_bits((v >> 8u) & 15u, 4);
        a = expand_bits((v >> 12u) & 15u, 4);
    }
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

inline void pack_pixel(const std::uint8_t *p, std::uint32_t format, std::uint8_t *out) noexcept {
    if (format == 3u) {
        std::memcpy(out, p, 4);
        return;
    }
    std::uint16_t v = 0;
    if (format == 0u) v = static_cast<std::uint16_t>((p[0] >> 3) | ((p[1] >> 2) << 5) | ((p[2] >> 3) << 11));
    else if (format == 1u) v = static_cast<std::uint16_t>((p[0] >> 3) | ((p[1] >> 3) << 5) | ((p[2] >> 3) << 10) | ((p[3] >= 128 ? 1 : 0) << 15));
    else v = static_cast<std::uint16_t>((p[0] >> 4) | ((p[1] >> 4) << 4) | ((p[2] >> 4) << 8) | ((p[3] >> 4) << 12));
    std::memcpy(out, &v, 2);
}

inline void unpack_rgb(std::uint32_t c, float out[3]) noexcept {
    out[0] = static_cast<float>(c & 0xFFu) / 255.0f;
    out[1] = static_cast<float>((c >> 8u) & 0xFFu) / 255.0f;
    out[2] = static_cast<float>((c >> 16u) & 0xFFu) / 255.0f;
}

} // namespace gpu_common
} // namespace mhf::psp
