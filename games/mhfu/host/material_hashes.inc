#pragma once

// Clean-room deterministic texture-identity -> MaterialClass registry for MHFU.
// Contains ONLY content/CLUT hashes and semantic class labels (zero copyrighted assets or texture data).
//
// How to audit / expand entries:
//   Run with MHF_VK_DUMP_MAT_KEYS=1 to log unique (data_hash, clut_hash, zone, w x h, blended) keys
//   and MHF_VK_DEBUG_MAT=1 to visualize the live R8_UINT MaterialClass buffer on screen.

#include "semantic_scene.hpp"

#include <cstdint>
#include <vector>

namespace mhf::mhfu {

inline std::vector<mhf::psp::MaterialRegistryEntry> material_registry_entries() {
    using MC = mhf::psp::MaterialClass;
    std::vector<mhf::psp::MaterialRegistryEntry> out;
    out.reserve(64);

    auto add = [&](std::uint32_t data_hash, std::uint32_t clut_hash, int zone_hint, MC mat) {
        out.push_back(mhf::psp::MaterialRegistryEntry{data_hash, clut_hash, zone_hint, mat});
    };
    auto add_data = [&](std::uint32_t data_hash, MC mat) {
        out.push_back(mhf::psp::MaterialRegistryEntry{data_hash, 0u, -1, mat});
    };

    add(0xdde85034u, 0xcc3b585fu, -1, MC::SplashAtlas);
    add(0xd5688155u, 0x3bb9f2bbu, -1, MC::SplashAtlas);

    // --- Swamp (Base, Areas 4, 5, 6): Forensically verified Layer-1 water surfaces & Layer-2 additive glints ---
    // Layer 1 (base scrolling water surface sheets):
    //   0xe16eeb4e = Swamp Area 5 (also Desert Area 6) scrolling water sheet (du=+0.002/frame)
    //   0x2d57be4a = Swamp Base / Area 6 scrolling water sheet (du=+0.002/frame)
    //   0xafee83f1 = Swamp Area 6 rain-slicked mud & puddle floor tiles
    //   0x9fb226a7 = Swamp Cave (Areas 3, 7, 9) translucent underground water pools (Z_WRITE=OFF)
    //   0x0df1f49d, 0xd6abf5df, 0xfbe666eb = Swamp poison/marsh pools
    for (std::uint32_t h : {0xe16eeb4eu, 0x2d57be4au, 0xafee83f1u, 0x9fb226a7u, 0x0df1f49du, 0xd6abf5dfu, 0xfbe666ebu}) {
        add_data(h, MC::ShallowWater);
    }
    //   0x87d43481 = Swamp Cave (Areas 3, 7) perforated rock floor whose alpha-cutout holes expose planar puddles
    //   0x9dd56203 = Swamp Cave (Area 9) perforated rock floor whose alpha-cutout holes expose planar puddles
    for (std::uint32_t h : {0x87d43481u, 0x9dd56203u}) {
        add_data(h, MC::CutoutPuddle);
    }
    // Layer 2 (8-frame additive caustic/sparkle atlas strips, du=+0.125/frame; gated on additive blend in VkBackend):
    //   0x775674cb = Swamp Area 5 water sparkle (top half) / tree reflection (bottom half, alpha-blended -> untouched!)
    //   0x008cf22e = Swamp Base water sparkle (top half) / tree reflection (bottom half, alpha-blended -> untouched!)
    for (std::uint32_t h : {0x775674cbu, 0x008cf22eu}) {
        add_data(h, MC::WaterSparkle);
    }
    for (std::uint32_t h : {0x2671fcabu, 0x43816346u, 0x466a0f3bu, 0x4c4070a9u, 0x5c7a440bu, 0xfdf4e43eu}) {
        add_data(h, MC::HazeIgnore);
    }
    for (std::uint32_t h : {0x0a4e0bfdu, 0x0d303d45u, 0x371bc1a7u, 0x707085fbu, 0x87e504e8u,
                            0x8d2eea69u, 0x9db606d6u, 0xa3ed4e5du, 0xa854cb85u, 0xbfe8e04eu}) {
        add_data(h, MC::Foliage);
    }

    // --- Pokke Village & Pokke Farm / Snowy Mountains: Forensically verified Layer-1 water & Layer-2 glints ---
    //   0xf992e123 = Pokke Village mossy green mountain stream (du=+0.001/frame, Z_WRITE=OFF)
    //   0x52f75431, 0xb3842d69 = Pokke Farm lake & Snowy Mtns scrolling base water sheets (du=+0.002/frame)
    //   0xbf303216, 0xe5fe3d0d = Pokke Farm river water variants
    for (std::uint32_t h : {0xf992e123u, 0x52f75431u, 0xb3842d69u, 0xbf303216u, 0xe5fe3d0du}) {
        add_data(h, MC::Water);
    }
    //   0xc2e72331 = Pokke Farm 8-frame additive water sparkle strip (du=+0.125/frame, Z_WRITE=OFF)
    add_data(0xc2e72331u, MC::WaterSparkle);
    for (std::uint32_t h : {0x6070f000u, 0x36b1775fu, 0x399f060au, 0x5cc69a1du}) {
        add_data(h, MC::Wood);
    }
    for (std::uint32_t h : {0x93f9e6cdu, 0xbd456c28u}) {
        add_data(h, MC::Metal);
    }
    for (std::uint32_t h : {0x058ca67eu, 0x2868fc9cu, 0x4a330b66u, 0x4e45c94cu}) {
        add_data(h, MC::Foliage);
    }

    // --- Jungle: Forensically verified Layer-1 river/ocean/cave water, Layer-2 additive glints, and sky/flare cards ---
    //   0x6a7dff00, 0x0d40e304 = Jungle Base Camp beach/ocean & stream water
    //   0x49a84412, 0xbadc0995 = Jungle Area 3/4/10 ocean Layer-1 water (du=+0.002/frame)
    //   0x8cbe626d, 0x864671b1 = Jungle Area 7 river/marsh water
    //   0xfa558fc5             = Jungle Area 9 underground cave pool (du/dv=+0.001/frame)
    for (std::uint32_t h : {0x6a7dff00u, 0x0d40e304u, 0x49a84412u, 0xbadc0995u,
                            0x8cbe626du, 0x864671b1u, 0xfa558fc5u,
                            0x919270f5u, 0x1ad0f53au, 0x35909976u, 0x8294b799u, 0xadbbaa85u}) {
        add_data(h, MC::Water);
    }
    //   0xc9c4fa61 = Jungle Area 3/4 4-frame additive caustic/glint strip (du=+0.250/frame)
    add_data(0xc9c4fa61u, MC::WaterSparkle);
    for (std::uint32_t h : {0x5563b924u, 0x84c1a2dau, 0x8c163b48u, 0xc6e5952cu, 0x6c090717u}) {
        add_data(h, MC::HazeIgnore);
    }

    // --- Forest & Hills: Stream/river water and sky/flare cards ---
    //   0x09de2e1b, 0x8199e3ba = Base Camp / river scrolling water (dv=+0.004/frame)
    //   0xa1d5b88d, 0xa9884b11 = Area 10 shallow stream & animated pool water
    //   0x89f831a6, 0x29145457 = Area 11 waterfall pool & river water
    for (std::uint32_t h : {0x09de2e1bu, 0x8199e3bau, 0xa1d5b88du, 0xa9884b11u, 0x89f831a6u, 0x29145457u}) {
        add_data(h, MC::Water);
    }
    for (std::uint32_t h : {0x31d56309u, 0x7f4bc010u, 0xbee06581u, 0xc6d1373eu, 0x712622deu, 0xc372fb49u}) {
        add_data(h, MC::HazeIgnore);
    }

    // --- Desert: Oasis & underground cave water pools + additive sparkle strips ---
    //   (Area 1 shares 0x6a7dff00; Area 6 shares 0xe16eeb4e)
    //   0x9ea469b5 = Desert Area 7 underground oasis pool (dv=+0.001/frame)
    add_data(0x9ea469b5u, MC::Water);
    //   0x1be7fcc9, 0xe1c1dcaf, 0x3378a0e6 = Desert 8-frame additive water sparkle strips (du=+0.125/frame)
    for (std::uint32_t h : {0x1be7fcc9u, 0xe1c1dcafu, 0x3378a0e6u}) {
        add_data(h, MC::WaterSparkle);
    }

    // Known haze / mist / weather-sheet textures from MHFU shadow_textures.hpp (clean-room hashes).
    // Tagged as non-surface HazeIgnore / WeatherSheet so they never receive diffuse SSAO darkening
    // and never overwrite the underlying physical surface MaterialClass in the R8_UINT MRT.
    for (std::uint64_t h : kShadowHazeTextures) {
        const auto data_h = static_cast<std::uint32_t>(h & 0xFFFFFFFFull);
        const auto clut_h = static_cast<std::uint32_t>(h >> 32u);
        add(data_h, clut_h, -1, MC::HazeIgnore);
    }
    for (std::uint64_t h : kShadowAirTextures) {
        const auto data_h = static_cast<std::uint32_t>(h & 0xFFFFFFFFull);
        const auto clut_h = static_cast<std::uint32_t>(h >> 32u);
        // Snowy Mountains blizzard camera sheet (0x3510dbf03d994cf9) -> WeatherSheet; others -> HazeIgnore.
        const MC cls = (h == 0x3510dbf03d994cf9ull) ? MC::WeatherSheet : MC::HazeIgnore;
        add(data_h, clut_h, -1, cls);
    }

    return out;
}

} // namespace mhf::mhfu
