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

    // --- Swamp (Areas 4, 5, 6): Marsh water, puddles, haze cards, and foliage ---
    for (std::uint32_t h : {0x0df1f49du, 0xd6abf5dfu, 0xfbe666ebu, 0x7f492b54u, 0xcd40fbf4u,
                            0x6253df69u, 0x4c8bfc2bu, 0x73d6a710u, 0x775674cbu, 0xa59cf8cdu}) {
        add_data(h, MC::ShallowWater);
    }
    for (std::uint32_t h : {0x2671fcabu, 0x43816346u, 0x466a0f3bu, 0x4c4070a9u, 0x5c7a440bu, 0xfdf4e43eu}) {
        add_data(h, MC::HazeIgnore);
    }
    for (std::uint32_t h : {0x0a4e0bfdu, 0x0d303d45u, 0x371bc1a7u, 0x707085fbu, 0x87e504e8u,
                            0x8d2eea69u, 0x9db606d6u, 0xa3ed4e5du, 0xa854cb85u, 0xbfe8e04eu}) {
        add_data(h, MC::Foliage);
    }

    // --- Pokke Farm: River water, wood, metal, and foliage ---
    for (std::uint32_t h : {0xbf303216u, 0xe5fe3d0du}) {
        add_data(h, MC::Water);
    }
    for (std::uint32_t h : {0x6070f000u, 0x36b1775fu, 0x399f060au, 0x5cc69a1du}) {
        add_data(h, MC::Wood);
    }
    for (std::uint32_t h : {0x93f9e6cdu, 0xbd456c28u}) {
        add_data(h, MC::Metal);
    }
    for (std::uint32_t h : {0x058ca67eu, 0x2868fc9cu, 0x4a330b66u, 0x4e45c94cu}) {
        add_data(h, MC::Foliage);
    }

    // --- Jungle: River/ocean water and sky/flare cards ---
    for (std::uint32_t h : {0x49a84412u, 0x919270f5u, 0x1ad0f53au, 0x35909976u, 0x8294b799u, 0xadbbaa85u}) {
        add_data(h, MC::Water);
    }
    for (std::uint32_t h : {0x5563b924u, 0x84c1a2dau, 0x8c163b48u, 0xc6e5952cu, 0x6c090717u}) {
        add_data(h, MC::HazeIgnore);
    }

    // --- Forest & Hills: Stream/river water and sky/flare cards ---
    for (std::uint32_t h : {0x09de2e1bu, 0x89f831a6u, 0x29145457u}) {
        add_data(h, MC::Water);
    }
    for (std::uint32_t h : {0x31d56309u, 0x7f4bc010u, 0xbee06581u, 0xc6d1373eu, 0x712622deu, 0xc372fb49u}) {
        add_data(h, MC::HazeIgnore);
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
