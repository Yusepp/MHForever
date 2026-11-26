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
