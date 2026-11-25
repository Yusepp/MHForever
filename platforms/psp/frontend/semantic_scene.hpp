#pragma once

// Game-agnostic semantic scene data pushed by a game host (e.g., elemental combat hit flashes,
// boss auras, campfires) and consumed by the Vulkan renderer alongside universal GE VFX heuristics.

#include <array>
#include <cstddef>
#include <cstdint>

namespace mhf::psp {

struct SemanticPointLight {
    float world_pos[3]{0.0f, 0.0f, 0.0f};
    float radius{120.0f};
    float color[3]{1.0f, 0.7f, 0.3f};
    float intensity{1.0f};
};

struct SceneCamera {
    std::array<float, 16> world_to_eye{};
    bool valid = false;
    std::array<float, 3> hunter_feet{};
    bool has_hunter_feet = false;
    std::uint16_t ground_area = 0;
    float ground_mist_weather = -1.0f; // 0..1 when supplied by the game host
};

struct SemanticSceneFrame {
    static constexpr std::size_t kMaxLights = 32;
    std::uint32_t light_count{0};
    std::array<SemanticPointLight, kMaxLights> lights{};
};

} // namespace mhf::psp
