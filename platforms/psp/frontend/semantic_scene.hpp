#pragma once

// Game-agnostic semantic scene data pushed by a game host (e.g., elemental combat hit flashes,
// boss auras, campfires, area-space camera/wake actors, ground mist profiles, terrain heightfields)
// and consumed by the Vulkan renderer alongside universal GE VFX heuristics.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace mhf::psp {

struct SemanticPointLight {
    float world_pos[3]{0.0f, 0.0f, 0.0f};
    float radius{120.0f};
    float color[3]{1.0f, 0.7f, 0.3f};
    float intensity{1.0f};
};

// Deterministic visible-surface semantic classes written to the dedicated R8_UINT MRT attachment.
// 0 = Unknown (cleared state / conservative fallback; never color-guessed at runtime).
enum class MaterialClass : std::uint8_t {
    Unknown      = 0,
    Water        = 1,  // Deep/flowing rivers, lakes, ocean beaches (translucent surface)
    ShallowWater = 2,  // Swamp puddles, wet shore mud (opaque/thin water film)
    Snow         = 3,  // Snowpack (restrained AO, sunlit micro-sparkle in Step 4)
    Ice          = 4,  // Ice caves, glaciers
    Metal        = 5,  // Metallic armor, weapons, ironwork
    Wood         = 6,  // Timber, roofs, fences, bridges
    Foliage      = 7,  // Leaves, grass, bushes (attenuated AO, backlit SSS in Step 4)
    Fire         = 8,  // Campfires, torches, fireballs (no diffuse AO)
    Lava         = 9,  // Molten magma (no diffuse AO)
    WeatherSheet = 10, // 2D camera rain/snow/sandstorm sheets (non-surface)
    HazeIgnore   = 11, // Sky/mist/haze cards (non-surface, no diffuse AO)
    WaterSparkle = 12, // Secondary additive water caustic/glint flipbook
    SplashAtlas  = 13, // Shared effect atlas with an authored water-impact tile
    CutoutPuddle = 14, // Alpha-cutout floor whose discarded holes expose a planar reflection puddle
};

// Clean-room texture-identity -> MaterialClass registry entry supplied by the game host.
// Uses composite (data_hash, clut_hash) + optional zone_hint (-1 = any zone) to prevent CLUT/context collisions.
struct MaterialRegistryEntry {
    std::uint32_t data_hash = 0u;
    std::uint32_t clut_hash = 0u; // 0 = match any CLUT (non-indexed or palette-invariant)
    int zone_hint = -1;           // -1 = any zone; >= 0 = specific index into shadow_zone_names
    MaterialClass mat = MaterialClass::Unknown;
};

struct SceneWakeActor {
    float x = 0.0f;
    float foot_y = 0.0f;
    float z = 0.0f;
    float radius = 75.0f;
};

struct GroundMistProfile {
    bool enabled = false;        // false -> ground mist disabled in this zone
    bool ground_mist_only = true;// true -> suppress tall upper-air froxel banks (clean ankle/waist ground mist)
    float base_height_cm = 20.0f;
    float surge_height_cm = 75.0f;
    float base_density = 0.14f;
    float surge_density = 0.34f;
    float wind_x = 0.85f;        // ambient Navier-Stokes breeze (texels/sec)
    float wind_z = 0.45f;
    float tint[3]{1.0f, 1.0f, 1.0f}; // multiplicative mist color tint
};

struct GroundHeightField {
    static constexpr std::uint32_t kSize = 64u;
    static constexpr float kInvalidHeight = -9000.0f;
    float origin_x = 0.0f;
    float origin_z = 0.0f;
    float cell_size = 150.0f; // 64 * 150 = 9600 cm (96m x 96m)
    std::uint32_t revision = 0u;
    std::uint16_t area = 0u;
    bool valid = false;
    std::shared_ptr<const std::array<float, kSize * kSize>> heights;
};

struct SceneCamera {
    static constexpr std::size_t kMaxWakeActors = 8u;
    std::array<float, 16> world_to_eye{};
    bool valid = false;
    std::array<float, 3> hunter_feet{};
    bool has_hunter_feet = false;
    std::uint16_t ground_area = 0;
    float ground_mist_weather = -1.0f; // 0..1 when supplied by the game host
    std::array<SceneWakeActor, kMaxWakeActors> wake_actors{};
    std::uint32_t wake_actor_count = 0u;
    GroundHeightField heightfield{};
};

struct SemanticSceneFrame {
    static constexpr std::size_t kMaxLights = 32;
    std::uint32_t light_count{0};
    std::array<SemanticPointLight, kMaxLights> lights{};
};

} // namespace mhf::psp
