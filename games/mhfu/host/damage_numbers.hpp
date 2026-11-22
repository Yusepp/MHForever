#pragma once

#include "psprecomp/memory.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <vector>

namespace mhf::mhfu {

struct CameraProjection {
    std::array<float, 3> focus_pos{};
    std::uint16_t yaw{0};
    float aspect{480.0f / 272.0f};
    bool valid{false};
};

// World/Rise/Wilds-style floating damage numbers projected from 3D monster hit positions
// onto the 480x272 premultiplied RGBA HUD overlay.
class DamageNumbers {
public:
    static constexpr std::uint64_t kLifetimeUs = 950'000u;

    void clear() noexcept;

    // Records a hit on a monster (safe to call from the guest thread hook).
    void record_hit(std::int32_t damage, const std::array<float, 3> &world_pos, std::uint64_t now_us) noexcept;

    // Reads Enemy* position and sp+62 damage from guest memory during mh_enemy_damage_apply.
    void on_enemy_damage_hook(const psprecomp::GuestMemory &memory, std::uint32_t enemy_ptr,
                              std::uint32_t sp, std::uint64_t now_us, bool enabled) noexcept;

    // Prunes expired popups and draws all active damage numbers onto `rgba` (480x272 premultiplied RGBA).
    // Returns true if at least one number was drawn.
    bool render(std::vector<std::uint32_t> &rgba, const CameraProjection &cam, std::uint64_t now_us);

    [[nodiscard]] std::size_t active_count() const;

private:
    struct Popup {
        std::int32_t damage{0};
        std::array<float, 3> world_pos{};
        std::uint64_t birth_us{0};
        float jitter_x{0.0f};
        float jitter_y{0.0f};
    };

    mutable std::mutex mutex_;
    std::vector<Popup> popups_;
    std::uint32_t hit_seq_{0};
};

} // namespace mhf::mhfu
