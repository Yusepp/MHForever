#pragma once

#include "monster_tracker.hpp"
#include "psprecomp/memory.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace mhf::mhfu {

// Monster Target Lock-On Camera (R3): press R3 to lock onto or cycle between large monsters
// in the hunter's current area; smoothly turns the camera toward the locked monster.
class TargetLock {
public:
    void clear() noexcept;

    // Updates lock-on state from the latest tracked large monsters and R3 button state.
    void update(const MonsterTracker &tracker, bool r3_pressed, bool enabled) noexcept;

    // Smoothly turns the camera controller's yaw (+0x80) toward the locked monster if active.
    // Returns true if the camera was steered this frame.
    bool steer_camera(psprecomp::GuestMemory &memory, std::uint32_t camera_controller,
                      const std::array<float, 3> &player_pos, bool manual_turn_active) const noexcept;

    [[nodiscard]] bool locked() const noexcept { return locked_enemy_ != 0u; }
    [[nodiscard]] std::uint32_t locked_enemy() const noexcept { return locked_enemy_; }
    [[nodiscard]] std::optional<TrackedMonster> target() const noexcept { return target_; }

    // Shortest-path angular step in MHFU's s16 yaw space (65536 = 360 deg).
    [[nodiscard]] static std::uint16_t step_yaw(std::uint16_t current, std::uint16_t desired, std::int32_t max_step) noexcept;
    [[nodiscard]] static std::uint16_t yaw_between(const std::array<float, 3> &from, const std::array<float, 3> &to) noexcept;

    // Draws a World/Rise-style golden target lock reticle badge onto a 480x272 premultiplied RGBA overlay.
    static void draw_lock_badge(std::vector<std::uint32_t> &rgba, int cx, int cy, float pulse_phase) noexcept;

private:
    std::uint32_t locked_enemy_{0};
    std::optional<TrackedMonster> target_;
    bool prev_r3_{false};
};

} // namespace mhf::mhfu
