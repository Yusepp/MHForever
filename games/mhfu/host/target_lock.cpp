#include "target_lock.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace mhf::mhfu {
namespace {

constexpr std::uint32_t kCameraYawOffset = 0x80u; // s16 target yaw in CameraController (s4)
constexpr std::int32_t kMaxTurnStep = 2200;       // ~12 deg per 30fps frame, smooth & responsive
constexpr float kMinTargetDistSq = 1.0f;
constexpr int kOverlayW = 480;
constexpr int kOverlayH = 272;

float dist_sq_xz(const std::array<float, 3> &a, const std::array<float, 3> &b) noexcept {
    const float dx = b[0] - a[0];
    const float dz = b[2] - a[2];
    return dx * dx + dz * dz;
}

void blend_pixel(std::vector<std::uint32_t> &rgba, int x, int y, std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) noexcept {
    if (x < 0 || x >= kOverlayW || y < 0 || y >= kOverlayH || a == 0u) return;
    const std::size_t idx = static_cast<std::size_t>(y * kOverlayW + x);
    const std::uint32_t dst = rgba[idx];
    const std::uint32_t dr = dst & 0xFFu;
    const std::uint32_t dg = (dst >> 8u) & 0xFFu;
    const std::uint32_t db = (dst >> 16u) & 0xFFu;
    const std::uint32_t da = (dst >> 24u) & 0xFFu;

    const std::uint32_t sr = (static_cast<std::uint32_t>(r) * a + 127u) / 255u;
    const std::uint32_t sg = (static_cast<std::uint32_t>(g) * a + 127u) / 255u;
    const std::uint32_t sb = (static_cast<std::uint32_t>(b) * a + 127u) / 255u;
    const std::uint32_t inv = 255u - a;

    const std::uint32_t or_ = std::min<std::uint32_t>(255u, sr + (dr * inv + 127u) / 255u);
    const std::uint32_t og_ = std::min<std::uint32_t>(255u, sg + (dg * inv + 127u) / 255u);
    const std::uint32_t ob_ = std::min<std::uint32_t>(255u, sb + (db * inv + 127u) / 255u);
    const std::uint32_t oa_ = std::min<std::uint32_t>(255u, a + (da * inv + 127u) / 255u);
    rgba[idx] = or_ | (og_ << 8u) | (ob_ << 16u) | (oa_ << 24u);
}

} // namespace

void TargetLock::clear() noexcept {
    locked_enemy_ = 0u;
    target_.reset();
    prev_r3_ = false;
}

void TargetLock::update(const MonsterTracker &tracker, bool r3_pressed, bool enabled) noexcept {
    const bool r3_edge = r3_pressed && !prev_r3_;
    prev_r3_ = r3_pressed;

    if (!enabled || !tracker.in_active_quest()) {
        locked_enemy_ = 0u;
        target_.reset();
        return;
    }

    const std::uint16_t area = tracker.player_area();
    const auto ppos = tracker.player_pos();

    // Gather all alive large monsters in the hunter's current area, sorted by distance.
    std::vector<TrackedMonster> candidates;
    for (const auto &m : tracker.large_monsters()) {
        if (m.area == area && m.hp > 0) candidates.push_back(m);
    }
    std::sort(candidates.begin(), candidates.end(), [&](const TrackedMonster &a, const TrackedMonster &b) {
        return dist_sq_xz(ppos, a.pos) < dist_sq_xz(ppos, b.pos);
    });

    if (candidates.empty()) {
        locked_enemy_ = 0u;
        target_.reset();
        return;
    }

    // Check if the previously locked monster is still present in this area.
    std::optional<std::size_t> cur_idx;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].address == locked_enemy_) {
            cur_idx = i;
            break;
        }
    }
    if (!cur_idx) {
        locked_enemy_ = 0u;
        target_.reset();
    }

    if (r3_edge) {
        if (!cur_idx) {
            locked_enemy_ = candidates[0].address;
            target_ = candidates[0];
        } else if (*cur_idx + 1u < candidates.size()) {
            const std::size_t next = *cur_idx + 1u;
            locked_enemy_ = candidates[next].address;
            target_ = candidates[next];
        } else {
            // Cycled past the last monster in the area: unlock.
            locked_enemy_ = 0u;
            target_.reset();
        }
        return;
    }

    if (cur_idx) target_ = candidates[*cur_idx];
}

std::uint16_t TargetLock::yaw_between(const std::array<float, 3> &from, const std::array<float, 3> &to) noexcept {
    const float dx = to[0] - from[0];
    const float dz = to[2] - from[2];
    const float rad = std::atan2(dx, dz);
    const auto s16 = static_cast<std::int32_t>(std::lround(rad * (32768.0f / std::numbers::pi_v<float>)));
    return static_cast<std::uint16_t>(s16);
}

std::uint16_t TargetLock::step_yaw(std::uint16_t current, std::uint16_t desired, std::int32_t max_step) noexcept {
    const auto diff = static_cast<std::int16_t>(static_cast<std::uint16_t>(desired - current));
    const std::int32_t delta = std::clamp<std::int32_t>(diff / 3, -max_step, max_step);
    if (std::abs(diff) <= 48) return desired;
    return static_cast<std::uint16_t>(current + static_cast<std::uint16_t>(delta == 0 ? (diff > 0 ? 1 : -1) : delta));
}

bool TargetLock::steer_camera(psprecomp::GuestMemory &memory, std::uint32_t camera_controller,
                              const std::array<float, 3> &player_pos, bool manual_turn_active) const noexcept {
    if (!target_ || manual_turn_active || camera_controller == 0u) return false;
    if (dist_sq_xz(player_pos, target_->pos) < kMinTargetDistSq) return false;
    const std::uint32_t yaw_addr = camera_controller + kCameraYawOffset;
    if (!memory.contains(yaw_addr, 2u)) return false;

    const std::uint16_t current = memory.load16(yaw_addr);
    const std::uint16_t desired = yaw_between(player_pos, target_->pos);
    const std::uint16_t next = step_yaw(current, desired, kMaxTurnStep);
    memory.store16(yaw_addr, next);
    return true;
}

void TargetLock::draw_lock_badge(std::vector<std::uint32_t> &rgba, int cx, int cy, float pulse_phase) noexcept {
    const float pulse = 0.80f + 0.20f * std::sin(pulse_phase);
    const auto rim_a = static_cast<std::uint8_t>(std::clamp(pulse * 240.0f, 0.0f, 255.0f));

    // Dark medallion + crisp amber-gold target lock reticle (World/Rise/Wilds style).
    for (int dy = -11; dy <= 11; ++dy) {
        for (int dx = -11; dx <= 11; ++dx) {
            const float d = std::sqrt(static_cast<float>(dx * dx + dy * dy));
            if (d <= 9.5f) {
                blend_pixel(rgba, cx + dx, cy + dy, 18, 14, 10, 185);
            }
            if (d >= 6.4f && d <= 8.2f) {
                blend_pixel(rgba, cx + dx, cy + dy, 255, 200, 70, rim_a);
            } else if (d <= 2.1f) {
                blend_pixel(rgba, cx + dx, cy + dy, 255, 225, 110, 245);
            }
        }
    }
    // 4 cardinal crosshair ticks
    for (int t = 4; t <= 10; ++t) {
        blend_pixel(rgba, cx + t, cy, 255, 215, 90, 245);
        blend_pixel(rgba, cx - t, cy, 255, 215, 90, 245);
        blend_pixel(rgba, cx, cy + t, 255, 215, 90, 245);
        blend_pixel(rgba, cx, cy - t, 255, 215, 90, 245);
    }
}

} // namespace mhf::mhfu
