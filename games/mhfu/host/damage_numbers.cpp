#include "damage_numbers.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <string>

namespace mhf::mhfu {
namespace {

constexpr std::uint32_t kEnemyPos = 0x200u;
constexpr int kOverlayW = 480;
constexpr int kOverlayH = 272;
constexpr std::size_t kMaxPopups = 32u;

// Crisp 5x7 bold bitmap digits '0'..'9' for World/Rise/Wilds-style floating numbers.
constexpr const char *kDigits[10][7] = {
    {"01110", "11011", "11011", "11011", "11011", "11011", "01110"}, // 0
    {"00110", "01110", "00110", "00110", "00110", "00110", "01111"}, // 1
    {"01110", "11011", "00011", "00110", "01100", "11000", "11111"}, // 2
    {"01110", "11011", "00011", "00110", "00011", "11011", "01110"}, // 3
    {"00110", "01110", "11010", "11010", "11111", "00010", "00010"}, // 4
    {"11111", "11000", "11110", "00011", "00011", "11011", "01110"}, // 5
    {"01110", "11000", "11110", "11011", "11011", "11011", "01110"}, // 6
    {"11111", "00011", "00110", "00110", "01100", "01100", "01100"}, // 7
    {"01110", "11011", "11011", "01110", "11011", "11011", "01110"}, // 8
    {"01110", "11011", "11011", "01111", "00011", "00110", "01100"}, // 9
};

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

void draw_digit(std::vector<std::uint32_t> &rgba, int x0, int y0, int digit, int scale,
                std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t alpha) noexcept {
    if (digit < 0 || digit > 9) return;
    const auto outline_a = static_cast<std::uint8_t>(std::min<int>(255, static_cast<int>(alpha) * 220 / 255));
    // Dark outline pass (1px halo + subtle drop shadow)
    for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 5; ++col) {
            if (kDigits[digit][row][col] != '1') continue;
            for (int sy = 0; sy < scale; ++sy) {
                for (int sx = 0; sx < scale; ++sx) {
                    const int px = x0 + col * scale + sx;
                    const int py = y0 + row * scale + sy;
                    blend_pixel(rgba, px - 1, py, 16, 12, 10, outline_a);
                    blend_pixel(rgba, px + 1, py, 16, 12, 10, outline_a);
                    blend_pixel(rgba, px, py - 1, 16, 12, 10, outline_a);
                    blend_pixel(rgba, px, py + 1, 16, 12, 10, outline_a);
                    blend_pixel(rgba, px + 1, py + 2, 10, 8, 8, static_cast<std::uint8_t>(outline_a * 2 / 3));
                }
            }
        }
    }
    // Bright foreground pass with subtle top highlight (World/Rise/Wilds gradient feel)
    for (int row = 0; row < 7; ++row) {
        const float top_boost = row <= 2 ? 1.12f : 1.0f;
        const auto cr = static_cast<std::uint8_t>(std::min<int>(255, static_cast<int>(r * top_boost)));
        const auto cg = static_cast<std::uint8_t>(std::min<int>(255, static_cast<int>(g * top_boost)));
        const auto cb = static_cast<std::uint8_t>(std::min<int>(255, static_cast<int>(b * top_boost)));
        for (int col = 0; col < 5; ++col) {
            if (kDigits[digit][row][col] != '1') continue;
            for (int sy = 0; sy < scale; ++sy) {
                for (int sx = 0; sx < scale; ++sx) {
                    blend_pixel(rgba, x0 + col * scale + sx, y0 + row * scale + sy, cr, cg, cb, alpha);
                }
            }
        }
    }
}

void draw_crit_star(std::vector<std::uint32_t> &rgba, int cx, int cy, std::uint8_t alpha) noexcept {
    // Small World/Rise-style 4-point crimson-gold crit diamond right underneath heavy/crit hits.
    for (int d = -3; d <= 3; ++d) {
        const auto a = static_cast<std::uint8_t>(alpha * (4 - std::abs(d)) / 4);
        blend_pixel(rgba, cx + d, cy, 255, 110, 80, a);
        blend_pixel(rgba, cx, cy + d, 255, 110, 80, a);
    }
    blend_pixel(rgba, cx, cy, 255, 240, 175, alpha);
}

bool project_world(const std::array<float, 3> &world, const CameraProjection &cam, float &sx, float &sy) noexcept {
    if (!cam.valid) {
        sx = kOverlayW * 0.5f;
        sy = kOverlayH * 0.42f;
        return true;
    }
    // Reconstruct third-person camera behind focus_pos using MHFU camera yaw.
    const float yaw_rad = static_cast<float>(static_cast<std::int16_t>(cam.yaw)) * (std::numbers::pi_v<float> / 32768.0f);
    const float fx = std::sin(yaw_rad);
    const float fz = std::cos(yaw_rad);
    const float rx = fz;
    const float rz = -fx;

    // Camera sits ~520 units behind and ~190 units above hunter, looking slightly down.
    const float cam_x = cam.focus_pos[0] - fx * 520.0f;
    const float cam_y = cam.focus_pos[1] + 190.0f;
    const float cam_z = cam.focus_pos[2] - fz * 520.0f;

    const float dx = world[0] - cam_x;
    const float dy = world[1] + 135.0f - cam_y;
    const float dz = world[2] - cam_z;

    const float view_z = dx * fx + dz * fz;
    if (view_z < 40.0f) return false;
    const float view_x = dx * rx + dz * rz;
    const float view_y = dy + view_z * 0.12f;

    const float fov_scale = 1.45f;
    const float aspect = cam.aspect > 0.5f ? cam.aspect : (static_cast<float>(kOverlayW) / static_cast<float>(kOverlayH));
    const float ndc_x = (view_x / view_z) * (fov_scale / aspect);
    const float ndc_y = (view_y / view_z) * fov_scale;

    sx = (0.5f + ndc_x * 0.5f) * static_cast<float>(kOverlayW);
    sy = (0.5f - ndc_y * 0.5f) * static_cast<float>(kOverlayH);
    sx = std::clamp(sx, 28.0f, static_cast<float>(kOverlayW - 28));
    sy = std::clamp(sy, 24.0f, static_cast<float>(kOverlayH - 28));
    return true;
}

} // namespace

void DamageNumbers::clear() noexcept {
    std::scoped_lock lock(mutex_);
    popups_.clear();
}

void DamageNumbers::record_hit(std::int32_t damage, const std::array<float, 3> &world_pos, std::uint64_t now_us) noexcept {
    if (damage <= 0 || damage > 9999) return;
    std::scoped_lock lock(mutex_);
    if (popups_.size() >= kMaxPopups) popups_.erase(popups_.begin());
    const std::uint32_t seq = hit_seq_++;
    // Deterministic gentle fan-out so rapid multi-hits never stack on top of each other.
    static constexpr float kFanX[8] = {0.0f, 14.0f, -14.0f, 24.0f, -24.0f, 8.0f, -8.0f, 18.0f};
    static constexpr float kFanY[8] = {0.0f, -6.0f, -4.0f, -12.0f, -10.0f, 5.0f, 4.0f, -16.0f};
    popups_.push_back({damage, world_pos, now_us, kFanX[seq & 7u], kFanY[seq & 7u]});
}

void DamageNumbers::on_enemy_damage_hook(const psprecomp::GuestMemory &memory, std::uint32_t enemy_ptr,
                                         std::uint32_t sp, std::uint64_t now_us, bool enabled) noexcept {
    if (!enabled || enemy_ptr == 0u || !memory.contains(enemy_ptr + kEnemyPos, 12u) || !memory.contains(sp + 62u, 2u)) {
        return;
    }
    const auto damage = static_cast<std::int16_t>(memory.load16(sp + 62u));
    if (damage <= 0) return;
    const std::array<float, 3> pos = {
        std::bit_cast<float>(memory.load32(enemy_ptr + kEnemyPos + 0u)),
        std::bit_cast<float>(memory.load32(enemy_ptr + kEnemyPos + 4u)),
        std::bit_cast<float>(memory.load32(enemy_ptr + kEnemyPos + 8u)),
    };
    record_hit(damage, pos, now_us);
}

std::size_t DamageNumbers::active_count() const {
    std::scoped_lock lock(mutex_);
    return popups_.size();
}

bool DamageNumbers::render(std::vector<std::uint32_t> &rgba, const CameraProjection &cam, std::uint64_t now_us) {
    std::scoped_lock lock(mutex_);
    std::erase_if(popups_, [now_us](const Popup &p) {
        return now_us < p.birth_us || (now_us - p.birth_us) >= kLifetimeUs;
    });
    if (popups_.empty()) return false;

    bool drew = false;
    for (const Popup &p : popups_) {
        float sx = 0.0f, sy = 0.0f;
        if (!project_world(p.world_pos, cam, sx, sy)) continue;

        const float t = static_cast<float>(now_us - p.birth_us) / static_cast<float>(kLifetimeUs);
        // Smooth ease-out rise + snappy pop-in + clean fade-out in final 28%.
        const float rise = 24.0f * (1.0f - (1.0f - t) * (1.0f - t));
        const float fade = t > 0.72f ? std::clamp((1.0f - t) / 0.28f, 0.0f, 1.0f) : 1.0f;
        const auto alpha = static_cast<std::uint8_t>(std::lround(fade * 245.0f));
        if (alpha == 0u) continue;

        // World/Rise/Wilds color palette:
        //  - Heavy/Crit (>= 85): rich golden-orange + crit star underneath
        //  - Weakspot   (>= 40): warm orange-gold (#FFAE2B)
        //  - Tough/Low  (<  40): crisp off-white/silver (#E4E8EE)
        std::uint8_t r = 232, g = 236, b = 242;
        bool crit_star = false;
        if (p.damage >= 85) {
            r = 255; g = 186; b = 52;
            crit_star = true;
        } else if (p.damage >= 40) {
            r = 255; g = 174; b = 43;
        }

        const std::string text = std::to_string(p.damage);
        const int scale = (t < 0.12f && p.damage >= 40) ? 2 : (p.damage >= 85 ? 2 : 1);
        const int glyph_w = 5 * scale + scale;
        const int total_w = static_cast<int>(text.size()) * glyph_w - scale;
        const int draw_x = static_cast<int>(std::lround(sx + p.jitter_x)) - total_w / 2;
        const int draw_y = static_cast<int>(std::lround(sy + p.jitter_y - rise)) - (7 * scale) / 2;

        for (std::size_t i = 0; i < text.size(); ++i) {
            draw_digit(rgba, draw_x + static_cast<int>(i) * glyph_w, draw_y, text[i] - '0', scale, r, g, b, alpha);
        }
        if (crit_star) {
            draw_crit_star(rgba, draw_x + total_w / 2, draw_y + 7 * scale + 4, alpha);
        }
        drew = true;
    }
    return drew;
}

} // namespace mhf::mhfu
