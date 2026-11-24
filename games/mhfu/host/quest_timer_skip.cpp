#include "quest_timer_skip.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mhf::mhfu {
namespace {

// Struct offsets verified identical across EU (ULES01213), US (ULUS10391), JP (ULJM05500) and FUComplete.
// QuestManager: +0x00 s8 state (1 = Quest Clear), +0x710 u32 post-quest countdown frames (1800 = 60s, 600 = 20s).
constexpr std::uint32_t kQuestState = 0x00u;
constexpr std::uint32_t kPostQuestCountdown = 0x710u;
constexpr std::uint32_t kQuestManagerMinSize = kPostQuestCountdown + 4u;
constexpr std::int8_t kQuestStateClear = 1;

// GameSys +0x2E: u8 quest phase (3 during post-quest countdown after Quest Clear).
constexpr std::uint32_t kGameSysQuestPhase = 0x2Eu;
constexpr std::uint8_t kQuestPhasePostClear = 3u;

bool readable_word_address(const psprecomp::GuestMemory &memory, std::uint32_t address) noexcept {
    return (address & 3u) == 0u && memory.contains(address, 4u);
}

// Native MHFU HUD palette (matching Pokke House / Smithy green ribbons & antique-brass frames).
constexpr mhf::overlay::Color kMhRibbonBg{0.114f, 0.251f, 0.078f};
constexpr mhf::overlay::Color kMhBrassLight{0.804f, 0.651f, 0.373f};
constexpr mhf::overlay::Color kMhBrassDark{0.463f, 0.349f, 0.188f};
constexpr mhf::overlay::Color kMhFrameInk{0.055f, 0.047f, 0.039f};
constexpr mhf::overlay::Color kMhHeaderYellow{0.925f, 0.871f, 0.341f};
constexpr mhf::overlay::Color kMhWhite{0.960f, 0.955f, 0.935f};
constexpr mhf::overlay::Color kBtnCapBg{0.110f, 0.110f, 0.110f};
constexpr mhf::overlay::Color kBtnCapBorder{0.620f, 0.610f, 0.560f};

std::uint32_t premul_rgba(float r, float g, float b, float a) noexcept {
    a = std::clamp(a, 0.0f, 1.0f);
    const auto to_u8 = [&](float c) -> std::uint32_t {
        return static_cast<std::uint32_t>(std::clamp(c * a * 255.0f + 0.5f, 0.0f, 255.0f));
    };
    const auto au8 = static_cast<std::uint32_t>(std::clamp(a * 255.0f + 0.5f, 0.0f, 255.0f));
    return to_u8(r) | (to_u8(g) << 8u) | (to_u8(b) << 16u) | (au8 << 24u);
}

std::uint32_t blend_over(std::uint32_t dst, std::uint32_t src) noexcept {
    const std::uint32_t sa = src >> 24u;
    if (sa == 0u) return dst;
    if (sa == 255u || dst == 0u) return src;
    const std::uint32_t inv = 255u - sa;
    const auto mix = [&](unsigned shift) -> std::uint32_t {
        const std::uint32_t sc = (src >> shift) & 0xFFu;
        const std::uint32_t dc = (dst >> shift) & 0xFFu;
        return std::min(255u, sc + (dc * inv + 127u) / 255u);
    };
    return mix(0u) | (mix(8u) << 8u) | (mix(16u) << 16u) | (mix(24u) << 24u);
}

// Rasterizes a crisp 96x96 anti-aliased MHFU circular medallion with clockwise radial hold-to-fill arc.
std::vector<std::uint32_t> build_radial_medallion(float progress, bool holding) {
    constexpr int kRes = 96;
    constexpr float kHalf = 48.0f;
    constexpr float kPi = 3.14159265359f;
    constexpr float kTwoPi = 2.0f * kPi;

    std::vector<std::uint32_t> px(static_cast<std::size_t>(kRes * kRes), 0u);
    const float fill_angle = std::clamp(progress, 0.0f, 1.0f) * kTwoPi;

    for (int y = 0; y < kRes; ++y) {
        for (int x = 0; x < kRes; ++x) {
            const float dx = (static_cast<float>(x) + 0.5f) - kHalf;
            const float dy = (static_cast<float>(y) + 0.5f) - kHalf;
            const float r = std::sqrt(dx * dx + dy * dy);
            if (r > 47.0f) continue;

            std::uint32_t col = 0u;

            // Soft outer drop-shadow halo
            if (r < 46.5f) {
                const float sh = std::clamp((46.5f - r) / 4.5f, 0.0f, 1.0f) * 0.42f;
                col = premul_rgba(0.0f, 0.0f, 0.0f, sh);
            }

            // Dark ink outer rim (r = 41.5..44.2)
            if (r < 44.5f) {
                const float edge = std::clamp(44.5f - r, 0.0f, 1.0f);
                col = blend_over(col, premul_rgba(kMhFrameInk.r, kMhFrameInk.g, kMhFrameInk.b, 0.96f * edge));
            }

            // Antique-brass MHFU outer ring with 8 subtle compass notches (r = 38.2..42.0)
            if (r >= 37.5f && r <= 42.5f) {
                const float band = std::clamp(std::min(r - 37.5f, 42.5f - r), 0.0f, 1.0f);
                float ang = std::atan2(dx, -dy);
                if (ang < 0.0f) ang += kTwoPi;
                const float sector = std::fmod(ang + kPi / 16.0f, kPi / 4.0f);
                const bool notch = sector < 0.075f;
                const auto &bc = notch ? kMhBrassDark : (holding ? kMhHeaderYellow : kMhBrassLight);
                col = blend_over(col, premul_rgba(bc.r, bc.g, bc.b, 0.96f * band));
            }

            // Smoked dark-charcoal interior medallion (r < 38.5)
            if (r < 38.5f) {
                const float inner = std::clamp(38.5f - r, 0.0f, 1.0f);
                col = blend_over(col, premul_rgba(0.065f, 0.060f, 0.052f, 0.92f * inner));
            }

            // Recessed radial progress track (r = 29.0..37.2)
            if (r >= 28.5f && r <= 37.5f) {
                const float track = std::clamp(std::min(r - 28.5f, 37.5f - r), 0.0f, 1.0f);
                col = blend_over(col, premul_rgba(0.18f, 0.24f, 0.14f, 0.55f * track));
            }

            // Clockwise radial hold-to-fill arc starting at 12 o'clock (r = 29.5..37.0)
            if (fill_angle > 0.005f && r >= 29.0f && r <= 37.5f) {
                float ang = std::atan2(dx, -dy); // 0 at 12 o'clock, increases clockwise
                if (ang < 0.0f) ang += kTwoPi;
                const float radial_aa = std::clamp(std::min(r - 29.0f, 37.5f - r), 0.0f, 1.0f);
                const float aa_ang = 1.2f / std::max(r, 1.0f);
                const float start_aa = std::clamp(ang / aa_ang, 0.0f, 1.0f);
                const float end_aa = std::clamp((fill_angle - ang) / aa_ang, 0.0f, 1.0f);
                const float cov = radial_aa * (progress >= 0.999f ? 1.0f : std::min(start_aa, end_aa));
                if (cov > 0.001f) {
                    const float t = std::clamp(ang / kTwoPi, 0.0f, 1.0f);
                    // Smooth gradient from MHFU emerald-green to bright antique-gold
                    const float cr = 0.16f + 0.80f * t;
                    const float cg = 0.82f + 0.10f * t;
                    const float cb = 0.28f + 0.06f * (1.0f - t);
                    col = blend_over(col, premul_rgba(cr, cg, cb, 0.98f * cov));

                    // Bright glowing tip at the leading edge
                    if (progress < 0.995f && std::abs(fill_angle - ang) < 0.14f) {
                        const float tip = 1.0f - std::abs(fill_angle - ang) / 0.14f;
                        col = blend_over(col, premul_rgba(1.0f, 0.98f, 0.78f, 0.72f * tip * radial_aa));
                    }
                }
            }

            // Inner brass bezel separating track from center button (r = 27.2..29.2)
            if (r >= 27.0f && r <= 29.4f) {
                const float ring = std::clamp(std::min(r - 27.0f, 29.4f - r), 0.0f, 1.0f);
                col = blend_over(col, premul_rgba(kMhBrassDark.r, kMhBrassDark.g, kMhBrassDark.b, 0.90f * ring));
            }

            px[static_cast<std::size_t>(y * kRes + x)] = col;
        }
    }
    return px;
}

// Rasterizes a crisp 64x64 anti-aliased MHFU hourglass that drains sand over 5s and flips 180 deg every 5s.
std::vector<std::uint32_t> build_hourglass_sprite(std::uint32_t remaining_frames) {
    constexpr int kRes = 64;
    constexpr float kHalf = 32.0f;
    constexpr float kPi = 3.14159265359f;

    const float sec_in_5 = std::fmod(static_cast<float>(remaining_frames) / 30.0f, 5.0f);
    constexpr float kFlipDur = 0.45f;
    float flip_angle = 0.0f;
    float top_fill = std::clamp(sec_in_5 / 5.0f, 0.0f, 1.0f);
    if (sec_in_5 < kFlipDur && remaining_frames > 15u) {
        const float u = std::clamp((kFlipDur - sec_in_5) / kFlipDur, 0.0f, 1.0f);
        const float ease = u * u * (3.0f - 2.0f * u);
        flip_angle = ease * kPi;
        top_fill = 0.0f;
    }
    const float bot_fill = 1.0f - top_fill;
    const float cs = std::cos(flip_angle);
    const float sn = std::sin(flip_angle);

    std::vector<std::uint32_t> px(static_cast<std::size_t>(kRes * kRes), 0u);
    for (int y = 0; y < kRes; ++y) {
        for (int x = 0; x < kRes; ++x) {
            const float rx = (static_cast<float>(x) + 0.5f) - kHalf;
            const float ry = (static_cast<float>(y) + 0.5f) - kHalf;
            // Inverse-rotate into hourglass local space
            const float lx = rx * cs + ry * sn;
            const float ly = -rx * sn + ry * cs;
            const float ax = std::abs(lx);
            const float ay = std::abs(ly);

            std::uint32_t col = 0u;

            // Antique-brass top & bottom horizontal endcaps (|ly| in 19..24, |lx| <= 16)
            if (ay >= 18.5f && ay <= 24.5f && ax <= 16.5f) {
                const float ex = std::clamp(16.5f - ax, 0.0f, 1.0f);
                const float ey = std::clamp(std::min(ay - 18.5f, 24.5f - ay), 0.0f, 1.0f);
                const float cov = std::min(ex, ey);
                const bool outer = ay > 23.0f || ay < 19.6f || ax > 15.0f;
                const auto &c = outer ? kMhFrameInk : (ay > 21.0f ? kMhBrassLight : kMhBrassDark);
                col = blend_over(col, premul_rgba(c.r, c.g, c.b, 0.98f * cov));
            }

            // Glass hourglass bulb profile (|ly| < 19.0): widens linearly/curved from waist (2.2) to cap (13.0)
            if (ay < 19.2f) {
                const float t = ay / 19.0f; // 0 at neck, 1 at cap
                const float half_w = 2.0f + 11.2f * std::pow(t, 0.85f);
                const float outer_w = half_w + 2.2f;
                if (ax <= outer_w) {
                    // Dark ink + subtle glass outline
                    if (ax >= half_w - 0.5f) {
                        const float wall = std::clamp(std::min(ax - (half_w - 0.5f), outer_w - ax), 0.0f, 1.0f);
                        const bool brass_edge = ax < half_w + 0.9f;
                        const auto &wc = brass_edge ? kMhBrassLight : kMhFrameInk;
                        col = blend_over(col, premul_rgba(wc.r, wc.g, wc.b, 0.92f * wall));
                    } else {
                        // Smoked glass interior
                        col = blend_over(col, premul_rgba(0.10f, 0.12f, 0.10f, 0.58f));

                        // Glowing MHFU emerald-gold sand inside top (ly < 0) and bottom (ly > 0) bulbs
                        const float fill_h_top = 17.5f * top_fill;
                        const float fill_h_bot = 17.5f * bot_fill;
                        bool in_sand = false;
                        float sand_glow = 0.0f;
                        if (ly < 0.0f && ay <= fill_h_top) {
                            in_sand = true;
                            sand_glow = 1.0f - (ay / std::max(fill_h_top, 1.0f)) * 0.35f;
                        } else if (ly > 0.0f && (19.0f - ay) <= fill_h_bot) {
                            in_sand = true;
                            sand_glow = 0.85f;
                        } else if (ly > 0.0f && top_fill > 0.02f && ax <= 1.25f) {
                            // Thin falling sand stream through the neck
                            in_sand = true;
                            sand_glow = 1.0f;
                        }
                        if (in_sand) {
                            const float edge = std::clamp(half_w - 0.8f - ax, 0.0f, 1.0f);
                            // Bright yellow-green sand matching the reference hourglass translated to MHFU
                            const float sr = 0.72f + 0.24f * sand_glow;
                            const float sg = 0.88f + 0.08f * sand_glow;
                            const float sb = 0.24f;
                            col = blend_over(col, premul_rgba(sr, sg, sb, 0.96f * edge));
                        }
                    }
                }
            }

            px[static_cast<std::size_t>(y * kRes + x)] = col;
        }
    }
    return px;
}

} // namespace

QuestTimerSkip::QuestTimerSkip(QuestTimerSkipSymbols symbols) noexcept : symbols_(symbols) {}

void QuestTimerSkip::clear() noexcept {
    active_quest_manager_ = 0;
    remaining_frames_ = 0;
    hold_start_us_ = 0;
    hold_progress_ = 0.0f;
    holding_ = false;
    triggered_ = false;
    can_skip_ = false;
}

std::string QuestTimerSkip::label(int language) const {
    if (!can_skip_ || triggered_) return {};
    if (holding_) {
        switch (language) {
        case 0: return "帰還中...";
        case 2: return "Retour en cours...";
        case 3: return "Volviendo...";
        case 4: return "Rückkehr...";
        case 5: return "Ritorno...";
        default: return "Returning...";
        }
    }
    switch (language) {
    case 0: return "すぐに帰還する";
    case 2: return "Terminer la quête";
    case 3: return "Finalizar misión ahora";
    case 4: return "Quest sofort beenden";
    case 5: return "Termina missione ora";
    default: return "End Quest Immediately";
    }
}

void QuestTimerSkip::render_card(mhf::overlay::Painter &painter, int language) const {
    if (!can_skip_ || triggered_) return;

    const float cx = 240.0f;
    // Anchor cleanly below MHFU's top Health & Stamina bars (y = 14..28) so there is zero HUD overlap.
    const float y0 = 33.0f;

    // 1. Top-center "Timer" heading (MHFU Header Yellow, matching reference layout)
    const std::string timer_hdr = (language == 0) ? "残り時間" : "Timer";
    const float hdr_w = painter.font_ok() ? painter.measure(timer_hdr, 7.6f, true, 0.2f) : 24.0f;
    painter.text(cx - hdr_w * 0.5f + 0.7f, y0 + 7.7f, timer_hdr, 7.6f, kMhFrameInk, 0.90f, true, 0.2f);
    painter.text(cx - hdr_w * 0.5f, y0 + 7.0f, timer_hdr, 7.6f, kMhHeaderYellow, 0.98f, true, 0.2f);

    // 2. Animated 5-second MHFU Hourglass centered below "Timer"
    const float hg_size = 19.0f;
    const auto hg_px = build_hourglass_sprite(remaining_frames_);
    painter.image(cx - hg_size * 0.5f, y0 + 8.5f, hg_size, hg_px.data(), 64, 1.0f);

    // 3. Real remaining countdown seconds (e.g. 60, 59, ..., 20, ..., 1)
    const std::uint32_t secs = remaining_seconds();
    const std::string sec_txt = std::to_string(secs);
    const float sec_size = 9.4f;
    const float sec_w = painter.font_ok() ? painter.measure(sec_txt, sec_size, true) : 14.0f;
    const mhf::overlay::Color sec_col = secs <= 10u ? mhf::overlay::Color{0.96f, 0.42f, 0.34f} : kMhWhite;
    painter.text(cx - sec_w * 0.5f + 0.8f, y0 + 36.3f, sec_txt, sec_size, kMhFrameInk, 0.92f, true);
    painter.text(cx - sec_w * 0.5f, y0 + 35.5f, sec_txt, sec_size, sec_col, 0.98f, true);

    // 4. Tapered MHFU dark forest-green & antique-brass center banner ("End Quest Immediately")
    const std::string text = label(language);
    const float text_size = 8.4f;
    const float text_w = painter.font_ok() ? painter.measure(text, text_size, false, 0.3f) : 92.0f;
    const float ban_w = std::clamp(text_w + 56.0f, 138.0f, 220.0f);
    const float ban_h = 16.5f;
    const float ban_x = cx - ban_w * 0.5f;
    const float ban_y = y0 + 54.5f;

    constexpr int kBands = 24;
    const float bw = ban_w / static_cast<float>(kBands);
    for (int i = 0; i < kBands; ++i) {
        const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(kBands);
        const float dist = std::abs(u - 0.5f) * 2.0f; // 0 at center, 1 at ends
        const float fade = dist <= 0.55f ? 1.0f : std::max(0.0f, 1.0f - (dist - 0.55f) / 0.45f);
        painter.rect(ban_x + static_cast<float>(i) * bw, ban_y, bw + 0.35f, ban_h, kMhRibbonBg, 0.84f * fade);
        painter.rect(ban_x + static_cast<float>(i) * bw, ban_y, bw + 0.35f, 0.8f, kMhBrassLight, 0.88f * fade);
        painter.rect(ban_x + static_cast<float>(i) * bw, ban_y + ban_h - 0.8f, bw + 0.35f, 0.8f, kMhBrassLight, 0.88f * fade);
    }
    // Small antique-brass diamond clasp centered on the bottom edge of the banner
    painter.rect(cx - 2.2f, ban_y + ban_h - 2.0f, 4.4f, 4.0f, kMhFrameInk, 0.95f);
    painter.rect(cx - 1.4f, ban_y + ban_h - 1.2f, 2.8f, 2.4f, kMhBrassLight, 0.98f);

    // Centered localized action text
    painter.text(cx - text_w * 0.5f + 0.7f, ban_y + 11.6f, text, text_size, kMhFrameInk, 0.90f, false, 0.3f);
    painter.text(cx - text_w * 0.5f, ban_y + 10.9f, text, text_size, holding_ ? kMhHeaderYellow : kMhWhite, 0.98f, false, 0.3f);

    // 5. Circular MHFU radial hold-to-fill medallion centered directly above the banner
    const float med_size = 20.0f;
    const float med_x = cx - med_size * 0.5f;
    const float med_y = y0 + 38.0f;
    const auto medallion = build_radial_medallion(hold_progress_, holding_);
    painter.image(med_x, med_y, med_size, medallion.data(), 96, 1.0f);

    const float cap_w = 11.5f;
    const float cap_h = 6.2f;
    const float cap_x = med_x + (med_size - cap_w) * 0.5f;
    const float cap_y = med_y + (med_size - cap_h) * 0.5f;
    painter.rrect(cap_x, cap_y, cap_w, cap_h, 1.6f, kBtnCapBg, 0.96f);
    painter.rrect(cap_x, cap_y, cap_w, cap_h, 1.6f, holding_ ? kMhHeaderYellow : kBtnCapBorder, 0.95f, 0.65f);
    const std::string cap_lbl = "SEL";
    const float cap_tw = painter.font_ok() ? painter.measure(cap_lbl, 4.4f, true, 0.15f) : 8.0f;
    painter.text(cap_x + (cap_w - cap_tw) * 0.5f, cap_y + 4.7f, cap_lbl, 4.4f,
                 holding_ ? kMhHeaderYellow : kMhWhite, 0.98f, true, 0.15f);
}

bool QuestTimerSkip::poll(psprecomp::GuestMemory &memory, std::uint32_t buttons, std::uint64_t now_us, bool enabled) {
    if (!enabled || !symbols_.complete()) {
        clear();
        return false;
    }
    if (!readable_word_address(memory, *symbols_.quest_manager) || !readable_word_address(memory, *symbols_.game_sys)) {
        clear();
        return false;
    }

    const std::uint32_t quest_mgr = memory.load32(*symbols_.quest_manager);
    const std::uint32_t game_sys = memory.load32(*symbols_.game_sys);
    if (quest_mgr == 0u || (quest_mgr & 3u) != 0u || !memory.contains(quest_mgr, kQuestManagerMinSize) ||
        game_sys == 0u || !memory.contains(game_sys + kGameSysQuestPhase, 1u)) {
        clear();
        return false;
    }

    const auto state = static_cast<std::int8_t>(memory.load8(quest_mgr + kQuestState));
    const std::uint8_t phase = memory.load8(game_sys + kGameSysQuestPhase);
    const std::uint32_t countdown = memory.load32(quest_mgr + kPostQuestCountdown);

    if (state != kQuestStateClear || phase != kQuestPhasePostClear || countdown <= kQuestTimerSkipMinFrames) {
        clear();
        return false;
    }

    if (quest_mgr != active_quest_manager_) {
        active_quest_manager_ = quest_mgr;
        hold_start_us_ = 0;
        hold_progress_ = 0.0f;
        holding_ = false;
        triggered_ = false;
    }
    remaining_frames_ = countdown;
    can_skip_ = !triggered_;
    if (triggered_) return false;

    if ((buttons & kQuestTimerSkipButton) == 0u) {
        holding_ = false;
        hold_start_us_ = 0;
        hold_progress_ = 0.0f;
        return false;
    }
    if (!holding_) {
        holding_ = true;
        hold_start_us_ = now_us;
        hold_progress_ = 0.0f;
        return false;
    }
    if (now_us < hold_start_us_) {
        hold_start_us_ = now_us;
        hold_progress_ = 0.0f;
        return false;
    }
    const std::uint64_t elapsed = now_us - hold_start_us_;
    hold_progress_ = std::clamp(static_cast<float>(elapsed) / static_cast<float>(kQuestTimerSkipHoldUs), 0.0f, 1.0f);
    if (elapsed < kQuestTimerSkipHoldUs) return false;

    memory.store32(quest_mgr + kPostQuestCountdown, kQuestTimerSkipTargetFrames);
    triggered_ = true;
    can_skip_ = false;
    hold_progress_ = 1.0f;
    return true;
}

} // namespace mhf::mhfu
