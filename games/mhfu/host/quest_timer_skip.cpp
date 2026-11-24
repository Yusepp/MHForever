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
constexpr mhf::overlay::Color kMhRibbonEdge{0.278f, 0.420f, 0.180f};
constexpr mhf::overlay::Color kMhRibbonTribal{0.235f, 0.392f, 0.145f};
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

} // namespace

QuestTimerSkip::QuestTimerSkip(QuestTimerSkipSymbols symbols) noexcept : symbols_(symbols) {}

void QuestTimerSkip::clear() noexcept {
    active_quest_manager_ = 0;
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
    case 0: return "長押しで村へ帰還";
    case 2: return "Maintenir : Retour";
    case 3: return "Mantener: Volver";
    case 4: return "Halten: Zurück";
    case 5: return "Tieni: Torna";
    default: return "Hold: Return Now";
    }
}

void QuestTimerSkip::render_card(mhf::overlay::Painter &painter, int language) const {
    if (!can_skip_ || triggered_) return;

    const std::string text = label(language);
    const float text_size = 8.6f;
    const float text_w = painter.font_ok() ? painter.measure(text, text_size, false, 0.35f) : 78.0f;

    // Left-anchored MHFU dark forest-green ribbon (matching Pokke House / Smithy HUD bars)
    // paired with a circular clockwise radial hold-to-fill medallion around [SELECT].
    const float med_size = 30.0f;
    const float med_x = 10.0f;
    const float med_y = 272.0f - med_size - 9.0f;
    const float rib_y = med_y + 4.5f;
    const float rib_h = 21.0f;
    const float rib_w = std::clamp(med_x + med_size + 8.0f + text_w + 26.0f, 146.0f, 224.0f);

    // Translucent MHFU forest-green fading ribbon with subtle olive rails and right tribal cap
    constexpr int kBands = 18;
    const float bw = rib_w / static_cast<float>(kBands);
    for (int i = 0; i < kBands; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kBands - 1);
        const float fade = t <= 0.72f ? 1.0f : std::max(0.0f, 1.0f - (t - 0.72f) / 0.28f);
        painter.rect(static_cast<float>(i) * bw, rib_y, bw + 0.35f, rib_h, kMhRibbonBg, 0.86f * fade);
    }
    painter.rect(0.0f, rib_y, rib_w - 14.0f, 0.85f, kMhRibbonEdge, 0.88f);
    painter.rect(0.0f, rib_y + rib_h - 0.85f, rib_w - 14.0f, 0.85f, kMhRibbonEdge, 0.88f);

    // Geometric MHFU tribal motif on the right end of the ribbon
    const float tx_end = rib_w - 22.0f;
    painter.rect(tx_end, rib_y + 2.0f, 1.2f, rib_h - 4.0f, kMhRibbonTribal, 0.82f);
    painter.rect(tx_end + 2.8f, rib_y + 3.0f, 7.5f, 1.2f, kMhRibbonTribal, 0.78f);
    painter.rect(tx_end + 2.8f, rib_y + rib_h - 4.2f, 7.5f, 1.2f, kMhRibbonTribal, 0.78f);
    painter.rect(tx_end + 5.5f, rib_y + 6.0f, 3.2f, rib_h - 12.0f, kMhRibbonTribal, 0.72f);

    // Circular MHFU radial hold-to-fill medallion
    const auto medallion = build_radial_medallion(hold_progress_, holding_);
    painter.image(med_x, med_y, med_size, medallion.data(), 96, 1.0f);

    // Centered PSP [SELECT] keycap inside the radial medallion
    const float cap_w = 15.5f;
    const float cap_h = 8.0f;
    const float cap_x = med_x + (med_size - cap_w) * 0.5f;
    const float cap_y = med_y + (med_size - cap_h) * 0.5f;
    painter.rrect(cap_x, cap_y, cap_w, cap_h, 2.0f, kBtnCapBg, 0.96f);
    painter.rrect(cap_x, cap_y, cap_w, cap_h, 2.0f, holding_ ? kMhHeaderYellow : kBtnCapBorder, 0.95f, 0.75f);
    const std::string cap_lbl = "SEL";
    const float cap_tw = painter.font_ok() ? painter.measure(cap_lbl, 5.2f, true, 0.25f) : 10.0f;
    painter.text(cap_x + (cap_w - cap_tw) * 0.5f, cap_y + 6.0f, cap_lbl, 5.2f,
                 holding_ ? kMhHeaderYellow : kMhWhite, 0.98f, true, 0.25f);

    // Crisp MHFU action text with dark drop shadow
    const float text_x = med_x + med_size + 7.0f;
    const float text_y = rib_y + 13.6f;
    painter.text(text_x + 0.75f, text_y + 0.75f, text, text_size, kMhFrameInk, 0.88f, false, 0.35f);
    painter.text(text_x, text_y, text, text_size, holding_ ? kMhHeaderYellow : kMhWhite, 0.98f, false, 0.35f);
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
