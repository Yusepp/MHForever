#include "quest_timer_skip.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

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

std::uint32_t premul(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) noexcept {
    const auto pr = static_cast<std::uint32_t>((static_cast<int>(r) * a + 127) / 255);
    const auto pg = static_cast<std::uint32_t>((static_cast<int>(g) * a + 127) / 255);
    const auto pb = static_cast<std::uint32_t>((static_cast<int>(b) * a + 127) / 255);
    return pr | (pg << 8u) | (pb << 16u) | (static_cast<std::uint32_t>(a) << 24u);
}

// Crisp 3x5 bitmap glyphs for "SELECT" inside the PSP-style pill button.
constexpr const char *kSelectGlyphs[6][5] = {
    {"111", "100", "111", "001", "111"}, // S
    {"111", "100", "110", "100", "111"}, // E
    {"100", "100", "100", "100", "111"}, // L
    {"111", "100", "110", "100", "111"}, // E
    {"111", "100", "100", "100", "111"}, // C
    {"111", "010", "010", "010", "010"}, // T
};

// Renders a Monster Hunter / PSP-styled [SELECT] pill with an amber-gold fill bar as the button is held.
std::shared_ptr<const std::vector<std::uint32_t>> make_select_pill_sprite(float progress) {
    constexpr int w = 36, h = 14;
    auto px = std::make_shared<std::vector<std::uint32_t>>(static_cast<std::size_t>(w * h), 0u);
    const int fill_x = 2 + static_cast<int>(std::lround(std::clamp(progress, 0.0f, 1.0f) * static_cast<float>(w - 4)));

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Rounded pill SDF
            const int rx = x < 5 ? 5 - x : (x >= w - 5 ? x - (w - 6) : 0);
            const int ry = y < 5 ? 5 - y : (y >= h - 5 ? y - (h - 6) : 0);
            const float d = std::sqrt(static_cast<float>(rx * rx + ry * ry));
            if (d > 5.4f) continue;

            std::uint32_t color = 0u;
            if (d >= 4.2f) {
                // Warm antique-gold MH border
                color = progress > 0.01f ? premul(255, 212, 96, 245) : premul(204, 174, 108, 235);
            } else if (x < fill_x && progress > 0.01f) {
                // Holding fill: warm MH quest-clear amber
                color = premul(215, 148, 36, 225);
            } else {
                // Deep charcoal PSP button interior with subtle top highlight
                const std::uint8_t shade = y <= 4 ? 46 : 26;
                color = premul(shade, static_cast<std::uint8_t>(shade - 2), static_cast<std::uint8_t>(shade - 6), 220);
            }
            (*px)[static_cast<std::size_t>(y * w + x)] = color;
        }
    }

    // Draw crisp "SELECT" lettering centered inside the pill
    const int start_x = 7;
    const int start_y = 5;
    for (int ch = 0; ch < 6; ++ch) {
        for (int row = 0; row < 5; ++row) {
            for (int col = 0; col < 3; ++col) {
                if (kSelectGlyphs[ch][row][col] == '1') {
                    const int x = start_x + ch * 4 + col;
                    const int y = start_y + row;
                    (*px)[static_cast<std::size_t>(y * w + x)] = premul(248, 244, 230, 250);
                }
            }
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

std::string QuestTimerSkip::hint(int language) const {
    if (!can_skip_ || triggered_) return {};
    if (holding_) {
        switch (language) {
        case 0: return "帰還中...";
        case 2: return "Retour au village...";
        case 3: return "Volviendo a la aldea...";
        case 4: return "Rückkehr ins Dorf...";
        case 5: return "Ritorno al villaggio...";
        default: return "Returning to village...";
        }
    }
    switch (language) {
    case 0: return "SELECT長押しで帰還";
    case 2: return "Maintenir SELECT : Retour";
    case 3: return "Mantener SELECT: Volver";
    case 4: return "SELECT halten: Zurück";
    case 5: return "Tieni SELECT: Torna";
    default: return "Hold SELECT: Return";
    }
}

std::vector<mhf::overlay::HintSprite> QuestTimerSkip::sprites() const {
    if (!can_skip_ || triggered_) return {};
    mhf::overlay::HintSprite pill;
    pill.pixels = make_select_pill_sprite(hold_progress_);
    pill.width = 36;
    pill.height = 14;
    pill.x_px = 416;
    pill.y_px = 248;
    pill.draw_width = 36;
    pill.draw_height = 14;
    return {std::move(pill)};
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
