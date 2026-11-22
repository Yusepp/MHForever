#include "quest_timer_skip.hpp"

#include <algorithm>

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

// MH-inspired warm dark leather/slate & antique-gold palette.
constexpr mhf::overlay::Color kMhCardTop{0.125f, 0.106f, 0.086f};
constexpr mhf::overlay::Color kMhCardBot{0.063f, 0.055f, 0.047f};
constexpr mhf::overlay::Color kMhGold{0.847f, 0.706f, 0.392f};
constexpr mhf::overlay::Color kMhGoldBright{0.988f, 0.843f, 0.451f};
constexpr mhf::overlay::Color kMhGoldDim{0.443f, 0.357f, 0.204f};
constexpr mhf::overlay::Color kMhParchment{0.969f, 0.937f, 0.855f};
constexpr mhf::overlay::Color kMhSubtle{0.761f, 0.706f, 0.588f};
constexpr mhf::overlay::Color kMhPillTop{0.216f, 0.184f, 0.141f};
constexpr mhf::overlay::Color kMhPillBot{0.118f, 0.098f, 0.075f};

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
    const float text_size = 9.0f;
    const float text_w = painter.font_ok() ? painter.measure(text, text_size, true) : 84.0f;

    const float pill_w = 38.0f;
    const float pill_h = 14.0f;
    const float pad_x = 8.0f;
    const float gap = 7.0f;
    const float card_w = std::clamp(pad_x * 2.0f + pill_w + gap + text_w, 126.0f, 198.0f);
    const float card_h = 28.0f;
    const float card_x = 10.0f;
    const float card_y = 272.0f - card_h - 10.0f;

    // Soft drop shadow + MH leather/slate gradient panel + double antique-gold bevel frame.
    painter.rrect(card_x + 1.5f, card_y + 2.0f, card_w, card_h, 5.0f, mhf::overlay::kBlack, 0.45f);
    painter.rrect(card_x, card_y, card_w, card_h, 4.5f, kMhCardTop, 0.90f, 0.0f, &kMhCardBot);
    painter.rrect(card_x, card_y, card_w, card_h, 4.5f, holding_ ? kMhGoldBright : kMhGold, 0.92f, 1.0f);
    painter.rrect(card_x + 2.0f, card_y + 2.0f, card_w - 4.0f, card_h - 4.0f, 3.0f, kMhGoldDim, 0.55f, 0.75f);

    // PSP [SELECT] button pill on the left.
    const float pill_x = card_x + pad_x;
    const float pill_y = card_y + (card_h - 4.0f - pill_h) * 0.5f;
    painter.rrect(pill_x, pill_y, pill_w, pill_h, 7.0f, holding_ ? kMhGold : kMhPillTop, 0.95f, 0.0f,
                  holding_ ? &kMhGoldBright : &kMhPillBot);
    painter.rrect(pill_x, pill_y, pill_w, pill_h, 7.0f, holding_ ? kMhGoldBright : kMhGold, 0.95f, 1.0f);
    const float sel_w = painter.font_ok() ? painter.measure("SELECT", 6.8f, true, 0.4f) : 28.0f;
    painter.text(pill_x + (pill_w - sel_w) * 0.5f, pill_y + 9.5f, "SELECT", 6.8f,
                 holding_ ? mhf::overlay::kInk : kMhParchment, 0.98f, true, 0.4f);

    // Localized action text in crisp anti-aliased TrueType font.
    const float tx = pill_x + pill_w + gap;
    const float ty = pill_y + 10.2f;
    painter.text(tx + 0.8f, ty + 0.8f, text, text_size, mhf::overlay::kBlack, 0.75f, true);
    painter.text(tx, ty, text, text_size, holding_ ? kMhGoldBright : kMhParchment, 0.98f, true);

    // Sleek hold-progress bar tucked along the bottom interior of the card.
    const float bar_x = card_x + 6.0f;
    const float bar_y = card_y + card_h - 5.5f;
    const float bar_w = card_w - 12.0f;
    painter.rrect(bar_x, bar_y, bar_w, 2.5f, 1.25f, kMhSubtle, 0.25f);
    if (hold_progress_ > 0.01f) {
        painter.rrect(bar_x, bar_y, std::max(3.0f, bar_w * hold_progress_), 2.5f, 1.25f, kMhGoldBright, 0.98f);
    }
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
