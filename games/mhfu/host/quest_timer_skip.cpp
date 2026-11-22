#include "quest_timer_skip.hpp"

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

} // namespace

QuestTimerSkip::QuestTimerSkip(QuestTimerSkipSymbols symbols) noexcept : symbols_(symbols) {}

void QuestTimerSkip::clear() noexcept {
    active_quest_manager_ = 0;
    hold_start_us_ = 0;
    holding_ = false;
    triggered_ = false;
    can_skip_ = false;
}

std::string QuestTimerSkip::hint() const {
    if (!can_skip_ || triggered_) return {};
    return holding_ ? "Returning to village..." : "Hold Select to return now";
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
        holding_ = false;
        triggered_ = false;
    }
    can_skip_ = !triggered_;
    if (triggered_) return false;

    if ((buttons & kQuestTimerSkipButton) == 0u) {
        holding_ = false;
        hold_start_us_ = 0;
        return false;
    }
    if (!holding_) {
        holding_ = true;
        hold_start_us_ = now_us;
        return false;
    }
    if (now_us < hold_start_us_) {
        hold_start_us_ = now_us;
        return false;
    }
    if (now_us - hold_start_us_ < kQuestTimerSkipHoldUs) return false;

    memory.store32(quest_mgr + kPostQuestCountdown, kQuestTimerSkipTargetFrames);
    triggered_ = true;
    can_skip_ = false;
    return true;
}

} // namespace mhf::mhfu
