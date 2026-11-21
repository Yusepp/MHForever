#include "quest_autoclear.hpp"

namespace mhf::mhfu {
namespace {

// Struct offsets verified identical across EU (ULES01213), US (ULUS10391), JP (ULJM05500) and FUComplete.
// QuestManager: +0x00 s8 state (0 active, 1 clear, 2 fail, 3 return), +0x10 u32 remaining time, +0x67C u8 required
// objectives (1 or 2), +0x728 and +0x730 objective slots (8 bytes each: u32 flags, u16 target id, u16 remaining).
constexpr std::uint32_t kQuestState = 0x00u;
constexpr std::uint32_t kQuestTime = 0x10u;
constexpr std::uint32_t kQuestRequiredObjectives = 0x67Cu;
constexpr std::uint32_t kQuestObjectives = 0x728u;
constexpr std::uint32_t kObjectiveStride = 0x08u;
constexpr std::uint32_t kObjectiveFlags = 0x00u;
constexpr std::uint32_t kObjectiveRemaining = 0x06u;
constexpr std::uint32_t kMaxObjectives = 2u;
constexpr std::uint32_t kQuestManagerMinSize = kQuestObjectives + kMaxObjectives * kObjectiveStride;

constexpr std::uint32_t kObjectiveActive = 0x10u;
constexpr std::uint32_t kObjectiveCleared = 0x20u;
constexpr std::uint32_t kObjectiveFailed = 0x40u;

// GameSys +0x2E: u8 quest phase (2 while a quest is actively running; transitions to 3 on Quest Clear).
constexpr std::uint32_t kGameSysQuestPhase = 0x2Eu;
constexpr std::uint8_t kQuestPhaseActive = 2u;

bool readable_word_address(const psprecomp::GuestMemory &memory, std::uint32_t address) noexcept {
    return (address & 3u) == 0u && memory.contains(address, 4u);
}

} // namespace

QuestAutoclear::QuestAutoclear(QuestAutoclearSymbols symbols) noexcept : symbols_(symbols) {}

void QuestAutoclear::clear() noexcept {
    active_quest_manager_ = 0;
    hold_start_us_ = 0;
    holding_ = false;
    triggered_ = false;
}

bool QuestAutoclear::poll(psprecomp::GuestMemory &memory, std::uint32_t buttons, std::uint64_t now_us, bool enabled) {
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

    const std::uint32_t remaining_time = memory.load32(quest_mgr + kQuestTime);
    const auto state = static_cast<std::int8_t>(memory.load8(quest_mgr + kQuestState));
    const std::uint8_t phase = memory.load8(game_sys + kGameSysQuestPhase);
    const std::uint8_t required = memory.load8(quest_mgr + kQuestRequiredObjectives);
    if (remaining_time == 0u || state != 0 || phase != kQuestPhaseActive || required == 0u || required > kMaxObjectives) {
        clear();
        return false;
    }

    if (quest_mgr != active_quest_manager_) {
        active_quest_manager_ = quest_mgr;
        hold_start_us_ = 0;
        holding_ = false;
        triggered_ = false;
    }
    if (triggered_) return false;

    if ((buttons & kQuestAutoclearCombo) != kQuestAutoclearCombo) {
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
    if (now_us - hold_start_us_ < kQuestAutoclearHoldUs) return false;

    // Require every required objective to be active and not already failed before writing anything.
    for (std::uint32_t i = 0; i < required; ++i) {
        const std::uint32_t flags = memory.load32(quest_mgr + kQuestObjectives + i * kObjectiveStride + kObjectiveFlags);
        if ((flags & kObjectiveActive) == 0u || (flags & kObjectiveFailed) != 0u) return false;
    }

    for (std::uint32_t i = 0; i < required; ++i) {
        const std::uint32_t slot = quest_mgr + kQuestObjectives + i * kObjectiveStride;
        const std::uint32_t flags = memory.load32(slot + kObjectiveFlags);
        memory.store32(slot + kObjectiveFlags, flags | kObjectiveCleared);
        memory.store16(slot + kObjectiveRemaining, 0u);
    }
    triggered_ = true;
    return true;
}

} // namespace mhf::mhfu
