#pragma once

// Quest auto-clear trigger (--quest-autoclear): holds L + R + Select for two seconds during an active quest to mark
// the quest's required objectives complete, letting the game's own objective check run the Quest Clear sequence,
// countdown and rewards. Region addresses come from mhfu_region_data.hpp via QuestAutoclearSymbols so the logic and
// its unit tests build without game data.

#include "mhf/psp/host.hpp"

#include "psprecomp/guest_memory.hpp"

#include <cstdint>
#include <optional>

namespace mhf::mhfu {

// Classic MHFU CWCheat activator (L + R + Select) held continuously to avoid accidental triggers.
inline constexpr std::uint32_t kQuestAutoclearCombo =
    psp::buttons::kLTrigger | psp::buttons::kRTrigger | psp::buttons::kSelect;
inline constexpr std::uint64_t kQuestAutoclearHoldUs = 2'000'000u;

struct QuestAutoclearSymbols {
    std::optional<std::uint32_t> quest_manager;
    std::optional<std::uint32_t> game_sys;

    [[nodiscard]] constexpr bool complete() const noexcept {
        return quest_manager.has_value() && game_sys.has_value();
    }
};

class QuestAutoclear {
public:
    explicit QuestAutoclear(QuestAutoclearSymbols symbols) noexcept;

    // Resets hold timing and the per-quest latch (called when outside the quest overlay or leaving a quest).
    void clear() noexcept;

    // Polls button state during a quest. Returns true on the single poll that marks the active quest's objectives
    // complete; holds a per-quest latch afterwards so keeping the buttons pressed never repeats the write.
    [[nodiscard]] bool poll(psprecomp::GuestMemory &memory, std::uint32_t buttons, std::uint64_t now_us, bool enabled);

private:
    QuestAutoclearSymbols symbols_{};
    std::uint32_t active_quest_manager_ = 0;
    std::uint64_t hold_start_us_ = 0;
    bool holding_ = false;
    bool triggered_ = false;
};

} // namespace mhf::mhfu
