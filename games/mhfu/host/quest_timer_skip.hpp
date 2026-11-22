#pragma once

#include "mhf/psp/host.hpp"
#include "psprecomp/guest_memory.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace mhf::mhfu {

struct QuestTimerSkipSymbols {
    std::optional<std::uint32_t> quest_manager; // QuestManager* pointer in guest RAM
    std::optional<std::uint32_t> game_sys;      // GameSys* pointer in guest RAM

    [[nodiscard]] constexpr bool complete() const noexcept {
        return quest_manager.has_value() && game_sys.has_value();
    }
};

// Hold Select for 0.8 seconds during the post-quest 60s countdown (Quest Clear) to fast-forward to return.
inline constexpr std::uint32_t kQuestTimerSkipButton = mhf::psp::buttons::kSelect;
inline constexpr std::uint64_t kQuestTimerSkipHoldUs = 800'000u;
inline constexpr std::uint32_t kQuestTimerSkipMinFrames = 60u;    // > 2s remaining before offering skip
inline constexpr std::uint32_t kQuestTimerSkipTargetFrames = 30u; // 1s left for clean transition

class QuestTimerSkip {
public:
    explicit QuestTimerSkip(QuestTimerSkipSymbols symbols = {}) noexcept;

    // Polls post-quest countdown state and returns true on the tick it fast-forwards the timer.
    bool poll(psprecomp::GuestMemory &memory, std::uint32_t buttons, std::uint64_t now_us, bool enabled);
    void clear() noexcept;

    [[nodiscard]] bool can_skip() const noexcept { return can_skip_; }
    [[nodiscard]] std::string hint() const;

private:
    QuestTimerSkipSymbols symbols_;
    std::uint32_t active_quest_manager_{0};
    std::uint64_t hold_start_us_{0};
    bool holding_{false};
    bool triggered_{false};
    bool can_skip_{false};
};

} // namespace mhf::mhfu
