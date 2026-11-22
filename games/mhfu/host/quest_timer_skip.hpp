#pragma once

#include "mhf/psp/host.hpp"
#include "overlay/song_panel.hpp"
#include "psprecomp/guest_memory.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

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
    [[nodiscard]] float hold_progress() const noexcept { return hold_progress_; }
    // Localized text prompt matching the game's system language (0=JP, 1=EN, 2=FR, 3=ES, 4=DE, 5=IT).
    [[nodiscard]] std::string hint(int language = 1) const;
    // MH-styled PSP [SELECT] pill + return emblem sprite with hold-progress highlight.
    [[nodiscard]] std::vector<mhf::overlay::HintSprite> sprites() const;

private:
    QuestTimerSkipSymbols symbols_;
    std::uint32_t active_quest_manager_{0};
    std::uint64_t hold_start_us_{0};
    float hold_progress_{0.0f};
    bool holding_{false};
    bool triggered_{false};
    bool can_skip_{false};
};

} // namespace mhf::mhfu
