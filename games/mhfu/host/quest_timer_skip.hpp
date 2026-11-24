#pragma once

#include "mhf/psp/host.hpp"
#include "overlay/painter.hpp"
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
    [[nodiscard]] float hold_progress() const noexcept { return hold_progress_; }
    [[nodiscard]] std::uint32_t remaining_seconds() const noexcept { return (remaining_frames_ + 29u) / 30u; }
    // Localized label matching the game's system language (0=JP, 1=EN, 2=FR, 3=ES, 4=DE, 5=IT).
    [[nodiscard]] std::string label(int language = 1) const;
    // Renders the top-center MHFU post-quest countdown (animated 5s hourglass, live timer, radial SELECT ring, banner).
    void render_card(mhf::overlay::Painter &painter, int language = 1) const;

private:
    QuestTimerSkipSymbols symbols_;
    std::uint32_t active_quest_manager_{0};
    std::uint32_t remaining_frames_{0};
    std::uint64_t hold_start_us_{0};
    float hold_progress_{0.0f};
    bool holding_{false};
    bool triggered_{false};
    bool can_skip_{false};
};

} // namespace mhf::mhfu
