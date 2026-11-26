#pragma once

#include "psprecomp/guest_memory.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace mhf::mhfu {

struct MonsterTrackerSymbols {
    std::optional<std::uint32_t> game_sys;            // GameSys*
    std::optional<std::uint32_t> player_manager;      // PlayerManager*
    std::optional<std::uint32_t> enemy_manager;       // EnemyManager*
    std::optional<std::uint32_t> enemy_species_table; // EnemySpecies[N] in game_task.ovl
    std::optional<std::uint32_t> quest_manager;       // QuestManager*

    [[nodiscard]] constexpr bool complete() const noexcept {
        return game_sys.has_value() && player_manager.has_value() && enemy_manager.has_value() &&
               enemy_species_table.has_value() && quest_manager.has_value();
    }
};

struct TrackedMonster {
    std::uint32_t address{0};
    std::uint8_t species_id{0};
    std::uint16_t area{0};
    std::int16_t hp{0};
    std::int16_t max_hp{0};
    std::array<float, 3> pos{};
    bool capturable{false};
};

// 60 seconds at 30 fps = 1800 frames for quest-start monster reveal.
inline constexpr std::uint32_t kQuestStartTrackFrames = 1800u;

// Scans active large monsters during quests:
// 1. Quest-start tracker: marks large monsters on the map for the first 60 seconds of a quest, then expires naturally.
// 2. Capturable indicator: flags when a capturable large monster in the hunter's area drops below its capture HP threshold.
class MonsterTracker {
public:
    explicit MonsterTracker(MonsterTrackerSymbols symbols = {}) noexcept;

    void poll(psprecomp::GuestMemory &memory, bool quest_start_tracker, bool capturable_indicator);
    void clear() noexcept;

    [[nodiscard]] const std::vector<TrackedMonster> &large_monsters() const noexcept { return monsters_; }
    [[nodiscard]] std::uint16_t player_area() const noexcept { return player_area_; }
    [[nodiscard]] std::uint32_t player_address() const noexcept { return player_address_; }
    [[nodiscard]] std::array<float, 3> player_pos() const noexcept { return player_pos_; }
    [[nodiscard]] bool capturable_in_area() const noexcept { return capturable_in_area_; }
    [[nodiscard]] bool is_enemy_capturable(std::uint32_t enemy_address) const noexcept;
    [[nodiscard]] bool in_active_quest() const noexcept { return in_active_quest_; }

private:
    MonsterTrackerSymbols symbols_;
    std::uint32_t tracked_quest_mgr_{0};
    bool quest_start_applied_{false};
    bool in_active_quest_{false};
    bool capturable_in_area_{false};
    std::uint16_t player_area_{0};
    std::uint32_t player_address_ = 0;
    std::array<float, 3> player_pos_{};
    std::vector<TrackedMonster> monsters_;
};

} // namespace mhf::mhfu
