#pragma once

#include "psprecomp/guest_memory.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace mhf::mhfu {

// Region-specific guest addresses resolved from games/mhfu/regions/<disc>/symbols.toml.
struct GuildScalingSymbols {
    std::optional<std::uint32_t> quest_manager;       // QuestManager* global pointer
    std::optional<std::uint32_t> player_manager;      // PlayerManager* global pointer
    std::optional<std::uint32_t> enemy_species_table; // EnemySpecies[N] in game_task.ovl (stride 0x1D0)

    [[nodiscard]] constexpr bool valid() const {
        return quest_manager.has_value() && player_manager.has_value() && enemy_species_table.has_value();
    }
};

// Modern player-count multipliers (Wilds/Rise/Sunbreak) anchored at Freedom Unite's 2-player Guild baseline (1.00x).
// Monster attack and defense multipliers stay untouched so each rank keeps its original hit weight and armor curve.
struct GuildScaleFactors {
    float hp = 1.0f;
    float stagger = 1.0f;
};

// True for Guild Hall (Low/High/G-Rank) and Guild Event/Download quest IDs. Village Elder/Nekoht, Treasure, and
// Training School quests return false and always keep their original scaling.
[[nodiscard]] bool is_guild_scalable_quest(std::uint32_t quest_id);

// True for large monster scaling groups (1 standard large, 2 Fatalis/Lao, 3 Akantor/Ukanlos/Shen/Yama).
// Group 0 (small monsters) and out-of-range values return false.
[[nodiscard]] bool is_large_monster_group(std::int8_t scale_group);

// Multipliers for `player_count` active human hunters (clamped to 1..4). Felyne comrades are excluded.
[[nodiscard]] GuildScaleFactors guild_scale_factors(int player_count);

// Scales a positive vanilla stat (HP or part stagger) and clamps to MHFU's signed 16-bit range [1, 32767].
[[nodiscard]] std::int16_t scale_stat_s16(float vanilla_value, float multiplier);

// Rescales an active HP value proportionally when max HP changes, preserving remaining HP percentage without
// reviving a dead monster (<= 0) or exceeding the new maximum.
[[nodiscard]] std::int16_t rescale_current_hp_s16(std::int16_t current_hp, std::int16_t old_max_hp,
                                                  std::int16_t new_max_hp);

// Counts active human hunters across PlayerManager slots 0..3 (excluding Felyne comrade slots 4..5).
// Returns std::nullopt when memory is unmapped/invalid or no human hunter slot is active yet.
[[nodiscard]] std::optional<int> active_human_hunters(const psprecomp::GuestMemory &memory,
                                                      std::uint32_t player_manager_ptr_addr);

// Reads the active quest ID from QuestManager (+0x10). Returns std::nullopt if unmapped or invalid.
[[nodiscard]] std::optional<std::uint32_t> active_quest_id(const psprecomp::GuestMemory &memory,
                                                           std::uint32_t quest_manager_ptr_addr);

// Reads the scaling group (+0x14F) of `em_id` from game_task.ovl's species table (stride 0x1D0).
[[nodiscard]] std::optional<std::int8_t> enemy_scale_group(const psprecomp::GuestMemory &memory,
                                                           std::uint32_t species_table_addr, std::uint8_t em_id);

// Stateful tracker for active Guild quests: intercepts large-monster HP and part-stagger initialization and
// dynamically rescales live monsters if the active human hunter count changes (e.g., mid-quest disconnect) or when
// the option is toggled from the in-game menu.
class GuildQuestScaling {
public:
    static constexpr std::size_t kMaxParts = 9; // game_task.ovl initializes parts 0..8

    explicit GuildQuestScaling(GuildScalingSymbols symbols = {}) : symbols_(symbols) {}

    void set_symbols(GuildScalingSymbols symbols) { symbols_ = symbols; }

    // Called at entry to calc_enemy_hp(Enemy* a0, int16_t a1). Records the pending enemy pointer for the matching
    // on_enemy_hp_computed call in the same function invocation.
    void on_enemy_hp_begin(std::uint32_t enemy_addr);

    // Called right before trunc.w.s in calc_enemy_hp. Returns the float HP value to place in $f0 (scaled when
    // enabled on an eligible Guild quest + large monster, otherwise unchanged). Always tracks eligible large monsters
    // so toggling the setting mid-quest can rescale them safely.
    [[nodiscard]] float on_enemy_hp_computed(const psprecomp::GuestMemory &memory, std::uint8_t em_id,
                                             float vanilla_hp, bool enabled);

    // Called right before storing the part stagger threshold at enemy + part*8 + 0x580. Returns the s16 value to store.
    [[nodiscard]] std::int16_t on_enemy_stagger_store(const psprecomp::GuestMemory &memory, std::uint32_t enemy_addr,
                                                      std::uint8_t em_id, std::uint32_t part_index,
                                                      std::int16_t vanilla_stagger, bool enabled);

    // Periodic host-poll check: clears state when leaving a quest and dynamically rescales tracked live large monsters
    // whenever active human hunter count changes or the toggle switches on/off.
    void sync(psprecomp::GuestMemory &memory, bool enabled);

    void clear();

    [[nodiscard]] std::size_t tracked_enemy_count() const { return enemies_.size(); }

private:
    struct TrackedPart {
        bool valid = false;
        std::int16_t vanilla_stagger = 0;
        std::int16_t applied_stagger = 0;
    };

    struct TrackedEnemy {
        std::uint32_t addr = 0;
        std::uint8_t em_id = 0;
        float vanilla_hp = 0.0f;
        std::int16_t applied_max_hp = 0;
        int applied_players = 2;
        bool has_hp = false;
        std::array<TrackedPart, kMaxParts> parts{};
    };

    TrackedEnemy *find_or_add_enemy(std::uint32_t enemy_addr, std::uint8_t em_id);
    void reset_if_quest_changed(std::uint32_t quest_id);

    GuildScalingSymbols symbols_{};
    std::uint32_t active_quest_id_ = 0;
    std::uint32_t pending_enemy_addr_ = 0;
    std::vector<TrackedEnemy> enemies_;
};

} // namespace mhf::mhfu
