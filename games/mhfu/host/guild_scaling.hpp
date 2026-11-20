#pragma once

#include "psprecomp/guest_memory.hpp"

#include <array>
#include <cstdint>
#include <limits>
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

inline constexpr std::int16_t kMaxEnemyHpS16 = std::numeric_limits<std::int16_t>::max(); // 32767 (signed lh in game_task)
inline constexpr std::int16_t kMaxPartStaggerS16 = 4000;                                 // game_task clamps part HP at 4000

// True for Guild Hall (Low/High/G-Rank) and Guild Event/Download quest IDs. Village, Treasure, and Training return false.
[[nodiscard]] bool is_guild_scalable_quest(std::uint32_t quest_id);

// True for large monster scaling groups (1 standard large, 2 Fatalis/Lao, 3 Akantor/Ukanlos/Shen/Yama).
[[nodiscard]] bool is_large_monster_group(std::int8_t scale_group);

// Multipliers for `player_count` active human hunters (clamped to 1..4). Felyne comrades are excluded.
[[nodiscard]] GuildScaleFactors guild_scale_factors(int player_count);

// Scales a positive vanilla stat (HP or part stagger) and clamps to [1, max_value].
[[nodiscard]] std::int16_t scale_stat_s16(float vanilla_value, float multiplier,
                                          std::int16_t max_value = kMaxEnemyHpS16);

// Rescales a countdown value (current HP or remaining part stagger HP) proportionally when its cap changes,
// preserving remaining percentage without reviving a dead monster / depleted part (<= 0).
[[nodiscard]] std::int16_t rescale_current_hp_s16(std::int16_t current, std::int16_t old_max, std::int16_t new_max);

// Counts active human hunters across PlayerManager slots 0..3 (excluding Felyne comrade slots 4..5).
[[nodiscard]] std::optional<int> active_human_hunters(const psprecomp::GuestMemory &memory,
                                                      std::uint32_t player_manager_ptr_addr);

// Stateful tracker for active Guild quests: intercepts large-monster HP and part-stagger initialization and
// dynamically rescales live monsters if the active hunter count changes (e.g., mid-quest disconnect) or when toggled.
class GuildQuestScaling {
public:
    static constexpr std::size_t kMaxParts = 9; // game_task.ovl initializes parts 0..8

    explicit GuildQuestScaling(GuildScalingSymbols symbols = {}) : symbols_(symbols) {}

    void on_enemy_hp_begin(std::uint32_t enemy_addr) { pending_enemy_addr_ = enemy_addr; }

    [[nodiscard]] float on_enemy_hp_computed(const psprecomp::GuestMemory &memory, std::uint8_t em_id,
                                             float vanilla_hp, bool enabled);

    [[nodiscard]] std::int16_t on_enemy_stagger_store(const psprecomp::GuestMemory &memory, std::uint32_t enemy_addr,
                                                      std::uint8_t em_id, std::uint32_t part_index,
                                                      std::int16_t vanilla_stagger, bool enabled);

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

    struct SpawnContext {
        int target_players = 2;
        bool apply = false;
    };

    std::optional<SpawnContext> resolve_spawn(const psprecomp::GuestMemory &memory, std::uint8_t em_id, bool enabled);
    TrackedEnemy *find_or_add_enemy(std::uint32_t enemy_addr, std::uint8_t em_id);
    void reset_if_quest_changed(std::uint32_t quest_id);

    GuildScalingSymbols symbols_{};
    std::uint32_t active_quest_id_ = 0;
    std::uint32_t pending_enemy_addr_ = 0;
    std::vector<TrackedEnemy> enemies_;
};

} // namespace mhf::mhfu
