#include "guild_scaling.hpp"

#include <algorithm>
#include <cmath>

namespace mhf::mhfu {
namespace {

// The quest-header pointer and ID were checked in FUComplete; the header signature guards other releases.
constexpr std::uint32_t kQuestManagerHeader = 0x4C;   // pointer to the active quest record
constexpr std::uint32_t kQuestHeaderMagic = 0x47444E32u; // "2NDG"
constexpr std::uint32_t kPlayerManagerPlayers = 0x0C; // Player*[6]: 0..3 human hunters, 4..5 Felyne comrades
constexpr std::uint32_t kHumanPlayerSlots = 4;
constexpr std::uint32_t kPlayerFlags = 0x04;          // u32: bit 0 set while the player entity is active
constexpr std::uint32_t kPlayerFlagActive = 0x01u;

constexpr std::uint32_t kSpeciesStride = 0x1D0;       // 464 bytes per species entry in game_task.ovl
constexpr std::uint32_t kSpeciesScaleGroup = 0x14F;   // s8: 0 small, 1 standard large, 2 Fatalis/Lao, 3 giant bosses

constexpr std::uint32_t kEnemySpeciesId = 0x1E8;      // u8: species ID
constexpr std::uint32_t kEnemyCurrentHp = 0x2E4;      // s16: current HP
constexpr std::uint32_t kEnemyMaxHp = 0x41E;          // s16: maximum HP
constexpr std::uint32_t kEnemyPartBase = 0x57C;       // 9 part records, 8 bytes each (remaining stagger HP at +4)
constexpr std::uint32_t kEnemyPartStride = 0x08;
constexpr std::uint32_t kEnemyPartRemaining = 0x04;   // s16 at +0x580 + part*8, counts down to <= 0 on hits
constexpr std::uint32_t kEnemyTrackedBytes =
    kEnemyPartBase + static_cast<std::uint32_t>(GuildQuestScaling::kMaxParts) * kEnemyPartStride;

constexpr int kVanillaGuildBaselinePlayers = 2;

// Player-count curve anchored at MHFU's 2-player Guild baseline (1.00x):
//   HP:      0.70 / 1.00 / 1.25 / 1.50  (equivalent to 1.00 / 1.43 / 1.79 / 2.14 from 1P)
//   Stagger: 0.75 / 1.00 / 1.25 / 1.50  (slightly firmer solo to keep MHFU's flinch weight; prevents 4P stunlock)
constexpr std::array<GuildScaleFactors, 4> kFactorsByPlayers = {{
    {0.70f, 0.75f}, // 1 player (solo)
    {1.00f, 1.00f}, // 2 players (vanilla Guild baseline)
    {1.25f, 1.25f}, // 3 players
    {1.50f, 1.50f}, // 4 players
}};

bool aligned_u32(std::uint32_t addr) { return (addr & 3u) == 0u; }

std::optional<std::uint32_t> read_quest_id(const psprecomp::GuestMemory &memory, std::uint32_t quest_manager_ptr_addr) {
    if (!aligned_u32(quest_manager_ptr_addr) || !memory.contains(quest_manager_ptr_addr, 4u)) return std::nullopt;
    const std::uint32_t manager = memory.load32(quest_manager_ptr_addr);
    if (manager == 0u || !aligned_u32(manager) || !memory.contains(manager, kQuestManagerHeader + 4u)) {
        return std::nullopt;
    }
    const std::uint32_t header = memory.load32(manager + kQuestManagerHeader);
    if (header == 0u || !aligned_u32(header) || !memory.contains(header, 0x66u) ||
        memory.load32(header + 4u) != kQuestHeaderMagic) return std::nullopt;
    return memory.load16(header + 0x64u);
}

std::optional<std::int8_t> read_scale_group(const psprecomp::GuestMemory &memory, std::uint32_t species_table_addr,
                                            std::uint8_t em_id) {
    const std::uint32_t addr = species_table_addr + static_cast<std::uint32_t>(em_id) * kSpeciesStride + kSpeciesScaleGroup;
    if (!memory.contains(addr, 1u)) return std::nullopt;
    return static_cast<std::int8_t>(memory.load8(addr));
}

} // namespace

bool is_guild_scalable_quest(std::uint32_t quest_id) {
    // Guild Low (1..199, 1001..1299), High (2001..2299), G-Rank (3001..3299), and Event/DLC (60001+).
    // Excludes Village Elder (10001..10599), Nekoht (11001..11299), Treasure (4001..4007), Training (20001..29999).
    return (quest_id >= 1u && quest_id <= 199u) || (quest_id >= 1001u && quest_id <= 1299u) ||
           (quest_id >= 2001u && quest_id <= 2299u) || (quest_id >= 3001u && quest_id <= 3299u) || quest_id >= 60001u;
}

bool is_large_monster_group(std::int8_t scale_group) { return scale_group >= 1 && scale_group <= 3; }

GuildScaleFactors guild_scale_factors(int player_count) {
    return kFactorsByPlayers[static_cast<std::size_t>(std::clamp(player_count, 1, 4) - 1)];
}

std::int16_t scale_stat_s16(float vanilla_value, float multiplier, std::int16_t max_value) {
    if (!std::isfinite(vanilla_value) || !std::isfinite(multiplier) || vanilla_value <= 0.0f || multiplier <= 0.0f ||
        max_value <= 0) {
        return 0;
    }
    const double scaled = std::round(static_cast<double>(vanilla_value) * static_cast<double>(multiplier));
    return static_cast<std::int16_t>(std::clamp(scaled, 1.0, static_cast<double>(max_value)));
}

std::int16_t rescale_current_hp_s16(std::int16_t current, std::int16_t old_max, std::int16_t new_max) {
    if (current <= 0 || old_max <= 0 || new_max <= 0) return current;
    if (old_max == new_max) return std::min(current, new_max);
    const double scaled =
        std::round((static_cast<double>(current) / static_cast<double>(old_max)) * static_cast<double>(new_max));
    return static_cast<std::int16_t>(std::clamp(scaled, 1.0, static_cast<double>(new_max)));
}

std::optional<int> active_human_hunters(const psprecomp::GuestMemory &memory, std::uint32_t player_manager_ptr_addr) {
    if (!aligned_u32(player_manager_ptr_addr) || !memory.contains(player_manager_ptr_addr, 4u)) return std::nullopt;
    const std::uint32_t manager = memory.load32(player_manager_ptr_addr);
    if (manager == 0u || !aligned_u32(manager) ||
        !memory.contains(manager + kPlayerManagerPlayers, kHumanPlayerSlots * 4u)) {
        return std::nullopt;
    }
    int count = 0;
    for (std::uint32_t slot = 0; slot < kHumanPlayerSlots; ++slot) {
        const std::uint32_t player = memory.load32(manager + kPlayerManagerPlayers + slot * 4u);
        if (player == 0u || !aligned_u32(player) || !memory.contains(player, kPlayerFlags + 4u)) continue;
        if ((memory.load32(player + kPlayerFlags) & kPlayerFlagActive) != 0u) ++count;
    }
    return count > 0 ? std::optional<int>{count} : std::nullopt;
}

void GuildQuestScaling::reset_if_quest_changed(std::uint32_t quest_id) {
    if (quest_id != active_quest_id_) {
        enemies_.clear();
        active_quest_id_ = quest_id;
    }
}

std::optional<GuildQuestScaling::SpawnContext>
GuildQuestScaling::resolve_spawn(const psprecomp::GuestMemory &memory, std::uint8_t em_id, bool enabled) {
    if (!symbols_.valid()) return std::nullopt;
    const auto qid = read_quest_id(memory, *symbols_.quest_manager);
    if (!qid || !is_guild_scalable_quest(*qid)) return std::nullopt;
    reset_if_quest_changed(*qid);

    const auto group = read_scale_group(memory, *symbols_.enemy_species_table, em_id);
    if (!group || !is_large_monster_group(*group)) return std::nullopt;

    const auto players = active_human_hunters(memory, *symbols_.player_manager);
    return SpawnContext{(enabled && players) ? *players : kVanillaGuildBaselinePlayers, enabled && players.has_value()};
}

GuildQuestScaling::TrackedEnemy *GuildQuestScaling::find_or_add_enemy(std::uint32_t enemy_addr, std::uint8_t em_id) {
    for (auto &enemy : enemies_) {
        if (enemy.addr == enemy_addr) {
            if (enemy.em_id != em_id) enemy = TrackedEnemy{enemy_addr, em_id};
            return &enemy;
        }
    }
    constexpr std::size_t kMaxTrackedEnemies = 16;
    if (enemies_.size() >= kMaxTrackedEnemies) enemies_.erase(enemies_.begin());
    enemies_.push_back(TrackedEnemy{enemy_addr, em_id});
    return &enemies_.back();
}

float GuildQuestScaling::on_enemy_hp_computed(const psprecomp::GuestMemory &memory, std::uint8_t em_id,
                                              float vanilla_hp, bool enabled) {
    const std::uint32_t enemy_addr = pending_enemy_addr_;
    pending_enemy_addr_ = 0;

    if (!std::isfinite(vanilla_hp) || vanilla_hp <= 0.0f) return vanilla_hp;
    const auto ctx = resolve_spawn(memory, em_id, enabled);
    if (!ctx) return vanilla_hp;

    const std::int16_t scaled = scale_stat_s16(vanilla_hp, guild_scale_factors(ctx->target_players).hp, kMaxEnemyHpS16);
    if (scaled <= 0) return vanilla_hp;

    if (enemy_addr != 0u && aligned_u32(enemy_addr) && memory.contains(enemy_addr, kEnemyMaxHp + 2u)) {
        TrackedEnemy *tracked = find_or_add_enemy(enemy_addr, em_id);
        tracked->vanilla_hp = vanilla_hp;
        tracked->applied_max_hp = scaled;
        tracked->applied_players = ctx->target_players;
        tracked->has_hp = true;
    }
    return ctx->apply ? static_cast<float>(scaled) : vanilla_hp;
}

std::int16_t GuildQuestScaling::on_enemy_stagger_store(const psprecomp::GuestMemory &memory, std::uint32_t enemy_addr,
                                                       std::uint8_t em_id, std::uint32_t part_index,
                                                       std::int16_t vanilla_stagger, bool enabled) {
    if (vanilla_stagger <= 0 || part_index >= kMaxParts) return vanilla_stagger;
    const auto ctx = resolve_spawn(memory, em_id, enabled);
    if (!ctx) return vanilla_stagger;

    const std::int16_t scaled =
        scale_stat_s16(static_cast<float>(vanilla_stagger), guild_scale_factors(ctx->target_players).stagger,
                       kMaxPartStaggerS16);
    if (scaled <= 0) return vanilla_stagger;

    if (enemy_addr != 0u && aligned_u32(enemy_addr) && memory.contains(enemy_addr, kEnemyTrackedBytes)) {
        TrackedEnemy *tracked = find_or_add_enemy(enemy_addr, em_id);
        tracked->parts[part_index] = TrackedPart{true, vanilla_stagger, scaled};
        if (!tracked->has_hp) tracked->applied_players = ctx->target_players;
    }
    return ctx->apply ? scaled : vanilla_stagger;
}

void GuildQuestScaling::sync(psprecomp::GuestMemory &memory, bool enabled) {
    if (!symbols_.valid()) return;
    const auto qid = read_quest_id(memory, *symbols_.quest_manager);
    if (!qid || !is_guild_scalable_quest(*qid)) {
        clear();
        return;
    }
    reset_if_quest_changed(*qid);
    if (enemies_.empty()) return;

    const auto players = active_human_hunters(memory, *symbols_.player_manager);
    if (!players) return;

    const int target_players = enabled ? *players : kVanillaGuildBaselinePlayers;
    const GuildScaleFactors factors = guild_scale_factors(target_players);

    std::size_t write = 0;
    for (std::size_t i = 0; i < enemies_.size(); ++i) {
        TrackedEnemy &enemy = enemies_[i];
        if (!memory.contains(enemy.addr, kEnemyTrackedBytes) || memory.load8(enemy.addr + kEnemySpeciesId) != enemy.em_id) {
            continue;
        }
        if (enemy.has_hp) {
            const auto current_max = static_cast<std::int16_t>(memory.load16(enemy.addr + kEnemyMaxHp));
            const auto current_hp = static_cast<std::int16_t>(memory.load16(enemy.addr + kEnemyCurrentHp));
            if (current_max != enemy.applied_max_hp || current_hp <= 0) continue;

            if (enemy.applied_players != target_players) {
                const std::int16_t new_max = scale_stat_s16(enemy.vanilla_hp, factors.hp, kMaxEnemyHpS16);
                if (new_max > 0) {
                    const std::int16_t new_hp = rescale_current_hp_s16(current_hp, current_max, new_max);
                    memory.store16(enemy.addr + kEnemyMaxHp, static_cast<std::uint16_t>(new_max));
                    memory.store16(enemy.addr + kEnemyCurrentHp, static_cast<std::uint16_t>(new_hp));
                    enemy.applied_max_hp = new_max;
                }
            } else if (current_hp > current_max) {
                // Guard against carried-over Elder Dragon HP exceeding a scaled maximum right after spawn.
                memory.store16(enemy.addr + kEnemyCurrentHp, static_cast<std::uint16_t>(current_max));
            }
        }
        if (enemy.applied_players != target_players) {
            // Part stagger at +0x580 + part*8 counts down to <= 0 on hits (and game_task clamps it at 4000),
            // so proportional rescaling preserves damage already dealt toward the next flinch.
            for (std::size_t p = 0; p < kMaxParts; ++p) {
                TrackedPart &part = enemy.parts[p];
                if (!part.valid || part.applied_stagger <= 0) continue;
                const std::uint32_t part_addr =
                    enemy.addr + kEnemyPartBase + static_cast<std::uint32_t>(p) * kEnemyPartStride;
                if (memory.load8(part_addr) == 0u) continue;
                const auto remaining = static_cast<std::int16_t>(memory.load16(part_addr + kEnemyPartRemaining));
                if (remaining <= 0) continue;
                const std::int16_t new_stagger =
                    scale_stat_s16(static_cast<float>(part.vanilla_stagger), factors.stagger, kMaxPartStaggerS16);
                if (new_stagger > 0) {
                    const std::int16_t rescaled = rescale_current_hp_s16(remaining, part.applied_stagger, new_stagger);
                    memory.store16(part_addr + kEnemyPartRemaining, static_cast<std::uint16_t>(rescaled));
                    part.applied_stagger = new_stagger;
                }
            }
            enemy.applied_players = target_players;
        }
        if (write != i) enemies_[write] = enemy;
        ++write;
    }
    enemies_.resize(write);
}

void GuildQuestScaling::clear() {
    active_quest_id_ = 0;
    pending_enemy_addr_ = 0;
    enemies_.clear();
}

} // namespace mhf::mhfu
