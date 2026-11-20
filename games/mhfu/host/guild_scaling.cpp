#include "guild_scaling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mhf::mhfu {
namespace {

// Struct offsets verified across ULES01213, ULUS10391, ULJM05500, and ULJM05500-FUC.
constexpr std::uint32_t kQuestManagerQuestId = 0x10;      // u32: active quest ID
constexpr std::uint32_t kPlayerManagerPlayers = 0x0C;     // Player*[6]: 0..3 human hunters, 4..5 Felyne comrades
constexpr std::uint32_t kHumanPlayerSlots = 4;
constexpr std::uint32_t kPlayerFlags = 0x04;              // u32: bit 0 set while the player entity is active
constexpr std::uint32_t kPlayerFlagActive = 0x01u;

constexpr std::uint32_t kSpeciesStride = 0x1D0;           // 464 bytes per species entry in game_task.ovl
constexpr std::uint32_t kSpeciesScaleGroup = 0x14F;       // s8: 0 small, 1 standard large, 2 Fatalis/Lao, 3 giant bosses

constexpr std::uint32_t kEnemySpeciesId = 0x1E8;          // u8: species ID
constexpr std::uint32_t kEnemyCurrentHp = 0x2E4;          // s16: current HP
constexpr std::uint32_t kEnemyMaxHp = 0x41E;              // s16: maximum HP
constexpr std::uint32_t kEnemyPartBase = 0x57C;           // 9 part records, 8 bytes each
constexpr std::uint32_t kEnemyPartStride = 0x08;
constexpr std::uint32_t kEnemyPartThreshold = 0x04;       // s16 at +0x580 + part*8

constexpr int kVanillaGuildBaselinePlayers = 2;
constexpr std::int16_t kMaxHpS16 = std::numeric_limits<std::int16_t>::max(); // 32767: all game_task HP loads are signed lh

// Player-count curve anchored at MHFU's 2-player Guild baseline (1.00x):
//   HP:      0.70 / 1.00 / 1.25 / 1.50  (equivalent to 1.00 / 1.43 / 1.79 / 2.14 from 1P)
//   Stagger: 0.75 / 1.00 / 1.25 / 1.50  (slightly firmer solo to keep MHFU's flinch weight; prevents 4P stunlock)
constexpr std::array<GuildScaleFactors, 4> kFactorsByPlayers = {{
    {0.70f, 0.75f}, // 1 player (solo)
    {1.00f, 1.00f}, // 2 players (vanilla Guild baseline)
    {1.25f, 1.25f}, // 3 players
    {1.50f, 1.50f}, // 4 players
}};

bool ram_contains(const psprecomp::GuestMemory &memory, std::uint32_t address, std::uint32_t size) {
    const std::uint32_t physical = psprecomp::GuestMemory::canonical(address);
    if (physical < psprecomp::GuestMemory::kPhysicalBase) return false;
    const std::uint32_t offset = physical - psprecomp::GuestMemory::kPhysicalBase;
    return offset <= memory.size() && size <= memory.size() - offset;
}

bool aligned_u32(std::uint32_t addr) { return (addr & 3u) == 0u; }

} // namespace

bool is_guild_scalable_quest(std::uint32_t quest_id) {
    // Guild Low Rank: 1..199 (1*-2*), 1001..1299 (3*-5*)
    // Guild High Rank: 2001..2299 (6*-8*)
    // Guild G-Rank: 3001..3299 (G1*-G3*)
    // Guild Event / Download Quests: 60001+
    // Explicitly excluded: Village Elder (10001..10599), Nekoht (11001..11299), Treasure (4001..4007), Training (20001..29999).
    if (quest_id >= 1u && quest_id <= 199u) return true;
    if (quest_id >= 1001u && quest_id <= 1299u) return true;
    if (quest_id >= 2001u && quest_id <= 2299u) return true;
    if (quest_id >= 3001u && quest_id <= 3299u) return true;
    if (quest_id >= 60001u) return true;
    return false;
}

bool is_large_monster_group(std::int8_t scale_group) {
    return scale_group >= 1 && scale_group <= 3;
}

GuildScaleFactors guild_scale_factors(int player_count) {
    const int clamped = std::clamp(player_count, 1, 4);
    return kFactorsByPlayers[static_cast<std::size_t>(clamped - 1)];
}

std::int16_t scale_stat_s16(float vanilla_value, float multiplier) {
    if (!std::isfinite(vanilla_value) || !std::isfinite(multiplier) || vanilla_value <= 0.0f || multiplier <= 0.0f) {
        return 0;
    }
    const double scaled = std::round(static_cast<double>(vanilla_value) * static_cast<double>(multiplier));
    if (scaled < 1.0) return 1;
    if (scaled > static_cast<double>(kMaxHpS16)) return kMaxHpS16;
    return static_cast<std::int16_t>(scaled);
}

std::int16_t rescale_current_hp_s16(std::int16_t current_hp, std::int16_t old_max_hp, std::int16_t new_max_hp) {
    if (current_hp <= 0) return current_hp; // never revive a slain monster
    if (old_max_hp <= 0 || new_max_hp <= 0) return current_hp;
    if (old_max_hp == new_max_hp) return std::min(current_hp, new_max_hp);
    const double ratio = static_cast<double>(current_hp) / static_cast<double>(old_max_hp);
    const double scaled = std::round(ratio * static_cast<double>(new_max_hp));
    const double clamped = std::clamp(scaled, 1.0, static_cast<double>(new_max_hp));
    return static_cast<std::int16_t>(clamped);
}

std::optional<int> active_human_hunters(const psprecomp::GuestMemory &memory, std::uint32_t player_manager_ptr_addr) {
    if (!aligned_u32(player_manager_ptr_addr) || !ram_contains(memory, player_manager_ptr_addr, 4u)) return std::nullopt;
    const std::uint32_t manager = memory.load32(player_manager_ptr_addr);
    if (manager == 0u || !aligned_u32(manager) ||
        !ram_contains(memory, manager + kPlayerManagerPlayers, kHumanPlayerSlots * 4u)) {
        return std::nullopt;
    }
    int count = 0;
    for (std::uint32_t slot = 0; slot < kHumanPlayerSlots; ++slot) {
        const std::uint32_t player = memory.load32(manager + kPlayerManagerPlayers + slot * 4u);
        if (player == 0u || !aligned_u32(player) || !ram_contains(memory, player, kPlayerFlags + 4u)) continue;
        if ((memory.load32(player + kPlayerFlags) & kPlayerFlagActive) != 0u) ++count;
    }
    if (count <= 0) return std::nullopt;
    return std::clamp(count, 1, static_cast<int>(kHumanPlayerSlots));
}

std::optional<std::uint32_t> active_quest_id(const psprecomp::GuestMemory &memory,
                                             std::uint32_t quest_manager_ptr_addr) {
    if (!aligned_u32(quest_manager_ptr_addr) || !ram_contains(memory, quest_manager_ptr_addr, 4u)) return std::nullopt;
    const std::uint32_t manager = memory.load32(quest_manager_ptr_addr);
    if (manager == 0u || !aligned_u32(manager) || !ram_contains(memory, manager, kQuestManagerQuestId + 4u)) {
        return std::nullopt;
    }
    return memory.load32(manager + kQuestManagerQuestId);
}

std::optional<std::int8_t> enemy_scale_group(const psprecomp::GuestMemory &memory, std::uint32_t species_table_addr,
                                             std::uint8_t em_id) {
    const std::uint32_t addr = species_table_addr + static_cast<std::uint32_t>(em_id) * kSpeciesStride + kSpeciesScaleGroup;
    if (!ram_contains(memory, addr, 1u)) return std::nullopt;
    return static_cast<std::int8_t>(memory.load8(addr));
}

void GuildQuestScaling::on_enemy_hp_begin(std::uint32_t enemy_addr) {
    pending_enemy_addr_ = enemy_addr;
}

void GuildQuestScaling::reset_if_quest_changed(std::uint32_t quest_id) {
    if (quest_id != active_quest_id_) {
        enemies_.clear();
        active_quest_id_ = quest_id;
    }
}

GuildQuestScaling::TrackedEnemy *GuildQuestScaling::find_or_add_enemy(std::uint32_t enemy_addr, std::uint8_t em_id) {
    for (auto &enemy : enemies_) {
        if (enemy.addr == enemy_addr) {
            if (enemy.em_id != em_id) {
                enemy = TrackedEnemy{};
                enemy.addr = enemy_addr;
                enemy.em_id = em_id;
            }
            return &enemy;
        }
    }
    // Cap tracked enemies conservatively; MHFU only hosts a handful of large monsters per quest.
    constexpr std::size_t kMaxTrackedEnemies = 16;
    if (enemies_.size() >= kMaxTrackedEnemies) enemies_.erase(enemies_.begin());
    TrackedEnemy fresh;
    fresh.addr = enemy_addr;
    fresh.em_id = em_id;
    enemies_.push_back(fresh);
    return &enemies_.back();
}

float GuildQuestScaling::on_enemy_hp_computed(const psprecomp::GuestMemory &memory, std::uint8_t em_id,
                                              float vanilla_hp, bool enabled) {
    const std::uint32_t enemy_addr = pending_enemy_addr_;
    pending_enemy_addr_ = 0;

    if (!symbols_.valid() || !std::isfinite(vanilla_hp) || vanilla_hp <= 0.0f) return vanilla_hp;

    const auto qid = active_quest_id(memory, *symbols_.quest_manager);
    if (!qid || !is_guild_scalable_quest(*qid)) return vanilla_hp;
    reset_if_quest_changed(*qid);

    const auto group = enemy_scale_group(memory, *symbols_.enemy_species_table, em_id);
    if (!group || !is_large_monster_group(*group)) return vanilla_hp;

    const auto players = active_human_hunters(memory, *symbols_.player_manager);
    const int target_players = (enabled && players) ? *players : kVanillaGuildBaselinePlayers;
    const float mult = guild_scale_factors(target_players).hp;
    const std::int16_t scaled = scale_stat_s16(vanilla_hp, mult);
    if (scaled <= 0) return vanilla_hp;

    if (enemy_addr != 0u && aligned_u32(enemy_addr) && ram_contains(memory, enemy_addr, kEnemyMaxHp + 2u)) {
        TrackedEnemy *tracked = find_or_add_enemy(enemy_addr, em_id);
        tracked->vanilla_hp = vanilla_hp;
        tracked->applied_max_hp = scaled;
        tracked->applied_players = target_players;
        tracked->has_hp = true;
    }

    return enabled && players ? static_cast<float>(scaled) : vanilla_hp;
}

std::int16_t GuildQuestScaling::on_enemy_stagger_store(const psprecomp::GuestMemory &memory, std::uint32_t enemy_addr,
                                                       std::uint8_t em_id, std::uint32_t part_index,
                                                       std::int16_t vanilla_stagger, bool enabled) {
    if (!symbols_.valid() || vanilla_stagger <= 0 || part_index >= kMaxParts) return vanilla_stagger;

    const auto qid = active_quest_id(memory, *symbols_.quest_manager);
    if (!qid || !is_guild_scalable_quest(*qid)) return vanilla_stagger;
    reset_if_quest_changed(*qid);

    const auto group = enemy_scale_group(memory, *symbols_.enemy_species_table, em_id);
    if (!group || !is_large_monster_group(*group)) return vanilla_stagger;

    const auto players = active_human_hunters(memory, *symbols_.player_manager);
    const int target_players = (enabled && players) ? *players : kVanillaGuildBaselinePlayers;
    const float mult = guild_scale_factors(target_players).stagger;
    const std::int16_t scaled = scale_stat_s16(static_cast<float>(vanilla_stagger), mult);
    if (scaled <= 0) return vanilla_stagger;

    if (enemy_addr != 0u && aligned_u32(enemy_addr) &&
        ram_contains(memory, enemy_addr, kEnemyPartBase + static_cast<std::uint32_t>(kMaxParts) * kEnemyPartStride)) {
        TrackedEnemy *tracked = find_or_add_enemy(enemy_addr, em_id);
        tracked->parts[part_index] = TrackedPart{true, vanilla_stagger, scaled};
        if (!tracked->has_hp) tracked->applied_players = target_players;
    }

    return enabled && players ? scaled : vanilla_stagger;
}

void GuildQuestScaling::sync(psprecomp::GuestMemory &memory, bool enabled) {
    if (!symbols_.valid()) return;

    const auto qid = active_quest_id(memory, *symbols_.quest_manager);
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
        const std::uint32_t needed = kEnemyPartBase + static_cast<std::uint32_t>(kMaxParts) * kEnemyPartStride;
        if (!ram_contains(memory, enemy.addr, needed)) continue;
        if (memory.load8(enemy.addr + kEnemySpeciesId) != enemy.em_id) continue;

        if (enemy.has_hp) {
            const auto current_max = static_cast<std::int16_t>(memory.load16(enemy.addr + kEnemyMaxHp));
            const auto current_hp = static_cast<std::int16_t>(memory.load16(enemy.addr + kEnemyCurrentHp));
            if (current_max != enemy.applied_max_hp || current_hp <= 0) continue; // enemy despawned, recycled, or dead

            if (enemy.applied_players != target_players) {
                const std::int16_t new_max = scale_stat_s16(enemy.vanilla_hp, factors.hp);
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
            for (std::size_t p = 0; p < kMaxParts; ++p) {
                TrackedPart &part = enemy.parts[p];
                if (!part.valid) continue;
                const std::uint32_t part_addr =
                    enemy.addr + kEnemyPartBase + static_cast<std::uint32_t>(p) * kEnemyPartStride;
                if (memory.load8(part_addr) == 0u) continue; // inactive part slot
                const auto current_stagger = static_cast<std::int16_t>(memory.load16(part_addr + kEnemyPartThreshold));
                if (current_stagger <= 0) continue;
                const std::int16_t new_stagger =
                    scale_stat_s16(static_cast<float>(part.vanilla_stagger), factors.stagger);
                if (new_stagger > 0 && part.applied_stagger > 0) {
                    const std::int16_t rescaled =
                        rescale_current_hp_s16(current_stagger, part.applied_stagger, new_stagger);
                    memory.store16(part_addr + kEnemyPartThreshold, static_cast<std::uint16_t>(rescaled));
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
