#include "monster_tracker.hpp"

#include <algorithm>
#include <bit>

namespace mhf::mhfu {
namespace {

// Struct offsets verified identical across EU (ULES01213), US (ULUS10391), JP (ULJM05500) and FUComplete.
constexpr std::uint32_t kGameSysPlayerSlot = 0x28u;
constexpr std::uint32_t kGameSysQuestPhase = 0x2Eu;
constexpr std::uint8_t kQuestPhaseActive = 2u;

constexpr std::uint32_t kPlayerManagerPlayers = 0x0Cu;
constexpr std::uint32_t kPlayerPos = 0x200u;
constexpr std::uint32_t kPlayerArea = 0x29Au;

constexpr std::uint32_t kEnemyManagerSlots = 0x1220u; // Enemy*[20]
constexpr std::size_t kMaxEnemySlots = 20u;

constexpr std::uint32_t kEnemyFlags = 0x04u;
constexpr std::uint32_t kEnemySpeciesId = 0x1E8u;
constexpr std::uint32_t kEnemyPos = 0x200u;
constexpr std::uint32_t kEnemyArea = 0x29Au;
constexpr std::uint32_t kEnemyCurrentHp = 0x2E4u;
constexpr std::uint32_t kEnemyPaintballActive = 0x3D0u; // u8: 1 while paintballed on minimap
constexpr std::uint32_t kEnemyPaintballTimer = 0x3D2u;  // u16: remaining frames (18000 = 10 min, 1800 = 60s)
constexpr std::uint32_t kEnemyMaxHp = 0x41Eu;
constexpr std::uint32_t kEnemyMinSize = kEnemyMaxHp + 2u;

constexpr std::uint32_t kSpeciesStride = 0x1D0u;
constexpr std::uint32_t kSpeciesScaleGroup = 0x14Fu;

bool readable_word(const psprecomp::GuestMemory &memory, std::uint32_t addr) noexcept {
    return (addr & 3u) == 0u && memory.contains(addr, 4u);
}

bool is_large_species(const psprecomp::GuestMemory &memory, std::uint32_t table, std::uint8_t species) noexcept {
    const std::uint32_t entry = table + static_cast<std::uint32_t>(species) * kSpeciesStride + kSpeciesScaleGroup;
    if (!memory.contains(entry, 1u)) return false;
    const auto group = static_cast<std::int8_t>(memory.load8(entry));
    return group >= 1 && group <= 3;
}

// Elder Dragons and siege monsters cannot be captured in MHFU.
bool is_non_capturable_species(std::uint8_t id) noexcept {
    switch (id) {
    case 0x0Bu: // Lao-Shan Lung
    case 0x11u: // Fatalis
    case 0x15u: // Crimson Fatalis
    case 0x16u: // Kirin
    case 0x24u: // Ashen Lao-Shan Lung
    case 0x26u: // Yama Tsukami
    case 0x2Cu: // Kushala Daora
    case 0x2Du: // Shen Gaoren
    case 0x31u: // Lunastra
    case 0x32u: // Teostra
    case 0x34u: // Chameleos
    case 0x3Au: // White Fatalis
    case 0x47u: // Akantor
    case 0x5Au: // Ukanlos
        return true;
    default: return false;
    }
}

} // namespace

MonsterTracker::MonsterTracker(MonsterTrackerSymbols symbols) noexcept : symbols_(symbols) {}

void MonsterTracker::clear() noexcept {
    tracked_quest_mgr_ = 0u;
    quest_start_applied_ = false;
    in_active_quest_ = false;
    capturable_in_area_ = false;
    player_area_ = 0u;
    player_pos_ = {};
    monsters_.clear();
}

void MonsterTracker::poll(psprecomp::GuestMemory &memory, bool quest_start_tracker, bool capturable_indicator) {
    monsters_.clear();
    capturable_in_area_ = false;
    in_active_quest_ = false;

    if (!symbols_.complete() || !readable_word(memory, *symbols_.game_sys) || !readable_word(memory, *symbols_.player_manager) ||
        !readable_word(memory, *symbols_.enemy_manager) || !readable_word(memory, *symbols_.quest_manager)) {
        clear();
        return;
    }

    const std::uint32_t game_sys = memory.load32(*symbols_.game_sys);
    const std::uint32_t player_mgr = memory.load32(*symbols_.player_manager);
    const std::uint32_t enemy_mgr = memory.load32(*symbols_.enemy_manager);
    const std::uint32_t quest_mgr = memory.load32(*symbols_.quest_manager);
    if (game_sys == 0u || player_mgr == 0u || enemy_mgr == 0u || quest_mgr == 0u ||
        !memory.contains(game_sys + kGameSysQuestPhase, 1u) ||
        !memory.contains(player_mgr + kPlayerManagerPlayers, 24u) ||
        !memory.contains(enemy_mgr + kEnemyManagerSlots, static_cast<std::uint32_t>(kMaxEnemySlots) * 4u)) {
        clear();
        return;
    }

    const std::uint8_t phase = memory.load8(game_sys + kGameSysQuestPhase);
    if (phase != kQuestPhaseActive) {
        clear();
        return;
    }
    in_active_quest_ = true;

    if (quest_mgr != tracked_quest_mgr_) {
        tracked_quest_mgr_ = quest_mgr;
        quest_start_applied_ = false;
    }

    const std::uint8_t slot = memory.load8(game_sys + kGameSysPlayerSlot);
    if (slot < 6u) {
        const std::uint32_t player = memory.load32(player_mgr + kPlayerManagerPlayers + static_cast<std::uint32_t>(slot) * 4u);
        if (player != 0u && memory.contains(player + kPlayerPos, 12u) && memory.contains(player + kPlayerArea, 2u)) {
            player_pos_ = {
                std::bit_cast<float>(memory.load32(player + kPlayerPos + 0u)),
                std::bit_cast<float>(memory.load32(player + kPlayerPos + 4u)),
                std::bit_cast<float>(memory.load32(player + kPlayerPos + 8u)),
            };
            player_area_ = memory.load16(player + kPlayerArea);
        }
    }

    bool marked_any = false;
    for (std::size_t i = 0; i < kMaxEnemySlots; ++i) {
        const std::uint32_t enemy = memory.load32(enemy_mgr + kEnemyManagerSlots + static_cast<std::uint32_t>(i) * 4u);
        if (enemy == 0u || (enemy & 3u) != 0u || !memory.contains(enemy, kEnemyMinSize)) continue;
        // Bit 0 of Enemy+0x04 is set while the enemy slot is active/alive (0 = pending destroy/inactive).
        if ((memory.load32(enemy + kEnemyFlags) & 1u) == 0u) continue;

        const std::uint8_t species = memory.load8(enemy + kEnemySpeciesId);
        if (!is_large_species(memory, *symbols_.enemy_species_table, species)) continue;

        const auto hp = static_cast<std::int16_t>(memory.load16(enemy + kEnemyCurrentHp));
        const auto max_hp = static_cast<std::int16_t>(memory.load16(enemy + kEnemyMaxHp));
        const std::uint16_t area = memory.load16(enemy + kEnemyArea);
        if (hp <= 0 || max_hp <= 0) continue;

        const bool capturable = !is_non_capturable_species(species) && (static_cast<int>(hp) * 100 <= static_cast<int>(max_hp) * 25);
        TrackedMonster m;
        m.address = enemy;
        m.species_id = species;
        m.area = area;
        m.hp = hp;
        m.max_hp = max_hp;
        m.pos = {
            std::bit_cast<float>(memory.load32(enemy + kEnemyPos + 0u)),
            std::bit_cast<float>(memory.load32(enemy + kEnemyPos + 4u)),
            std::bit_cast<float>(memory.load32(enemy + kEnemyPos + 8u)),
        };
        m.capturable = capturable;
        monsters_.push_back(m);

        if (capturable_indicator && capturable && area == player_area_) {
            capturable_in_area_ = true;
        }

        // Quest-start tracker: paint large monsters for the first 60 seconds (1800 frames), once per quest.
        if (quest_start_tracker && !quest_start_applied_) {
            const std::uint8_t active = memory.load8(enemy + kEnemyPaintballActive);
            const std::uint16_t cur_timer = memory.load16(enemy + kEnemyPaintballTimer);
            if (active == 0u || cur_timer < kQuestStartTrackFrames) {
                memory.store8(enemy + kEnemyPaintballActive, 1u);
                memory.store16(enemy + kEnemyPaintballTimer, static_cast<std::uint16_t>(kQuestStartTrackFrames));
            }
            marked_any = true;
        }
    }
    if (marked_any) quest_start_applied_ = true;
}

bool MonsterTracker::is_enemy_capturable(std::uint32_t enemy_address) const noexcept {
    if (enemy_address == 0u) return false;
    for (const auto &m : monsters_) {
        if (m.address == enemy_address) return m.capturable;
    }
    return false;
}

} // namespace mhf::mhfu
