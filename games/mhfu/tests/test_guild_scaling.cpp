// Synthetic tests for dynamic Guild quest scaling (no game data required).

#include "guild_scaling.hpp"

#include "check.hpp"

#include <cmath>
#include <cstdint>

namespace {

using mhf::mhfu::GuildQuestScaling;
using mhf::mhfu::GuildScalingSymbols;
using mhf::mhfu::active_human_hunters;
using mhf::mhfu::guild_scale_factors;
using mhf::mhfu::is_guild_scalable_quest;
using mhf::mhfu::is_large_monster_group;
using mhf::mhfu::kMaxPartStaggerS16;
using mhf::mhfu::rescale_current_hp_s16;
using mhf::mhfu::scale_stat_s16;

constexpr std::uint32_t kQuestMgrPtr = 0x08900000u;
constexpr std::uint32_t kQuestMgr = 0x08901000u;
constexpr std::uint32_t kPlayerMgrPtr = 0x08900004u;
constexpr std::uint32_t kPlayerMgr = 0x08902000u;
constexpr std::uint32_t kSpeciesTable = 0x09B00000u;
constexpr std::uint32_t kEnemyBase = 0x09000000u;
constexpr std::uint32_t kPlayerBase = 0x08910000u;

constexpr std::uint32_t kSpeciesStride = 0x1D0u;
constexpr std::uint32_t kSpeciesScaleGroup = 0x14Fu;
constexpr std::uint32_t kEnemySpeciesId = 0x1E8u;
constexpr std::uint32_t kEnemyCurrentHp = 0x2E4u;
constexpr std::uint32_t kEnemyMaxHp = 0x41Eu;
constexpr std::uint32_t kEnemyPartBase = 0x57Cu;
constexpr std::uint32_t kEnemyPartStride = 0x08u;
constexpr std::uint32_t kEnemyPartThreshold = 0x04u;

GuildScalingSymbols make_symbols() { return GuildScalingSymbols{kQuestMgrPtr, kPlayerMgrPtr, kSpeciesTable}; }

void setup_quest(psprecomp::GuestMemory &mem, std::uint32_t qid) {
    mem.store32(kQuestMgrPtr, kQuestMgr);
    constexpr std::uint32_t header = 0x08903000u;
    mem.store32(kQuestMgr + 0x4Cu, header);
    mem.store32(header + 4u, 0x47444E32u);
    mem.store16(header + 0x64u, static_cast<std::uint16_t>(qid));
}

void setup_players(psprecomp::GuestMemory &mem, int human_count, bool add_comrades = true) {
    mem.store32(kPlayerMgrPtr, kPlayerMgr);
    for (int slot = 0; slot < 6; ++slot) {
        const std::uint32_t player = kPlayerBase + static_cast<std::uint32_t>(slot) * 0x100u;
        const bool active = (slot < 4 && slot < human_count) || (slot >= 4 && add_comrades);
        mem.store32(kPlayerMgr + 0x0Cu + static_cast<std::uint32_t>(slot) * 4u, active ? player : 0u);
        mem.store32(player + 0x04u, active ? 1u : 0u);
    }
}

void set_species_group(psprecomp::GuestMemory &mem, std::uint8_t em_id, std::int8_t group) {
    mem.store8(kSpeciesTable + static_cast<std::uint32_t>(em_id) * kSpeciesStride + kSpeciesScaleGroup,
               static_cast<std::uint8_t>(group));
}

void setup_enemy(psprecomp::GuestMemory &mem, std::uint32_t addr, std::uint8_t em_id, std::int16_t max_hp,
                 std::int16_t cur_hp) {
    mem.store8(addr + kEnemySpeciesId, em_id);
    mem.store16(addr + kEnemyMaxHp, static_cast<std::uint16_t>(max_hp));
    mem.store16(addr + kEnemyCurrentHp, static_cast<std::uint16_t>(cur_hp));
}

void setup_part(psprecomp::GuestMemory &mem, std::uint32_t enemy_addr, std::uint32_t part, std::int16_t stagger) {
    const std::uint32_t p = enemy_addr + kEnemyPartBase + part * kEnemyPartStride;
    mem.store8(p, 1u);
    mem.store16(p + kEnemyPartThreshold, static_cast<std::uint16_t>(stagger));
}

void test_quest_classification() {
    // Guild Hall Low / High / G-Rank and Event/DLC
    MHF_CHECK(is_guild_scalable_quest(1));
    MHF_CHECK(is_guild_scalable_quest(101));
    MHF_CHECK(is_guild_scalable_quest(1001));
    MHF_CHECK(is_guild_scalable_quest(1220));
    MHF_CHECK(is_guild_scalable_quest(2001));
    MHF_CHECK(is_guild_scalable_quest(2236));
    MHF_CHECK(is_guild_scalable_quest(3001));
    MHF_CHECK(is_guild_scalable_quest(3233));
    MHF_CHECK(is_guild_scalable_quest(60001));

    // Excluded modes: none, Treasure, Village Elder, Village Nekoht, Training School
    MHF_CHECK(!is_guild_scalable_quest(0));
    MHF_CHECK(!is_guild_scalable_quest(4001));
    MHF_CHECK(!is_guild_scalable_quest(4007));
    MHF_CHECK(!is_guild_scalable_quest(10001));
    MHF_CHECK(!is_guild_scalable_quest(10511));
    MHF_CHECK(!is_guild_scalable_quest(11001));
    MHF_CHECK(!is_guild_scalable_quest(11226));
    MHF_CHECK(!is_guild_scalable_quest(20001));
    MHF_CHECK(!is_guild_scalable_quest(29999));
}

void test_groups_and_math() {
    MHF_CHECK(!is_large_monster_group(0));  // small monsters
    MHF_CHECK(!is_large_monster_group(-1));
    MHF_CHECK(is_large_monster_group(1));   // standard large monsters
    MHF_CHECK(is_large_monster_group(2));   // Fatalis / Lao-Shan Lung
    MHF_CHECK(is_large_monster_group(3));   // Akantor / Ukanlos / Shen / Yama
    MHF_CHECK(!is_large_monster_group(4));

    const auto f1 = guild_scale_factors(1);
    const auto f2 = guild_scale_factors(2);
    const auto f3 = guild_scale_factors(3);
    const auto f4 = guild_scale_factors(4);
    MHF_CHECK(f1.hp == 0.55f && f2.hp == 0.90f && f3.hp == 1.30f && f4.hp == 1.60f);
    MHF_CHECK(f1.stagger == 0.55f && f2.stagger == 0.90f && f3.stagger == 1.30f && f4.stagger == 1.60f);
    MHF_CHECK(guild_scale_factors(0).hp == f1.hp);
    MHF_CHECK(guild_scale_factors(9).hp == f4.hp);

    // Normal scaling + signed 16-bit clamp on 26,666-HP bosses at 3P/4P
    MHF_CHECK(scale_stat_s16(5400.0f, f1.hp) == 2970);
    MHF_CHECK(scale_stat_s16(5400.0f, f2.hp) == 4860);
    MHF_CHECK(scale_stat_s16(5400.0f, f3.hp) == 7020);
    MHF_CHECK(scale_stat_s16(5400.0f, f4.hp) == 8640);
    MHF_CHECK(scale_stat_s16(26666.0f, f3.hp) == 32767);
    MHF_CHECK(scale_stat_s16(26666.0f, f4.hp) == 32767);
    MHF_CHECK(scale_stat_s16(3500.0f, f4.stagger, kMaxPartStaggerS16) == 4000); // matches game_task's 4000 stagger cap
    MHF_CHECK(scale_stat_s16(1.0f, 0.1f) == 1);
    MHF_CHECK(scale_stat_s16(0.0f, 1.0f) == 0);
    MHF_CHECK(scale_stat_s16(-10.0f, 1.0f) == 0);

    // Proportional HP rescaling preserves percentage, never revives dead monsters
    MHF_CHECK(rescale_current_hp_s16(2700, 5400, 3780) == 1890); // 50% -> 50%
    MHF_CHECK(rescale_current_hp_s16(1, 5400, 3780) == 1);       // alive stays >= 1
    MHF_CHECK(rescale_current_hp_s16(0, 5400, 3780) == 0);       // dead stays dead
    MHF_CHECK(rescale_current_hp_s16(-5, 5400, 3780) == -5);     // overkill stays dead
}

void test_memory_readers_and_safety() {
    psprecomp::GuestMemory mem;
    // Uninitialized / out-of-bounds / unaligned pointers fail safely
    MHF_CHECK(!active_human_hunters(mem, kPlayerMgrPtr).has_value());
    MHF_CHECK(!active_human_hunters(mem, 0x00001000u).has_value());
    MHF_CHECK(!active_human_hunters(mem, kPlayerMgrPtr + 1u).has_value());
    GuildQuestScaling bad_symbols(GuildScalingSymbols{0x00001000u, 0x00001004u, 0x00001008u});
    bad_symbols.on_enemy_hp_begin(kEnemyBase);
    MHF_CHECK(bad_symbols.on_enemy_hp_computed(mem, 1, 5000.0f, true) == 5000.0f);
    MHF_CHECK(bad_symbols.on_enemy_stagger_store(mem, kEnemyBase, 1, 0, 300, true) == 300);

    // Felyne comrades in slots 4..5 are ignored when counting human hunters
    setup_players(mem, 1, /*add_comrades=*/true);
    MHF_CHECK(active_human_hunters(mem, kPlayerMgrPtr) == 1);
    setup_players(mem, 3, /*add_comrades=*/true);
    MHF_CHECK(active_human_hunters(mem, kPlayerMgrPtr) == 3);

    // Clearing the active flag on slot 2 drops count from 3 to 2 (disconnect simulation)
    mem.store32(kPlayerBase + 2u * 0x100u + 0x04u, 0u);
    MHF_CHECK(active_human_hunters(mem, kPlayerMgrPtr) == 2);
}

void test_spawn_scaling_and_exclusions() {
    psprecomp::GuestMemory mem;
    GuildQuestScaling scaling(make_symbols());
    set_species_group(mem, 1, 1); // Rathalos (large, group 1)
    set_species_group(mem, 3, 0); // Velociprey (small, group 0)
    set_species_group(mem, 2, 2); // Fatalis (boss, group 2)

    // Village quest (10101): never scaled even when enabled and solo
    setup_quest(mem, 10101);
    setup_players(mem, 1);
    scaling.on_enemy_hp_begin(kEnemyBase);
    MHF_CHECK(scaling.on_enemy_hp_computed(mem, 1, 3200.0f, true) == 3200.0f);
    MHF_CHECK(scaling.on_enemy_stagger_store(mem, kEnemyBase, 1, 0, 200, true) == 200);
    MHF_CHECK(scaling.tracked_enemy_count() == 0);

    // Guild G-Rank quest (3201) solo (1P): large monster scaled to 0.55x HP and stagger, small monster untouched
    setup_quest(mem, 3201);
    setup_players(mem, 1);
    scaling.on_enemy_hp_begin(kEnemyBase);
    MHF_CHECK(scaling.on_enemy_hp_computed(mem, 1, 8000.0f, true) == 4400.0f);
    MHF_CHECK(scaling.on_enemy_stagger_store(mem, kEnemyBase, 1, 0, 400, true) == 220);

    // Small monster in the same Guild quest stays untouched
    scaling.on_enemy_hp_begin(kEnemyBase + 0x1000u);
    MHF_CHECK(scaling.on_enemy_hp_computed(mem, 3, 120.0f, true) == 120.0f);
    MHF_CHECK(scaling.on_enemy_stagger_store(mem, kEnemyBase + 0x1000u, 3, 0, 40, true) == 40);
    MHF_CHECK(scaling.tracked_enemy_count() == 1);

    // 4-player Fatalis clamps cleanly to s16 max (32767) and part stagger clamps at game_task's 4000 cap
    setup_players(mem, 4);
    scaling.on_enemy_hp_begin(kEnemyBase + 0x2000u);
    MHF_CHECK(scaling.on_enemy_hp_computed(mem, 2, 26666.0f, true) == 32767.0f);
    MHF_CHECK(scaling.on_enemy_stagger_store(mem, kEnemyBase + 0x2000u, 2, 0, 3200, true) == 4000);
}

void test_quest_header_not_countdown_selects_scaling() {
    psprecomp::GuestMemory mem;
    GuildQuestScaling scaling(make_symbols());
    constexpr std::uint32_t header = 0x08903000u;
    setup_players(mem, 1);
    set_species_group(mem, 1, 1);
    mem.store32(kQuestMgrPtr, kQuestMgr);
    mem.store32(kQuestMgr + 0x4cu, header);
    mem.store32(header + 4u, 0x47444e32u);
    mem.store32(kQuestMgr + 0x10u, 90000u);
    mem.store16(header + 0x64u, 10101u); // Village, despite a large countdown
    scaling.on_enemy_hp_begin(kEnemyBase);
    MHF_CHECK(scaling.on_enemy_hp_computed(mem, 1, 5000.0f, true) == 5000.0f);
    mem.store32(kQuestMgr + 0x10u, 50000u);
    mem.store16(header + 0x64u, 1014u); // Guild, despite a smaller countdown
    scaling.on_enemy_hp_begin(kEnemyBase);
    MHF_CHECK(scaling.on_enemy_hp_computed(mem, 1, 5000.0f, true) == 2750.0f);
    mem.store32(header + 4u, 0u); // invalid header fails closed
    scaling.on_enemy_hp_begin(kEnemyBase);
    MHF_CHECK(scaling.on_enemy_hp_computed(mem, 1, 5000.0f, true) == 5000.0f);
}

void test_dynamic_mid_quest_rescaling_and_live_toggle() {
    psprecomp::GuestMemory mem;
    GuildQuestScaling scaling(make_symbols());
    set_species_group(mem, 1, 1); // Rathalos
    setup_quest(mem, 3005);       // Guild G1*
    setup_players(mem, 3);        // 3 hunters depart together (1.30x HP & stagger)

    scaling.on_enemy_hp_begin(kEnemyBase);
    const auto spawned_hp = static_cast<std::int16_t>(scaling.on_enemy_hp_computed(mem, 1, 6000.0f, true));
    MHF_CHECK(spawned_hp == 7800);
    setup_enemy(mem, kEnemyBase, 1, spawned_hp, spawned_hp);

    const std::int16_t spawned_stagger = scaling.on_enemy_stagger_store(mem, kEnemyBase, 1, 0, 400, true);
    MHF_CHECK(spawned_stagger == 520);
    setup_part(mem, kEnemyBase, 0, spawned_stagger);

    // Hunters deal 50% damage to HP (3900 left) and 50% to part stagger (260 left)
    mem.store16(kEnemyBase + kEnemyCurrentHp, 3900);
    mem.store16(kEnemyBase + kEnemyPartBase + kEnemyPartThreshold, 260);

    // Two teammates disconnect -> drops from 3P (1.30x) to 1P (0.55x HP = 3300, stagger = 220)
    setup_players(mem, 1);
    scaling.sync(mem, true);
    MHF_CHECK(static_cast<std::int16_t>(mem.load16(kEnemyBase + kEnemyMaxHp)) == 3300);
    MHF_CHECK(static_cast<std::int16_t>(mem.load16(kEnemyBase + kEnemyCurrentHp)) == 1650); // exact 50% preserved
    MHF_CHECK(static_cast<std::int16_t>(mem.load16(kEnemyBase + kEnemyPartBase + kEnemyPartThreshold)) == 110);

    // Player toggles the option OFF in Esc > Game -> restores vanilla 2P (6000 HP, 400 stagger), still 50% remaining
    scaling.sync(mem, false);
    MHF_CHECK(static_cast<std::int16_t>(mem.load16(kEnemyBase + kEnemyMaxHp)) == 6000);
    MHF_CHECK(static_cast<std::int16_t>(mem.load16(kEnemyBase + kEnemyCurrentHp)) == 3000);
    MHF_CHECK(static_cast<std::int16_t>(mem.load16(kEnemyBase + kEnemyPartBase + kEnemyPartThreshold)) == 200);

    // Monster dies -> sync drops it from tracking and never revives it
    mem.store16(kEnemyBase + kEnemyCurrentHp, 0);
    scaling.sync(mem, true);
    MHF_CHECK(static_cast<std::int16_t>(mem.load16(kEnemyBase + kEnemyCurrentHp)) == 0);
    MHF_CHECK(scaling.tracked_enemy_count() == 0);

    // Leaving quest clears state
    setup_quest(mem, 0);
    scaling.sync(mem, true);
    MHF_CHECK(scaling.tracked_enemy_count() == 0);
}

void test_late_player_init_and_carryover_hp_clamp() {
    psprecomp::GuestMemory mem;
    GuildQuestScaling scaling(make_symbols());
    set_species_group(mem, 1, 1);
    setup_quest(mem, 2105); // Guild High Rank
    setup_players(mem, 0, false); // players not initialized yet at spawn instant

    // Falls back to vanilla 1.00x at spawn when 0 players are active
    scaling.on_enemy_hp_begin(kEnemyBase);
    const auto spawned_hp = static_cast<std::int16_t>(scaling.on_enemy_hp_computed(mem, 1, 5000.0f, true));
    MHF_CHECK(spawned_hp == 5000);
    setup_enemy(mem, kEnemyBase, 1, spawned_hp, spawned_hp);

    // Once the solo hunter finishes loading, sync scales to 1P (2750 HP)
    setup_players(mem, 1);
    scaling.sync(mem, true);
    MHF_CHECK(static_cast<std::int16_t>(mem.load16(kEnemyBase + kEnemyMaxHp)) == 2750);
    MHF_CHECK(static_cast<std::int16_t>(mem.load16(kEnemyBase + kEnemyCurrentHp)) == 2750);

    // If Elder Dragon carry-over overwrote current HP above scaled max, sync clamps it to max
    mem.store16(kEnemyBase + kEnemyCurrentHp, 4500);
    scaling.sync(mem, true);
    MHF_CHECK(static_cast<std::int16_t>(mem.load16(kEnemyBase + kEnemyCurrentHp)) == 2750);
}

void test_two_player_toggle_restores_vanilla() {
    psprecomp::GuestMemory mem;
    GuildQuestScaling scaling(make_symbols());
    set_species_group(mem, 1, 1);
    setup_quest(mem, 2105);
    setup_players(mem, 2);
    scaling.on_enemy_hp_begin(kEnemyBase);
    const auto hp = static_cast<std::int16_t>(scaling.on_enemy_hp_computed(mem, 1, 5000.0f, true));
    MHF_CHECK(hp == 4500);
    setup_enemy(mem, kEnemyBase, 1, hp, hp);
    scaling.sync(mem, false);
    MHF_CHECK(mem.load16(kEnemyBase + kEnemyMaxHp) == 5000);
    scaling.sync(mem, true);
    MHF_CHECK(mem.load16(kEnemyBase + kEnemyMaxHp) == 4500);
}

} // namespace

int main() {
    test_quest_classification();
    test_groups_and_math();
    test_memory_readers_and_safety();
    test_spawn_scaling_and_exclusions();
    test_quest_header_not_countdown_selects_scaling();
    test_dynamic_mid_quest_rescaling_and_live_toggle();
    test_late_player_init_and_carryover_hp_clamp();
    test_two_player_toggle_restores_vanilla();
    return mhf::test::finish("mhfu_guild_scaling");
}
