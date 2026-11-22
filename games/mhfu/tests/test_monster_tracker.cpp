#include "monster_tracker.hpp"
#include "check.hpp"

#include <cstdint>

namespace {

constexpr std::uint32_t kGameSysPtr = 0x089C7508u;
constexpr std::uint32_t kPlayerMgrPtr = 0x09A4AE04u;
constexpr std::uint32_t kEnemyMgrPtr = 0x09C0D430u;
constexpr std::uint32_t kSpeciesTable = 0x09BB3C20u;
constexpr std::uint32_t kQuestMgrPtr = 0x08A5DD4Cu;

constexpr std::uint32_t kGameSys = 0x09020000u;
constexpr std::uint32_t kPlayerMgr = 0x09030000u;
constexpr std::uint32_t kPlayer = 0x09031000u;
constexpr std::uint32_t kEnemyMgr = 0x09040000u;
constexpr std::uint32_t kEnemy = 0x09050000u;
constexpr std::uint32_t kQuestMgr = 0x09060000u;

void setup_scene(psprecomp::GuestMemory &mem, std::uint8_t species, std::int16_t hp, std::int16_t max_hp, std::uint16_t area) {
    mem.store32(kGameSysPtr, kGameSys);
    mem.store32(kPlayerMgrPtr, kPlayerMgr);
    mem.store32(kEnemyMgrPtr, kEnemyMgr);
    mem.store32(kQuestMgrPtr, kQuestMgr);

    mem.store8(kGameSys + 0x28u, 0u);
    mem.store8(kGameSys + 0x2Eu, 2u); // active quest
    mem.store32(kPlayerMgr + 0x0Cu, kPlayer);
    mem.store16(kPlayer + 0x29Au, 3u); // player in area 3

    mem.store32(kEnemyMgr + 0x1220u, kEnemy);
    mem.store32(kEnemy + 0x04u, 0u);
    mem.store8(kEnemy + 0x1E8u, species);
    mem.store16(kEnemy + 0x29Au, area);
    mem.store16(kEnemy + 0x2E4u, static_cast<std::uint16_t>(hp));
    mem.store16(kEnemy + 0x41Eu, static_cast<std::uint16_t>(max_hp));
    mem.store32(kEnemy + 0x414u, 0u);

    // Mark species as large monster (scale_group = 1).
    mem.store8(kSpeciesTable + static_cast<std::uint32_t>(species) * 0x1D0u + 0x14Fu, 1u);
}

void test_quest_start_tracker_and_capturable() {
    psprecomp::GuestMemory mem;
    setup_scene(mem, 0x01u /* Rathian */, 1000, 1000, 3u);

    mhf::mhfu::MonsterTracker tracker({kGameSysPtr, kPlayerMgrPtr, kEnemyMgrPtr, kSpeciesTable, kQuestMgrPtr});
    tracker.poll(mem, true, true);
    MHF_CHECK(tracker.in_active_quest());
    MHF_CHECK(tracker.large_monsters().size() == 1u);
    MHF_CHECK(mem.load32(kEnemy + 0x414u) == mhf::mhfu::kQuestStartTrackFrames);
    MHF_CHECK(!tracker.capturable_in_area());

    // Simulate timer expiring to 0 -> must NOT be re-applied during the same quest!
    mem.store32(kEnemy + 0x414u, 0u);
    // Drop HP to 20% -> now capturable!
    mem.store16(kEnemy + 0x2E4u, 200u);
    tracker.poll(mem, true, true);
    MHF_CHECK(mem.load32(kEnemy + 0x414u) == 0u);
    MHF_CHECK(tracker.capturable_in_area());

    // Elder Dragon (Kirin 0x16) at 10% HP -> never capturable!
    setup_scene(mem, 0x16u, 100, 1000, 3u);
    tracker.poll(mem, false, true);
    MHF_CHECK(!tracker.capturable_in_area());
}

} // namespace

int main() {
    test_quest_start_tracker_and_capturable();
    return 0;
}
