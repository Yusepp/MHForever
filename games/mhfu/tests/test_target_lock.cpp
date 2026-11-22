#include "target_lock.hpp"
#include "check.hpp"

#include <bit>
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
constexpr std::uint32_t kEnemy1 = 0x09050000u;
constexpr std::uint32_t kEnemy2 = 0x09052000u;
constexpr std::uint32_t kQuestMgr = 0x09060000u;
constexpr std::uint32_t kCameraCtrl = 0x09070000u;

void put_vec3(psprecomp::GuestMemory &mem, std::uint32_t addr, float x, float y, float z) {
    mem.store32(addr + 0u, std::bit_cast<std::uint32_t>(x));
    mem.store32(addr + 4u, std::bit_cast<std::uint32_t>(y));
    mem.store32(addr + 8u, std::bit_cast<std::uint32_t>(z));
}

void setup_two_monsters(psprecomp::GuestMemory &mem) {
    mem.store32(kGameSysPtr, kGameSys);
    mem.store32(kPlayerMgrPtr, kPlayerMgr);
    mem.store32(kEnemyMgrPtr, kEnemyMgr);
    mem.store32(kQuestMgrPtr, kQuestMgr);
    mem.store8(kGameSys + 0x28u, 0u);
    mem.store8(kGameSys + 0x2Eu, 2u);
    mem.store32(kPlayerMgr + 0x0Cu, kPlayer);
    mem.store16(kPlayer + 0x29Au, 5u);
    put_vec3(mem, kPlayer + 0x200u, 0.0f, 0.0f, 0.0f);

    auto init_enemy = [&](std::uint32_t slot, std::uint32_t e, std::uint8_t sp, float x, float z) {
        mem.store32(kEnemyMgr + 0x1220u + slot * 4u, e);
        mem.store32(e + 0x04u, 1u); // bit 0 = active
        mem.store8(e + 0x1E8u, sp);
        mem.store16(e + 0x29Au, 5u);
        mem.store16(e + 0x2E4u, 800u);
        mem.store16(e + 0x41Eu, 1000u);
        put_vec3(mem, e + 0x200u, x, 0.0f, z);
        mem.store8(kSpeciesTable + static_cast<std::uint32_t>(sp) * 0x1D0u + 0x14Fu, 1u);
    };
    init_enemy(0, kEnemy1, 0x01u, 100.0f, 0.0f); // closer (+X -> yaw ~ 16384)
    init_enemy(1, kEnemy2, 0x02u, 0.0f, 300.0f); // farther (+Z -> yaw ~ 0)
}

void test_lock_cycle_and_steer() {
    psprecomp::GuestMemory mem;
    setup_two_monsters(mem);

    mhf::mhfu::MonsterTracker tracker({kGameSysPtr, kPlayerMgrPtr, kEnemyMgrPtr, kSpeciesTable, kQuestMgrPtr});
    tracker.poll(mem, false, false);

    mhf::mhfu::TargetLock lock;
    // Press R3 -> locks nearest (kEnemy1).
    lock.update(tracker, true, true);
    MHF_CHECK(lock.locked() && lock.locked_enemy() == kEnemy1);

    // Steering camera from yaw 0 toward +X (16384) moves yaw positively.
    mem.store16(kCameraCtrl + 0x80u, 0u);
    MHF_CHECK(lock.steer_camera(mem, kCameraCtrl, tracker.player_pos(), false));
    MHF_CHECK(mem.load16(kCameraCtrl + 0x80u) > 0u);

    // Release and press R3 again -> cycles to kEnemy2.
    lock.update(tracker, false, true);
    lock.update(tracker, true, true);
    MHF_CHECK(lock.locked() && lock.locked_enemy() == kEnemy2);

    // Release and press R3 again -> unlocks.
    lock.update(tracker, false, true);
    lock.update(tracker, true, true);
    MHF_CHECK(!lock.locked());
}

} // namespace

int main() {
    test_lock_cycle_and_steer();
    return 0;
}
