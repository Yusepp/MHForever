#include "infinite_consumables.hpp"
#include "check.hpp"

#include <cstdint>

namespace {

constexpr std::uint32_t kGameSysPtrAddr = 0x089C7508u;
constexpr std::uint32_t kGameSysAddr = 0x09020000u;
constexpr std::uint32_t kPouchBase = kGameSysAddr + 0x6AEA6u;

void set_slot(psprecomp::GuestMemory &mem, std::size_t slot, std::uint16_t id, std::uint16_t count) {
    const std::uint32_t addr = kPouchBase + static_cast<std::uint32_t>(slot) * 4u;
    mem.store16(addr, id);
    mem.store16(addr + 2u, count);
}

void test_infinite_paintballs_and_whetstone() {
    psprecomp::GuestMemory mem;
    mem.store32(kGameSysPtrAddr, kGameSysAddr);
    mem.store8(kGameSysAddr + 0x2Eu, 2u); // active quest
    set_slot(mem, 0, mhf::mhfu::kItemPaintball, 1u);
    set_slot(mem, 1, mhf::mhfu::kItemWhetstone, 5u);

    mhf::mhfu::InfiniteConsumables qol({kGameSysPtrAddr});
    qol.poll(mem, true, true);

    // Simulate throwing the last paintball (game clears slot to 0, 0) and using one whetstone (5 -> 4).
    set_slot(mem, 0, 0u, 0u);
    set_slot(mem, 1, mhf::mhfu::kItemWhetstone, 4u);
    qol.poll(mem, true, true);

    MHF_CHECK(mem.load16(kPouchBase + 0u) == mhf::mhfu::kItemPaintball);
    MHF_CHECK(mem.load16(kPouchBase + 2u) == 1u);
    MHF_CHECK(mem.load16(kPouchBase + 4u) == mhf::mhfu::kItemWhetstone);
    MHF_CHECK(mem.load16(kPouchBase + 6u) == 5u);
}

void test_temp_whetstone_and_paintball_granted_and_cleaned_up() {
    psprecomp::GuestMemory mem;
    mem.store32(kGameSysPtrAddr, kGameSysAddr);
    mem.store8(kGameSysAddr + 0x2Eu, 2u); // active quest, empty pouch

    mhf::mhfu::InfiniteConsumables qol({kGameSysPtrAddr});
    qol.poll(mem, true, true);

    // Temporary whetstone in slot 23, temporary paintball in slot 22.
    const std::uint32_t slot23 = kPouchBase + 23u * 4u;
    const std::uint32_t slot22 = kPouchBase + 22u * 4u;
    MHF_CHECK(mem.load16(slot23) == mhf::mhfu::kItemWhetstone && mem.load16(slot23 + 2u) == 1u);
    MHF_CHECK(mem.load16(slot22) == mhf::mhfu::kItemPaintball && mem.load16(slot22 + 2u) == 1u);

    // Return to village -> both temporary items removed cleanly.
    mem.store8(kGameSysAddr + 0x2Eu, 0u);
    qol.poll(mem, true, true);
    MHF_CHECK(mem.load16(slot23) == 0u && mem.load16(slot23 + 2u) == 0u);
    MHF_CHECK(mem.load16(slot22) == 0u && mem.load16(slot22 + 2u) == 0u);
}

} // namespace

int main() {
    test_infinite_paintballs_and_whetstone();
    test_temp_whetstone_and_paintball_granted_and_cleaned_up();
    return 0;
}
