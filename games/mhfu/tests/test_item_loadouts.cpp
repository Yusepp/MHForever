#include "item_loadouts.hpp"
#include "check.hpp"

#include <cstdint>

namespace {

constexpr std::uint32_t kGameSysPtrAddr = 0x089C7508u;
constexpr std::uint32_t kGameSysAddr = 0x09020000u;
constexpr std::uint32_t kPouchBase = kGameSysAddr + 0x6AEA6u;
constexpr std::uint32_t kBoxBase = kGameSysAddr + 0x3428u;

void set_slot(psprecomp::GuestMemory &mem, std::uint32_t base, std::size_t idx, std::uint16_t id, std::uint16_t count) {
    const std::uint32_t addr = base + static_cast<std::uint32_t>(idx) * 4u;
    mem.store16(addr, id);
    mem.store16(addr + 2u, count);
}

void test_save_and_restock_roundtrip() {
    psprecomp::GuestMemory mem;
    mem.store32(kGameSysPtrAddr, kGameSysAddr);
    mem.store8(kGameSysAddr + 0x2Eu, 0u); // Village

    // Pouch has 10 Potions (id=10) and 5 Mega Potions (id=11).
    set_slot(mem, kPouchBase, 0, 10u, 10u);
    set_slot(mem, kPouchBase, 1, 11u, 5u);
    // Box has 50 Potions (id=10) and 20 Mega Potions (id=11).
    set_slot(mem, kBoxBase, 0, 10u, 50u);
    set_slot(mem, kBoxBase, 1, 11u, 20u);

    mhf::mhfu::ItemLoadouts loadouts({kGameSysPtrAddr});
    MHF_CHECK(loadouts.can_use_loadouts(mem));
    MHF_CHECK(loadouts.save_preset(mem, 0).ok);

    // Simulate returning from a quest with 3 Potions left and 0 Mega Potions.
    set_slot(mem, kPouchBase, 0, 10u, 3u);
    set_slot(mem, kPouchBase, 1, 0u, 0u);

    const auto res = loadouts.apply_preset(mem, 0);
    MHF_CHECK(res.ok);
    // Pouch restocked to 10 Potions and 5 Mega Potions.
    MHF_CHECK(mem.load16(kPouchBase + 0u) == 10u && mem.load16(kPouchBase + 2u) == 10u);
    MHF_CHECK(mem.load16(kPouchBase + 4u) == 11u && mem.load16(kPouchBase + 6u) == 5u);
    // Box counts conserved: 50 + 3 - 10 = 43 Potions, 20 - 5 = 15 Mega Potions.
    MHF_CHECK(mem.load16(kBoxBase + 0u) == 10u && mem.load16(kBoxBase + 2u) == 43u);
    MHF_CHECK(mem.load16(kBoxBase + 4u) == 11u && mem.load16(kBoxBase + 6u) == 15u);
}

void test_blocked_during_quest() {
    psprecomp::GuestMemory mem;
    mem.store32(kGameSysPtrAddr, kGameSysAddr);
    mem.store8(kGameSysAddr + 0x2Eu, 2u); // Active quest

    mhf::mhfu::ItemLoadouts loadouts({kGameSysPtrAddr});
    MHF_CHECK(!loadouts.can_use_loadouts(mem));
    MHF_CHECK(!loadouts.save_preset(mem, 0).ok);
}

} // namespace

int main() {
    test_save_and_restock_roundtrip();
    test_blocked_during_quest();
    return 0;
}
