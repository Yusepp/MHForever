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

    mhf::mhfu::ItemLoadouts loadouts({kGameSysPtrAddr, std::nullopt});
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

    mhf::mhfu::ItemLoadouts loadouts({kGameSysPtrAddr, std::nullopt});
    MHF_CHECK(!loadouts.can_use_loadouts(mem));
    MHF_CHECK(!loadouts.save_preset(mem, 0).ok);
}

void test_armor_set_save_and_equip_after_box_sort() {
    psprecomp::GuestMemory mem;
    mem.store32(kGameSysPtrAddr, kGameSysAddr);
    mem.store8(kGameSysAddr + 0x2Eu, 0u); // Village

    constexpr std::uint32_t kEqBase = kGameSysAddr + 0x4C0u;
    constexpr std::uint32_t kEqBox = kGameSysAddr + 0x548u;
    constexpr std::uint32_t kEqIdx = kGameSysAddr + 0x4A0u + 16418u;

    // Equip weapon (slot 0) at box index 2, head (slot 2) at box index 5.
    auto put_piece = [&](std::uint32_t addr, std::uint8_t type, std::uint8_t id) {
        mem.store8(addr + 1u, type);
        mem.store8(addr + 2u, id);
    };
    put_piece(kEqBase + 0u * 12u, 5u, 42u);
    put_piece(kEqBox + 2u * 12u, 5u, 42u);
    mem.store16(kEqIdx + 0u * 2u, 2u);

    put_piece(kEqBase + 2u * 12u, 2u, 19u);
    put_piece(kEqBox + 5u * 12u, 2u, 19u);
    mem.store16(kEqIdx + 2u * 2u, 5u);

    mhf::mhfu::ItemLoadouts loadouts({kGameSysPtrAddr, std::nullopt});
    MHF_CHECK(loadouts.save_armor_preset(mem, 0).ok);

    // Equip different gear AND move saved pieces to new box slots (simulating box sort!).
    put_piece(kEqBase + 0u * 12u, 5u, 99u);
    put_piece(kEqBase + 2u * 12u, 2u, 88u);
    put_piece(kEqBox + 2u * 12u, 0u, 0u);
    put_piece(kEqBox + 5u * 12u, 0u, 0u);
    put_piece(kEqBox + 12u * 12u, 5u, 42u);
    put_piece(kEqBox + 15u * 12u, 2u, 19u);

    MHF_CHECK(loadouts.apply_armor_preset(mem, 0).ok);
    MHF_CHECK(mem.load8(kEqBase + 0u * 12u + 2u) == 42u);
    MHF_CHECK(mem.load16(kEqIdx + 0u * 2u) == 12u);
    MHF_CHECK(mem.load8(kEqBase + 2u * 12u + 2u) == 19u);
    MHF_CHECK(mem.load16(kEqIdx + 2u * 2u) == 15u);
}

} // namespace

int main() {
    test_save_and_restock_roundtrip();
    test_blocked_during_quest();
    test_armor_set_save_and_equip_after_box_sort();
    return 0;
}
