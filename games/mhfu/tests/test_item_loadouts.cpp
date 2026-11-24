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

    mhf::mhfu::ItemLoadouts loadouts({kGameSysPtrAddr, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
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

    mhf::mhfu::ItemLoadouts loadouts({kGameSysPtrAddr, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
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

    mhf::mhfu::ItemLoadouts loadouts({kGameSysPtrAddr, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
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

void test_select_tap_sorts_and_hold_opens() {
    using namespace mhf::psp::buttons;
    psprecomp::GuestMemory mem;
    mem.store32(kGameSysPtrAddr, kGameSysAddr);
    mem.store8(kGameSysAddr + 0x2Eu, 0u); // Village

    mhf::mhfu::ItemLoadouts loadouts({kGameSysPtrAddr, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
    loadouts.mark_box_open(100'000u);
    loadouts.poll_box_ui(mem, 0u, 100'000u, true);
    MHF_CHECK(!loadouts.panel_open());
    MHF_CHECK(loadouts.game_button_clear_mask() == kSelect);
    MHF_CHECK(loadouts.game_button_force_mask() == 0u);

    // Short tap (< 0.35s) forwards a clean SELECT sort pulse without opening the panel.
    loadouts.mark_box_open(120'000u);
    loadouts.poll_box_ui(mem, kSelect, 120'000u, true);
    MHF_CHECK(!loadouts.panel_open());
    MHF_CHECK(loadouts.select_hold_progress(220'000u) > 0.2f);
    loadouts.mark_box_open(220'000u);
    loadouts.poll_box_ui(mem, 0u, 220'000u, true);
    MHF_CHECK(!loadouts.panel_open());
    MHF_CHECK(loadouts.game_button_force_mask() == kSelect);
    MHF_CHECK(loadouts.game_button_clear_mask() == 0u);

    // After pulse expires, SELECT is masked again.
    loadouts.mark_box_open(300'000u);
    loadouts.poll_box_ui(mem, 0u, 300'000u, true);
    MHF_CHECK(loadouts.game_button_force_mask() == 0u);
    MHF_CHECK(loadouts.game_button_clear_mask() == kSelect);

    // Holding SELECT for >= 0.35s opens Loadouts without firing a sort pulse.
    loadouts.mark_box_open(320'000u);
    loadouts.poll_box_ui(mem, kSelect, 320'000u, true);
    loadouts.mark_box_open(700'000u);
    loadouts.poll_box_ui(mem, kSelect, 700'000u, true);
    MHF_CHECK(loadouts.panel_open());
    MHF_CHECK(loadouts.suppress_game_input());
    MHF_CHECK(loadouts.game_button_force_mask() == 0u);

    // Closing with Cross and releasing does not emit an accidental sort pulse.
    loadouts.mark_box_open(740'000u);
    loadouts.poll_box_ui(mem, 0u, 740'000u, true);
    loadouts.mark_box_open(760'000u);
    loadouts.poll_box_ui(mem, kCross, 760'000u, true);
    MHF_CHECK(!loadouts.panel_open());
    loadouts.mark_box_open(780'000u);
    loadouts.poll_box_ui(mem, 0u, 780'000u, true);
    MHF_CHECK(loadouts.game_button_force_mask() == 0u);
}

void test_rename_loadout_and_osk() {
    using namespace mhf::psp::buttons;
    psprecomp::GuestMemory mem;
    mem.store32(kGameSysPtrAddr, kGameSysAddr);
    mem.store8(kGameSysAddr + 0x2Eu, 0u);

    mhf::mhfu::ItemLoadouts loadouts({kGameSysPtrAddr, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
    // Open panel via hold SELECT
    loadouts.mark_box_open(100'000u);
    loadouts.poll_box_ui(mem, kSelect, 100'000u, true);
    loadouts.mark_box_open(500'000u);
    loadouts.poll_box_ui(mem, kSelect, 500'000u, true);
    MHF_CHECK(loadouts.panel_open());

    // Press Square to open Rename OSK
    loadouts.mark_box_open(520'000u);
    loadouts.poll_box_ui(mem, kSquare, 520'000u, true);
    MHF_CHECK(loadouts.renaming());

    // Type 'A' via OSK (row 0, col 0) with Circle, then 'K' (row 1, col 0), then physical keyboard "47", then Confirm with START
    loadouts.mark_box_open(540'000u);
    loadouts.poll_box_ui(mem, kCircle, 540'000u, true);
    loadouts.mark_box_open(560'000u);
    loadouts.poll_box_ui(mem, kDown, 560'000u, true);
    loadouts.mark_box_open(580'000u);
    loadouts.poll_box_ui(mem, kCircle, 580'000u, true);
    loadouts.handle_text_input("47");
    loadouts.mark_box_open(600'000u);
    loadouts.poll_box_ui(mem, kStart, 600'000u, true);

    MHF_CHECK(!loadouts.renaming());
    MHF_CHECK(loadouts.preset(0).name == "AK47");

    // Clear the selected set with SELECT
    loadouts.mark_box_open(620'000u);
    loadouts.poll_box_ui(mem, 0u, 620'000u, true);
    loadouts.mark_box_open(640'000u);
    loadouts.poll_box_ui(mem, kSelect, 640'000u, true);
    MHF_CHECK(!loadouts.has_preset(0));
    MHF_CHECK(loadouts.preset(0).name.empty());
}

} // namespace

int main() {
    test_save_and_restock_roundtrip();
    test_blocked_during_quest();
    test_armor_set_save_and_equip_after_box_sort();
    test_select_tap_sorts_and_hold_opens();
    test_rename_loadout_and_osk();
    return 0;
}
