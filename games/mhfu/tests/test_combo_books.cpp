#include "combo_books.hpp"
#include "test_support.hpp"

#include <cstdint>

namespace {

constexpr std::uint32_t kGameSysPtrAddr = 0x089C7508u;
constexpr std::uint32_t kGameSysAddr = 0x09020000u;
constexpr std::uint32_t kPouchBase = kGameSysAddr + 0x6AEA6u;
constexpr std::uint32_t kBoxBase = kGameSysAddr + 0x3428u;

void set_slot(psprecomp::GuestMemory &mem, std::uint32_t base, std::size_t slot, std::uint16_t id, std::uint16_t count) {
    const std::uint32_t addr = base + static_cast<std::uint32_t>(slot) * 4u;
    mem.store16(addr, id);
    mem.store16(addr + 2u, count);
}

void test_requires_all_five_books() {
    psprecomp::GuestMemory mem;
    mem.store32(kGameSysPtrAddr, kGameSysAddr);
    const mhf::mhfu::ComboBooksSymbols syms{kGameSysPtrAddr};

    // Only 4 books in item box -> not 100%.
    for (std::uint16_t id = 1; id <= 4; ++id) set_slot(mem, kBoxBase, id, id, 1u);
    MHF_CHECK(!mhf::mhfu::owns_all_five_combo_books(mem, syms));
    MHF_CHECK(mhf::mhfu::apply_combo_books_rate(mem, syms, 75u, true) == 75u);

    // 5th book in pouch -> all 5 owned across pouch + box -> 100%!
    set_slot(mem, kPouchBase, 0, 5u, 1u);
    MHF_CHECK(mhf::mhfu::owns_all_five_combo_books(mem, syms));
    MHF_CHECK(mhf::mhfu::apply_combo_books_rate(mem, syms, 55u, true) == 100u);
    // Disabled -> unchanged.
    MHF_CHECK(mhf::mhfu::apply_combo_books_rate(mem, syms, 55u, false) == 55u);
    // Invalid recipe (-1) -> unchanged.
    MHF_CHECK(mhf::mhfu::apply_combo_books_rate(mem, syms, static_cast<std::uint32_t>(-1), true) == static_cast<std::uint32_t>(-1));
}

} // namespace

int main() {
    test_requires_all_five_books();
    return 0;
}
