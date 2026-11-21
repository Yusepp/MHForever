// Synthetic guest RAM only. The gauge reads what the game uses for the stock sword icon.
#include "sharpness.hpp"

#include "check.hpp"

namespace {

using mhf::mhfu::read_sharpness;
using mhf::mhfu::SharpnessSymbols;

constexpr std::uint32_t kGameSys = 0x09000000, kManager = 0x09001000, kPlayer = 0x09002000,
                        kCockpit = 0x09004000, kRows = 0x09005000;
constexpr SharpnessSymbols kSymbols{0x08A00000, 0x08A00004, 0x08A00008, 0x08A01000, 0x08A10000,
                                    0x08A11000, kRows};

} // namespace

int main() {
    psprecomp::GuestMemory memory;
    memory.store32(kSymbols.game_sys, kGameSys);
    memory.store32(kSymbols.player_manager, kManager);
    memory.store32(kSymbols.cockpit, kCockpit);
    memory.store8(kGameSys + 0x28, 2);
    memory.store32(kManager + 0x14, kPlayer);
    memory.store8(kPlayer + 0x4E9, 5); // blademaster
    memory.store16(kPlayer + 0x4EA, 3);
    memory.store16(kSymbols.weapons + 3 * 24, 10);
    memory.store8(kSymbols.weapons + 3 * 24 + 8, 2);
    memory.store8(kSymbols.model_classes + 10, 4);
    memory.store32(kSymbols.class_rows + 4 * 4, kRows);
    const std::array<std::uint16_t, 7> row{30, 60, 100, 150, 200, 200, 450};
    for (std::uint32_t i = 0; i < row.size(); ++i) memory.store16(kRows + 2 * 14 + i * 2, row[i]);

    memory.store16(kPlayer + 0x60C, 150);
    memory.store16(kPlayer + 0x10F2, 120);
    const auto green = read_sharpness(memory, kSymbols);
    MHF_CHECK(green && green->current == 120 && green->maximum == 150);
    MHF_CHECK(green && green->ends[2] == 100 && green->ends[3] == 150 && green->ends[4] == 150);

    for (const std::uint16_t current : {std::uint16_t{100}, std::uint16_t{0}, std::uint16_t{150}}) {
        memory.store16(kPlayer + 0x10F2, current);
        const auto gauge = read_sharpness(memory, kSymbols);
        MHF_CHECK(gauge && gauge->current == current);
    }
    memory.store16(kPlayer + 0x60C, 220);
    memory.store16(kPlayer + 0x10F2, 220);
    const auto extended = read_sharpness(memory, kSymbols);
    MHF_CHECK(extended && extended->maximum == 220 && extended->ends[4] == 200 &&
              extended->ends[5] == 200 && extended->ends[6] == 220);

    memory.store16(kPlayer + 0x10F2, 221);
    MHF_CHECK(!read_sharpness(memory, kSymbols));
    memory.store16(kPlayer + 0x10F2, 220);
    memory.store8(kCockpit + 0x604, 1);
    MHF_CHECK(!read_sharpness(memory, kSymbols));
    memory.store8(kCockpit + 0x604, 0);
    memory.store8(kPlayer + 0x4E9, 6);
    MHF_CHECK(!read_sharpness(memory, kSymbols));
    memory.store8(kPlayer + 0x4E9, 5);
    memory.store32(kSymbols.class_rows + 4 * 4, kRows - 14);
    MHF_CHECK(!read_sharpness(memory, kSymbols));
    memory.store32(kSymbols.class_rows + 4 * 4, kRows);
    memory.store32(kManager + 0x14, 0xFFFFFFFE);
    MHF_CHECK(!read_sharpness(memory, kSymbols));
    return mhf::test::finish("test_sharpness");
}
