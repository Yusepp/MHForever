#include "sharpness.hpp"

#include "quest_player.hpp"

#include <algorithm>

namespace mhf::mhfu {
namespace {

// The local hunter's equipment, effective capacity (including skills), and the value used by the cockpit's sword.
constexpr std::uint32_t kWeaponType = 0x4E9, kWeaponId = 0x4EA, kMaximum = 0x60C, kCurrent = 0x10F2;
constexpr std::uint8_t kBlademaster = 5;
constexpr std::uint32_t kWeaponSize = 24, kWeaponRow = 8, kWeapons = 1150, kModels = 553, kClasses = 11;
constexpr std::uint32_t kRowsPerClass = 8, kRows = 40, kRowSize = 14;

} // namespace

std::optional<Sharpness> read_sharpness(const psprecomp::GuestMemory &memory, const SharpnessSymbols &symbols) {
    if (!memory.contains(symbols.game_sys, 4) || !memory.contains(symbols.cockpit, 4)) return std::nullopt;
    const std::uint32_t game_sys = memory.load32(symbols.game_sys);
    const std::uint32_t player = local_player(memory, game_sys, symbols.player_manager, kCurrent + 2);
    if (player == 0 || quest_menu_open(memory, game_sys, memory.load32(symbols.cockpit)) ||
        memory.load8(player + kWeaponType) != kBlademaster) return std::nullopt;

    Sharpness gauge{{}, memory.load16(player + kCurrent), memory.load16(player + kMaximum)};
    if (gauge.maximum == 0 || gauge.current > gauge.maximum) return std::nullopt;

    const std::uint32_t id = memory.load16(player + kWeaponId);
    if (id >= kWeapons || !memory.contains(symbols.weapons, kWeapons * kWeaponSize)) return std::nullopt;
    const std::uint32_t weapon = symbols.weapons + id * kWeaponSize;
    const std::uint32_t model = memory.load16(weapon);
    const std::uint32_t row = memory.load8(weapon + kWeaponRow);
    if (model >= kModels || row >= kRowsPerClass || !memory.contains(symbols.model_classes, kModels) ||
        !memory.contains(symbols.class_rows, kClasses * 4) || !memory.contains(symbols.rows, kRows * kRowSize)) return std::nullopt;

    const std::uint32_t weapon_class = memory.load8(symbols.model_classes + model);
    if (weapon_class >= kClasses) return std::nullopt;
    const std::uint32_t group = memory.load32(symbols.class_rows + weapon_class * 4);
    if (group < symbols.rows || group - symbols.rows > (kRows - kRowsPerClass) * kRowSize ||
        (group - symbols.rows) % (kRowsPerClass * kRowSize) != 0) return std::nullopt;
    const std::uint32_t address = group + row * kRowSize;
    std::uint16_t previous = 0;
    for (std::uint32_t tier = 0; tier < gauge.ends.size(); ++tier) {
        const std::uint16_t end = memory.load16(address + tier * 2);
        if (end < previous) return std::nullopt;
        gauge.ends[tier] = std::min(end, gauge.maximum);
        previous = end;
    }
    if (previous < gauge.maximum) return std::nullopt;
    return gauge;
}

} // namespace mhf::mhfu
