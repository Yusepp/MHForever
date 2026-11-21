#pragma once

#include "psprecomp/guest_memory.hpp"

#include <array>
#include <cstdint>
#include <optional>

namespace mhf::mhfu {

struct SharpnessSymbols {
    std::uint32_t game_sys, player_manager, cockpit;
    std::uint32_t weapons, model_classes, class_rows, rows;
};

struct Sharpness {
    std::array<std::uint16_t, 7> ends; // cumulative tier ends in the game's order, clipped to maximum
    std::uint16_t current, maximum;
    bool operator==(const Sharpness &) const = default;
};

// Read-only. The caller checks that the hunt overlay is active before following the local player.
std::optional<Sharpness> read_sharpness(const psprecomp::GuestMemory &memory, const SharpnessSymbols &symbols);

} // namespace mhf::mhfu
