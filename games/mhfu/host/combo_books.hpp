#pragma once

#include "psprecomp/guest_memory.hpp"

#include <cstdint>
#include <optional>

namespace mhf::mhfu {

struct ComboBooksSymbols {
    std::optional<std::uint32_t> game_sys; // GameSys* pointer in guest RAM

    [[nodiscard]] constexpr bool complete() const noexcept {
        return game_sys.has_value();
    }
};

// Book of Combos 1..5 item IDs in MHFU (verified identical across EU, US, JP, FUComplete).
inline constexpr std::uint16_t kComboBookFirstId = 1u;
inline constexpr std::uint16_t kComboBookLastId = 5u;

// Returns true when all 5 Books of Combos (IDs 1..5) are owned across the hunter's Pouch and Item Box.
[[nodiscard]] bool owns_all_five_combo_books(const psprecomp::GuestMemory &memory, ComboBooksSymbols symbols) noexcept;

// Applies the 100% combine rate override if enabled, rate >= 0, and all 5 Books of Combos are owned.
[[nodiscard]] std::uint32_t apply_combo_books_rate(const psprecomp::GuestMemory &memory, ComboBooksSymbols symbols,
                                                   std::uint32_t current_rate, bool enabled) noexcept;

} // namespace mhf::mhfu
