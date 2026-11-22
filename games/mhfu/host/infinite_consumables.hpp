#pragma once

#include "psprecomp/memory.hpp"

#include <array>
#include <cstdint>
#include <optional>

namespace mhf::mhfu {

struct InfiniteConsumablesSymbols {
    std::optional<std::uint32_t> game_sys; // GameSys* pointer in guest RAM

    [[nodiscard]] constexpr bool complete() const noexcept {
        return game_sys.has_value();
    }
};

// MHFU item IDs (verified identical across EU, US, JP, FUComplete).
inline constexpr std::uint16_t kItemWhetstone = 153u;
inline constexpr std::uint16_t kItemPaintball = 156u;

// Keeps Whetstones (World/Rise/Wilds-style infinite sharpening) and Paintballs (carrying >= 1 = infinite)
// from being consumed during quests without creating permanent items in the village.
class InfiniteConsumables {
public:
    explicit InfiniteConsumables(InfiniteConsumablesSymbols symbols = {}) noexcept;

    void poll(psprecomp::GuestMemory &memory, bool infinite_whetstone, bool infinite_paintballs);
    void clear(psprecomp::GuestMemory *memory = nullptr) noexcept;

private:
    static constexpr std::size_t kPouchSlots = 24u;
    struct SlotSnapshot {
        std::uint16_t id{0};
        std::uint16_t count{0};
    };

    void remove_temp_whetstone(psprecomp::GuestMemory &memory, std::uint32_t game_sys) noexcept;

    InfiniteConsumablesSymbols symbols_;
    std::array<SlotSnapshot, kPouchSlots> prev_slots_{};
    std::optional<std::size_t> temp_whetstone_slot_;
    bool in_quest_{false};
};

} // namespace mhf::mhfu
