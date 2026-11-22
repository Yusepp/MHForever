#pragma once

#include "psprecomp/guest_memory.hpp"

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

// MHFU item IDs (verified from MHFU/MHP2G item tables, identical across EU, US, JP, FUComplete).
inline constexpr std::uint16_t kItemPaintball = 64u;       // 0x0040
inline constexpr std::uint16_t kItemWhetstone = 153u;      // 0x0099
inline constexpr std::uint16_t kItemMiniWhetstone = 867u;  // 0x0363

// Keeps Whetstones and Paintballs infinite during quests (restoring consumed uses and supplying a
// temporary item when none were packed, removed cleanly upon returning to the village).
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

    void remove_temp_item(psprecomp::GuestMemory &memory, std::uint32_t game_sys,
                          std::optional<std::size_t> &slot, std::uint16_t item_id) noexcept;

    InfiniteConsumablesSymbols symbols_;
    std::array<SlotSnapshot, kPouchSlots> prev_slots_{};
    std::optional<std::size_t> temp_whetstone_slot_;
    std::optional<std::size_t> temp_paintball_slot_;
    bool in_quest_{false};
};

} // namespace mhf::mhfu
