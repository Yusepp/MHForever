#pragma once

#include "psprecomp/guest_memory.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace mhf::mhfu {

struct ItemLoadoutsSymbols {
    std::optional<std::uint32_t> game_sys; // GameSys* pointer in guest RAM

    [[nodiscard]] constexpr bool complete() const noexcept {
        return game_sys.has_value();
    }
};

struct ItemSlot {
    std::uint16_t id{0};
    std::uint16_t count{0};
};

inline constexpr std::size_t kItemPouchSlots = 24u;
inline constexpr std::size_t kItemBoxSlots = 1000u;
inline constexpr std::size_t kItemLoadoutCount = 5u;

struct ItemLoadoutPreset {
    bool saved{false};
    std::array<ItemSlot, kItemPouchSlots> slots{};
};

struct LoadoutResult {
    bool ok{false};
    std::string message;
};

// Manages up to 5 named Item Pouch loadouts (Village / Gathering Hall only, never mid-quest).
// Transfers items cleanly between the hunter's Pouch and Item Box without duplicating or losing items.
class ItemLoadouts {
public:
    explicit ItemLoadouts(ItemLoadoutsSymbols symbols = {}, std::string storage_path = {}) noexcept;

    void set_storage_path(std::string path);
    void load_from_disk();
    void save_to_disk() const;

    // True when the hunter is in Village / Gathering Hall (not on an active quest or post-quest countdown).
    [[nodiscard]] bool can_use_loadouts(const psprecomp::GuestMemory &memory) const noexcept;
    [[nodiscard]] bool has_preset(std::size_t index) const noexcept;
    [[nodiscard]] const ItemLoadoutPreset &preset(std::size_t index) const noexcept;

    // Saves current pouch into preset `index` (0..4).
    LoadoutResult save_preset(const psprecomp::GuestMemory &memory, std::size_t index);
    // Deposits current pouch into Item Box, then withdraws preset `index` (0..4) from Item Box.
    LoadoutResult apply_preset(psprecomp::GuestMemory &memory, std::size_t index) const;

private:
    ItemLoadoutsSymbols symbols_;
    std::string storage_path_;
    std::array<ItemLoadoutPreset, kItemLoadoutCount> presets_{};
};

} // namespace mhf::mhfu
