#pragma once

#include "mhf/psp/host.hpp"
#include "psprecomp/guest_memory.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

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

    // House & Guild Hall Item Box integration (driven by mh_item_box_update hook).
    void mark_box_open(std::uint64_t now_us) noexcept;
    [[nodiscard]] bool box_open(std::uint64_t now_us) const noexcept;
    [[nodiscard]] bool panel_open() const noexcept { return panel_open_; }
    void poll_box_ui(psprecomp::GuestMemory &memory, std::uint32_t buttons, std::uint64_t now_us, bool enabled);
    // Draws the MH-styled [SELECT] ITEM SETS badge or full Item Sets panel onto a 480x272 premultiplied RGBA overlay.
    // Returns true if anything was drawn.
    bool render_box_ui(std::vector<std::uint32_t> &rgba, std::uint64_t now_us) const noexcept;

private:
    ItemLoadoutsSymbols symbols_;
    std::string storage_path_;
    std::array<ItemLoadoutPreset, kItemLoadoutCount> presets_{};
    std::uint64_t box_seen_us_{0};
    std::uint32_t prev_buttons_{0};
    std::size_t selected_{0};
    bool panel_open_{false};
    std::string toast_;
    std::uint64_t toast_until_us_{0};
};

} // namespace mhf::mhfu
