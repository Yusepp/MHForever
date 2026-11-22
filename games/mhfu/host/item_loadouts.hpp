#pragma once

#include "mhf/psp/host.hpp"
#include "overlay/painter.hpp"
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

// One 12-byte equipment record in MHFU (type, level/upgrade, model/id, 3 decoration slots).
using EquipPiece = std::array<std::uint8_t, 12u>;

inline constexpr std::size_t kItemPouchSlots = 24u;
inline constexpr std::size_t kItemBoxSlots = 1000u;
inline constexpr std::size_t kEquipSlots = 6u;       // Weapon, Legs, Head, Chest, Arms, Waist
inline constexpr std::size_t kEquipBoxSlots = 1000u;
inline constexpr std::size_t kItemLoadoutCount = 5u;

struct ItemLoadoutPreset {
    bool saved{false};
    std::array<ItemSlot, kItemPouchSlots> slots{};
};

struct ArmorLoadoutPreset {
    bool saved{false};
    std::array<std::uint16_t, kEquipSlots> box_indices{}; // 0xFFFF = empty
    std::array<EquipPiece, kEquipSlots> pieces{};
};

struct LoadoutResult {
    bool ok{false};
    std::string message;
};

// Manages Hunter Loadouts (5 Item Pouch Sets + 5 Armor/Equipment Sets) in Village / Gathering Hall.
// Integrated directly into the House & Guild Hall Item Box with an MH-styled TrueType overlay UI.
class ItemLoadouts {
public:
    enum class Tab { Items = 0, Armor = 1 };

    explicit ItemLoadouts(ItemLoadoutsSymbols symbols = {}, std::string storage_path = {}) noexcept;

    void set_storage_path(std::string path);
    void load_from_disk();
    void save_to_disk() const;

    // True when the hunter is in Village / Gathering Hall (not on an active quest or post-quest countdown).
    [[nodiscard]] bool can_use_loadouts(const psprecomp::GuestMemory &memory) const noexcept;
    [[nodiscard]] bool has_preset(std::size_t index) const noexcept;
    [[nodiscard]] bool has_armor_preset(std::size_t index) const noexcept;
    [[nodiscard]] const ItemLoadoutPreset &preset(std::size_t index) const noexcept;
    [[nodiscard]] const ArmorLoadoutPreset &armor_preset(std::size_t index) const noexcept;

    // Item Pouch Sets (0..4)
    LoadoutResult save_preset(const psprecomp::GuestMemory &memory, std::size_t index);
    LoadoutResult apply_preset(psprecomp::GuestMemory &memory, std::size_t index) const;

    // Armor & Weapon Sets (0..4)
    LoadoutResult save_armor_preset(const psprecomp::GuestMemory &memory, std::size_t index);
    LoadoutResult apply_armor_preset(psprecomp::GuestMemory &memory, std::size_t index) const;

    // House & Guild Hall Item Box integration (driven by mh_item_box_update hook).
    void mark_box_open(std::uint64_t now_us) noexcept;
    [[nodiscard]] bool box_open(std::uint64_t now_us) const noexcept;
    [[nodiscard]] bool panel_open() const noexcept { return panel_open_; }
    [[nodiscard]] bool suppress_game_input() const noexcept { return panel_open_ || release_guard_; }
    void poll_box_ui(psprecomp::GuestMemory &memory, std::uint32_t buttons, std::uint64_t now_us, bool enabled);
    // Draws the MH-styled [SELECT] Loadouts pill or full two-column Item & Armor Sets visualizer using TrueType Painter.
    bool render_box_ui(mhf::overlay::Painter &painter, const psprecomp::GuestMemory &memory, std::uint64_t now_us) const;

private:
    [[nodiscard]] std::string armor_storage_path() const;

    ItemLoadoutsSymbols symbols_;
    std::string storage_path_;
    std::array<ItemLoadoutPreset, kItemLoadoutCount> presets_{};
    std::array<ArmorLoadoutPreset, kItemLoadoutCount> armor_presets_{};
    std::uint64_t box_seen_us_{0};
    std::uint32_t prev_buttons_{0};
    std::size_t selected_{0};
    Tab tab_{Tab::Items};
    bool panel_open_{false};
    bool release_guard_{false};
    std::string toast_;
    bool toast_ok_{true};
    std::uint64_t toast_until_us_{0};
};

} // namespace mhf::mhfu
