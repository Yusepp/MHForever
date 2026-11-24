#pragma once

#include "gathering_art.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace mhf::psp::textures {
class Pack;
}

namespace mhf::mhfu {
struct LoadoutSprite {
    int size{0};
    std::shared_ptr<const std::vector<std::uint32_t>> pixels;

    [[nodiscard]] bool valid() const noexcept {
        return size > 0 && pixels && pixels->size() == static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
    }
};

// Native MHFU item and equipment icons cropped from the player's own DATA.BIN (or active HD texture pack).
struct LoadoutIcons {
    LoadoutSprite pouch;
    LoadoutSprite chest;
    LoadoutSprite potion;
    LoadoutSprite book;
    LoadoutSprite bomb;
    LoadoutSprite trap;
    LoadoutSprite whetstone;
    LoadoutSprite pickaxe;
    LoadoutSprite paintball;
    LoadoutSprite ammo;
    LoadoutSprite weapon_blade;
    LoadoutSprite weapon_gun;
    // Per-class 16x16 native weapon icons indexed by MHFU weapon class (0..10):
    // 0:GS, 1:HBG, 2:Hammer, 3:Lance, 4:SnS, 5:LBG, 6:DB, 7:LS, 8:HH, 9:GL, 10:Bow.
    std::array<LoadoutSprite, 11u> weapon_classes{};
    LoadoutSprite head;
    LoadoutSprite chest_armor;
    LoadoutSprite arms;
    LoadoutSprite waist;
    LoadoutSprite legs;
};

// Original swizzled HUD/items TMH atlases, read from the player's own game.
std::optional<GatheringArt> decode_game_gathering_art(std::span<const std::uint8_t> hud,
                                                      std::span<const std::uint8_t> items,
                                                      const psp::textures::Pack *pack = nullptr,
                                                      unsigned *replaced = nullptr);
std::optional<GatheringArt> load_game_gathering_art(const std::filesystem::path &iso,
                                                    const std::filesystem::path &executable, std::uint32_t hud_file,
                                                    std::uint32_t items_file, const psp::textures::Pack *pack = nullptr,
                                                    unsigned *replaced = nullptr);
// Master localized text archive (item, weapon, and armor names) read from the player's own DATA.BIN.
std::vector<std::uint8_t> load_game_text_archive(const std::filesystem::path &iso,
                                                 const std::filesystem::path &executable);
// Native item & equipment icon sprites cropped from the player's own DATA.BIN item atlas (or HD texture pack).
std::optional<LoadoutIcons> decode_game_loadout_icons(std::span<const std::uint8_t> items,
                                                      const psp::textures::Pack *pack = nullptr);
std::optional<LoadoutIcons> load_game_loadout_icons(const std::filesystem::path &iso,
                                                    const std::filesystem::path &executable, std::uint32_t items_file,
                                                    const psp::textures::Pack *pack = nullptr);
} // namespace mhf::mhfu
