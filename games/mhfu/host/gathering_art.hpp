#pragma once

#include "gathering_logic.hpp"
#include "overlay/song_panel.hpp"
#include <array>

namespace mhf::mhfu {
struct GatheringImage {
    int size{};
    std::shared_ptr<const std::vector<std::uint32_t>> pixels;
};
// Cropped atlas images: frame, Circle, pickaxe, bug net, question mark, fishing rod.
enum class GatheringArtSlot : std::size_t { frame = 0, circle, pickaxe, bug_net, question, fishing, count };
using GatheringArt = std::array<GatheringImage, static_cast<std::size_t>(GatheringArtSlot::count)>;
std::uint8_t gathering_alpha(std::uint8_t previous, bool active);
std::vector<overlay::HintSprite> gathering_sprites(const GatheringArt &art, GatheringKind kind, std::uint8_t alpha,
                                                   bool button, bool unavailable = false);
} // namespace mhf::mhfu
