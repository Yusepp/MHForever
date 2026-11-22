#include "gathering_art.hpp"
#include <algorithm>

namespace mhf::mhfu {
namespace {

GatheringArtSlot icon_slot_for(GatheringKind kind) noexcept {
    switch (kind) {
    case GatheringKind::mining:  return GatheringArtSlot::pickaxe;
    case GatheringKind::bugs:    return GatheringArtSlot::bug_net;
    case GatheringKind::fishing: return GatheringArtSlot::fishing;
    default:                     return GatheringArtSlot::question;
    }
}

// Procedural 32x32 premultiplied-RGBA camp chest icon (blue = Supply Box, red = Delivery Box), no game assets.
std::shared_ptr<const std::vector<std::uint32_t>> make_chest_icon(bool red) {
    auto pixels = std::make_shared<std::vector<std::uint32_t>>(32u * 32u, 0u);
    auto rgba = [](std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a = 255u) -> std::uint32_t {
        return static_cast<std::uint32_t>(r) | (static_cast<std::uint32_t>(g) << 8u) |
               (static_cast<std::uint32_t>(b) << 16u) | (static_cast<std::uint32_t>(a) << 24u);
    };
    const std::uint32_t outline = rgba(20, 16, 16);
    const std::uint32_t lid_hi = red ? rgba(238, 86, 76) : rgba(82, 166, 245);
    const std::uint32_t lid_mid = red ? rgba(208, 48, 42) : rgba(48, 122, 214);
    const std::uint32_t body = red ? rgba(162, 30, 28) : rgba(30, 86, 172);
    const std::uint32_t body_lo = red ? rgba(122, 22, 22) : rgba(20, 62, 134);
    const std::uint32_t strap = rgba(206, 166, 74);
    const std::uint32_t clasp_hi = rgba(252, 222, 120);
    const std::uint32_t clasp_lo = rgba(168, 124, 42);

    for (int y = 7; y <= 25; ++y) {
        for (int x = 4; x <= 27; ++x) {
            // Chamfered top corners so the chest silhouette reads cleanly inside the circular HUD frame.
            const bool corner = (y == 7 && (x <= 5 || x >= 26)) || (y == 8 && (x == 4 || x == 27)) ||
                                (y == 25 && (x == 4 || x == 27));
            if (corner) continue;
            const bool border = y == 7 || y == 25 || x == 4 || x == 27 ||
                                (y == 8 && (x == 5 || x == 26)) || (y == 14);
            std::uint32_t c = outline;
            if (!border) {
                if (y <= 10) c = lid_hi;
                else if (y < 14) c = lid_mid;
                else if (y <= 22) c = body;
                else c = body_lo;
                if (x == 9 || x == 10 || x == 21 || x == 22) c = strap;
            }
            (*pixels)[static_cast<std::size_t>(y * 32 + x)] = c;
        }
    }
    // Center antique-gold clasp & keyhole
    for (int y = 12; y <= 17; ++y) {
        for (int x = 13; x <= 18; ++x) {
            const bool edge = y == 12 || y == 17 || x == 13 || x == 18;
            const bool keyhole = (x == 15 || x == 16) && (y == 14 || y == 15);
            (*pixels)[static_cast<std::size_t>(y * 32 + x)] =
                edge ? outline : (keyhole ? clasp_lo : (y == 13 ? clasp_hi : strap));
        }
    }
    return pixels;
}

} // namespace

std::uint8_t gathering_alpha(std::uint8_t previous, bool active) {
    if (active) return static_cast<std::uint8_t>(previous == 0 ? 15 : std::min(255, int(previous) + 64));
    return static_cast<std::uint8_t>(previous == 0 ? 0 : std::max(15, int(previous) - 64));
}
std::vector<overlay::HintSprite> gathering_sprites(const GatheringArt &art, GatheringKind kind, std::uint8_t alpha,
                                                   bool button, bool unavailable) {
    if (!alpha) return {};
    std::vector<overlay::HintSprite> sprites;
    auto add = [&](GatheringArtSlot slot, float x, float y, float size, float opacity) {
        const auto &img = art[static_cast<std::size_t>(slot)];
        sprites.push_back({x, y, size, opacity, img.size, img.pixels});
    };
    add(GatheringArtSlot::frame, 225, 80, 32, float(alpha) / 255.0f);
    if (alpha == 255 && kind != GatheringKind::none) {
        if (kind == GatheringKind::supply_box || kind == GatheringKind::delivery_box) {
            static const auto blue_chest = make_chest_icon(false);
            static const auto red_chest = make_chest_icon(true);
            sprites.push_back({229, 84, 24, 1, 32, kind == GatheringKind::delivery_box ? red_chest : blue_chest});
        } else {
            add(icon_slot_for(kind), 229, 84, 24, 1);
        }
        if (unavailable) {
            // Premultiplied sprite: a red X with a dark edge, generated without game
            // assets.
            static const auto cross = [] {
                auto pixels = std::make_shared<std::vector<std::uint32_t>>(32 * 32, 0);
                for (int y = 3; y < 29; ++y)
                    for (int x = 3; x < 29; ++x) {
                        const int distance = std::min(std::abs(x - y), std::abs(x + y - 31));
                        if (distance <= 3) (*pixels)[y * 32 + x] = distance <= 1 ? 0xff3030f0u : 0xff101018u;
                    }
                return std::shared_ptr<const std::vector<std::uint32_t>>(pixels);
            }();
            sprites.push_back({229, 84, 24, 1, 32, cross});
        } else if (button)
            add(GatheringArtSlot::circle, 219, 100, 15, 1);
    }
    return sprites;
}
} // namespace mhf::mhfu
