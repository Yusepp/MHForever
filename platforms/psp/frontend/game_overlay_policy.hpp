#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace mhf::psp {

enum class OverlayOwner { none, menu, dialog, notice, songs, game };
enum class GameOverlayAction { keep, show, hide };

// UI overlays and interactive placement have priority over a game's image.
constexpr GameOverlayAction game_overlay_action(OverlayOwner owner, bool wants_image, bool blocked) {
    if (owner != OverlayOwner::none && owner != OverlayOwner::game) return GameOverlayAction::keep;
    if (blocked || !wants_image) return owner == OverlayOwner::game ? GameOverlayAction::hide : GameOverlayAction::keep;
    return GameOverlayAction::show;
}

inline void blend_over(std::vector<std::uint32_t> &frame, const std::vector<std::uint32_t> &overlay) {
    for (std::size_t i = 0; i < frame.size() && i < overlay.size(); ++i) {
        const std::uint32_t o = overlay[i], a = o >> 24u;
        if (a == 0) continue;
        const std::uint32_t f = frame[i], inv = 255u - a;
        std::uint32_t out = 0xFF000000u;
        for (unsigned shift = 0; shift < 24; shift += 8) {
            const std::uint32_t c = ((o >> shift) & 0xFFu) + (((f >> shift) & 0xFFu) * inv + 127u) / 255u;
            out |= std::min(c, 255u) << shift;
        }
        frame[i] = out;
    }
}

template <typename Refresh>
void refresh_game_overlay(std::vector<std::uint32_t> &frame, const std::shared_ptr<const std::vector<std::uint32_t>> &image,
                          bool blocked, Refresh &&refresh) {
    std::forward<Refresh>(refresh)();
    if (!blocked && image) blend_over(frame, *image);
}

} // namespace mhf::psp
