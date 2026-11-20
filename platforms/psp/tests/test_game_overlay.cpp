#include "game_overlay_policy.hpp"
#include "check.hpp"

#include <initializer_list>

using namespace mhf::psp;

int main() {
    // A game may show its marker only when no higher-priority view owns the overlay.
    MHF_CHECK(game_overlay_action(OverlayOwner::none, true, false) == GameOverlayAction::show);
    MHF_CHECK(game_overlay_action(OverlayOwner::game, true, false) == GameOverlayAction::show);
    for (auto owner : {OverlayOwner::menu, OverlayOwner::dialog, OverlayOwner::notice, OverlayOwner::songs}) {
        MHF_CHECK(game_overlay_action(owner, true, false) == GameOverlayAction::keep);
        MHF_CHECK(game_overlay_action(owner, false, true) == GameOverlayAction::keep);
    }
    // When placement starts, release an existing marker; don't reclaim it until placement ends.
    MHF_CHECK(game_overlay_action(OverlayOwner::game, true, true) == GameOverlayAction::hide);
    MHF_CHECK(game_overlay_action(OverlayOwner::none, true, true) == GameOverlayAction::keep);
    MHF_CHECK(game_overlay_action(OverlayOwner::none, true, false) == GameOverlayAction::show);
    MHF_CHECK(game_overlay_action(OverlayOwner::game, false, false) == GameOverlayAction::hide);
    MHF_CHECK(game_overlay_action(OverlayOwner::none, false, false) == GameOverlayAction::keep);
    // A page changes on the same poll that presents a software frame: its old key must not be blended.
    std::vector<std::uint32_t> frame{0xFF000000u};
    std::shared_ptr<const std::vector<std::uint32_t>> image = std::make_shared<const std::vector<std::uint32_t>>(1, 0xFFFFFFFFu);
    unsigned calls = 0;
    refresh_game_overlay(frame, image, false, [&] { image.reset(); ++calls; });
    MHF_CHECK(calls == 1 && frame[0] == 0xFF000000u);
    image = std::make_shared<const std::vector<std::uint32_t>>(1, 0xFFFFFFFFu);
    refresh_game_overlay(frame, image, true, [&] { ++calls; });
    MHF_CHECK(calls == 2 && frame[0] == 0xFF000000u); // a menu covers the game
    return mhf::test::finish("game_overlay");
}
