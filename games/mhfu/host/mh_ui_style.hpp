#pragma once

#include "overlay/painter.hpp"

namespace mhf::mhfu {

// Shared native MHFU HUD/menu palette (matching Pokke House / Smithy green ribbons & antique-brass frames).
inline constexpr mhf::overlay::Color kMhRibbonBg{0.114f, 0.251f, 0.078f};
inline constexpr mhf::overlay::Color kMhBrassLight{0.804f, 0.651f, 0.373f};
inline constexpr mhf::overlay::Color kMhBrassDark{0.463f, 0.349f, 0.188f};
inline constexpr mhf::overlay::Color kMhFrameInk{0.055f, 0.047f, 0.039f};
inline constexpr mhf::overlay::Color kMhHeaderYellow{0.925f, 0.871f, 0.341f};
inline constexpr mhf::overlay::Color kMhWhite{0.960f, 0.955f, 0.935f};

} // namespace mhf::mhfu
