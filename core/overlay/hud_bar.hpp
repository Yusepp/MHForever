#pragma once

#include <array>
#include <vector>

namespace mhf::overlay {

// A segmented bar in logical HUD coordinates, anchored to the top left like the game's status bars.
struct HudBar {
    struct Segment {
        float end; // end of the segment as a fraction of the track
        std::array<float, 3> rgb;
        bool operator==(const Segment &) const = default;
    };
    float x, y, width, height, fill;
    std::vector<Segment> segments;
    bool operator==(const HudBar &) const = default;
};

} // namespace mhf::overlay
