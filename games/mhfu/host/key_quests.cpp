#include "key_quests.hpp"

namespace mhf::mhfu {
namespace {

// The selected quest's fixed header and the details panel's layout have the same fields in all four releases.
constexpr std::uint32_t kKeyQuest = 0x04000000u, kHeaderSize = 0x34, kPanelLeft = 0x18;

bool ram_contains(const psprecomp::GuestMemory &memory, std::uint32_t address, std::uint32_t size) {
    const auto physical = psprecomp::GuestMemory::canonical(address);
    if (physical < psprecomp::GuestMemory::kPhysicalBase) return false;
    const auto offset = physical - psprecomp::GuestMemory::kPhysicalBase;
    return offset <= memory.size() && size <= memory.size() - offset;
}

// An original 8-pixel key: a ring, a shaft and two teeth. The surrounding pixel is its dark outline.
bool gold(int x, int y) {
    if (x < 0 || x >= 8 || y < 0 || y >= 8) return false;
    const int radius = (x - 2) * (x - 2) + (y - 2) * (y - 2);
    return (radius >= 2 && radius <= 5) || (x >= 4 && y == 2) || ((x == 5 || x == 7) && y >= 2 && y <= 4);
}

} // namespace

std::optional<QuestMarker> key_quest_marker(const psprecomp::GuestMemory &memory, std::uint32_t record,
                                          std::uint32_t layout, int title_y) {
    if ((record & 3u) != 0 || (layout & 1u) != 0 || !ram_contains(memory, record, kHeaderSize) ||
        !ram_contains(memory, layout, 0x20) || title_y < 0 || title_y > 260) return std::nullopt;
    if ((memory.load32(record) & kKeyQuest) == 0) return std::nullopt;
    const int left = memory.load16(layout + kPanelLeft);
    if (left < 12 || left > 480) return std::nullopt;
    // Outside the panel's left edge: even the longest localized title keeps all of its space.
    return QuestMarker{left - 12, title_y + 2};
}

std::vector<std::uint32_t> key_quest_pixels(QuestMarker marker) {
    std::vector<std::uint32_t> pixels(480u * 272u);
    if (marker.x < 0 || marker.x > 470 || marker.y < 0 || marker.y > 262) return pixels;
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            bool outline = false;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) outline |= gold(x - 1 + dx, y - 1 + dy);
            }
            pixels[static_cast<std::size_t>((marker.y + y) * 480 + marker.x + x)] =
                gold(x - 1, y - 1) ? 0xFF5ACFEFu : outline ? 0xFF24201Cu : 0u;
        }
    }
    return pixels;
}

void KeyQuestMarkers::begin() {
    pending_.reset();
    drawing_ = true;
}

void KeyQuestMarkers::capture(const psprecomp::GuestMemory &memory, std::uint32_t record, std::uint32_t layout, int title_y) {
    if (drawing_) pending_ = key_quest_marker(memory, record, layout, title_y);
}

void KeyQuestMarkers::end() {
    if (drawing_) current_ = pending_;
    drawing_ = false;
}

void KeyQuestMarkers::clear() {
    pending_.reset();
    current_.reset();
    drawing_ = false;
}

} // namespace mhf::mhfu
