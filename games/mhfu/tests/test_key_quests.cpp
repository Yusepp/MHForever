// Synthetic quest headers and layouts only: no game data or localized quest names.
#include "key_quests.hpp"
#include "check.hpp"

#include <algorithm>
#include <limits>

using namespace mhf::mhfu;

int main() {
    psprecomp::GuestMemory memory;
    constexpr std::uint32_t record = 0x08810000, layout = 0x08811000;
    memory.store16(layout + 0x18, 232);
    memory.store32(record, 0x04000004);
    auto marker = key_quest_marker(memory, record, layout, 62);
    MHF_CHECK(marker.has_value());
    if (!marker) return mhf::test::finish("key_quests");
    MHF_CHECK((marker == QuestMarker{220, 64}));
    // Completion and the language's title bytes do not change classification.
    memory.store32(record + 0x18, 1001);
    memory.store8(record + 0x34, 0xE3);
    MHF_CHECK(key_quest_marker(memory, record, layout, 62) == marker);
    memory.store32(record, 4);
    MHF_CHECK(!key_quest_marker(memory, record, layout, 62));
    memory.store32(record, 0x04000004);
    for (const std::uint32_t bad : {0u, record + 1, 0x04000000u, 0x09FFFFF0u, std::numeric_limits<std::uint32_t>::max()}) {
        MHF_CHECK(!key_quest_marker(memory, bad, layout, 62));
        MHF_CHECK(!key_quest_marker(memory, record, bad, 62));
    }
    MHF_CHECK(!key_quest_marker(memory, record, layout, -20));
    MHF_CHECK(!key_quest_marker(memory, record, layout, 265));
    memory.store16(layout + 0x18, 5);
    MHF_CHECK(!key_quest_marker(memory, record, layout, 62));
    memory.store16(layout + 0x18, 232);

    KeyQuestMarkers pages;
    MHF_CHECK(!pages.marker());
    pages.begin();
    pages.capture(memory, record, layout, 62);
    MHF_CHECK(!pages.marker()); // publish a complete UI pass, never half of the next page
    pages.end();
    MHF_CHECK(pages.marker() == marker);
    pages.begin();
    memory.store32(record, 4); // paging to an ordinary quest
    pages.capture(memory, record, layout, 62);
    pages.end();
    MHF_CHECK(!pages.marker());
    pages.begin();
    memory.store32(record, 0x04000004);
    pages.capture(memory, record, layout, 62);
    pages.end();
    MHF_CHECK(pages.marker() == marker);
    pages.begin(); // rank menu or returning to the village: no title was drawn
    pages.end();
    MHF_CHECK(!pages.marker());
    pages.capture(memory, record, layout, 62); // outside a UI pass: ignored
    pages.end();
    MHF_CHECK(!pages.marker());
    pages.begin();
    pages.capture(memory, record, layout, 62);
    pages.end();
    pages.clear(); // loading another overlay
    MHF_CHECK(!pages.marker());

    const auto pixels = key_quest_pixels(*marker);
    MHF_CHECK(pixels.size() == 480u * 272u);
    if (pixels.size() != 480u * 272u) return mhf::test::finish("key_quests");
    MHF_CHECK(std::ranges::any_of(pixels, [](auto p) { return p == 0xFF5ACFEFu; }));
    for (int y = 0; y < 272; ++y) {
        for (int x = 0; x < 480; ++x) {
            if (x < marker->x || x >= marker->x + 10 || y < marker->y || y >= marker->y + 10)
                MHF_CHECK(pixels[static_cast<std::size_t>(y * 480 + x)] == 0);
        }
    }
    return mhf::test::finish("key_quests");
}
