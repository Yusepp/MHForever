#pragma once

#include "psprecomp/guest_memory.hpp"
#include <cstdint>
#include <optional>
#include <vector>

namespace mhf::mhfu {

struct QuestMarker {
    int x, y;
    bool operator==(const QuestMarker &) const = default;
};
std::optional<QuestMarker> key_quest_marker(const psprecomp::GuestMemory &memory, std::uint32_t record,
                                          std::uint32_t layout, int title_y);
std::vector<std::uint32_t> key_quest_pixels(QuestMarker marker);

class KeyQuestMarkers {
public:
    void begin();
    void capture(const psprecomp::GuestMemory &memory, std::uint32_t record, std::uint32_t layout, int title_y);
    void end();
    void clear();
    std::optional<QuestMarker> marker() const { return current_; }
private:
    bool drawing_ = false;
    std::optional<QuestMarker> pending_, current_;
};

} // namespace mhf::mhfu
