#pragma once

#include <cstdint>
#include <optional>
#include <span>

namespace mhf::mhfu {
enum class GatheringKind { none, gather, mining, bugs, fishing, carve, supply_box, delivery_box };

// High-tier MHFU tool/bait IDs used for virtual (tool-free) gathering.
constexpr std::uint16_t kMegaPickaxeId = 0x92;
constexpr std::uint16_t kMegaBugNetId = 0x96;
constexpr std::uint16_t kMegaFishingFlyId = 0x8F;

struct GatheringItem {
    std::uint16_t id{}, count{};
};
struct GatheringInput {
    std::uint32_t node_index = 0xffff;
    std::uint16_t node_type{};
    std::span<const GatheringItem> inventory;
    bool exhausted = false, carving = false, fishing = false, infinite_tools = false;
    bool supply_box = false, delivery_box = false, has_deliverable = false;
};
struct GatheringDecision {
    GatheringKind kind = GatheringKind::none;
    std::optional<std::uint16_t> tool_slot;
    std::uint16_t virtual_tool_id = 0;
    bool uses_virtual_tool = false;
    bool available = false;
};
GatheringDecision decide_gathering(const GatheringInput &input);
} // namespace mhf::mhfu
