#include "gathering_logic.hpp"

namespace mhf::mhfu {
namespace {

bool is_tool_for(GatheringKind kind, std::uint16_t id) noexcept {
    switch (kind) {
    case GatheringKind::mining:  return (id >= 0x90 && id <= 0x92) || id == 0x4c9;
    case GatheringKind::bugs:    return (id >= 0x94 && id <= 0x96) || id == 0x4ca;
    case GatheringKind::fishing: return id >= 0x88 && id <= 0x8f;
    default:                     return false;
    }
}

std::uint16_t virtual_tool_for(GatheringKind kind) noexcept {
    switch (kind) {
    case GatheringKind::mining:  return kMegaPickaxeId;
    case GatheringKind::bugs:    return kMegaBugNetId;
    case GatheringKind::fishing: return kMegaFishingFlyId;
    default:                     return 0;
    }
}

} // namespace

GatheringDecision decide_gathering(const GatheringInput &in) {
    GatheringDecision out;
    if (in.supply_box) {
        out.kind = GatheringKind::supply_box;
        out.available = !in.exhausted;
        return out;
    }
    if (in.delivery_box) {
        out.kind = GatheringKind::delivery_box;
        out.available = in.has_deliverable;
        return out;
    }
    if (!in.fishing && in.node_index >= 0xffff) return out;
    out.kind = in.fishing          ? GatheringKind::fishing
               : in.carving        ? GatheringKind::carve
               : in.node_type == 3 ? GatheringKind::mining
               : in.node_type == 4 ? GatheringKind::bugs
                                   : GatheringKind::gather;
    if (out.kind == GatheringKind::gather || out.kind == GatheringKind::carve) {
        out.available = !in.exhausted;
        return out;
    }
    if (in.infinite_tools) {
        out.uses_virtual_tool = true;
        out.virtual_tool_id = virtual_tool_for(out.kind);
        out.available = !in.exhausted && out.virtual_tool_id != 0;
        return out;
    }
    for (std::size_t i = 0; i < in.inventory.size() && i < 25; ++i) {
        const auto &item = in.inventory[i];
        if (is_tool_for(out.kind, item.id) && item.count) {
            out.tool_slot = static_cast<std::uint16_t>(i);
            break;
        }
    }
    out.available = !in.exhausted && out.tool_slot.has_value();
    return out;
}
} // namespace mhf::mhfu
